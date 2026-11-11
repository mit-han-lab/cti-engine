from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from scripts.environment.contract import TorchMemorySaverLock
from tests.support.checkpoints import QWEN3_0_6B
from tests.support.gpu import (
    ApiServerProcess,
    find_forbidden_log_patterns,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.graph_captures import capture_line
from tests.support.paths import REPO_ROOT, source_pythonpath

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)


def _tms_component() -> None:
    from importlib.metadata import distribution

    import torch
    from torch_memory_saver import TorchMemorySaver

    assert torch.cuda.get_allocator_backend() == "native"
    saver = TorchMemorySaver()
    paused_tags: set[str] = set()

    def assert_physical_transition(delta: int) -> None:
        # cudaMemGetInfo is device-global. Discrete GPUs normally expose the
        # allocator's 20 MiB segment delta, while GB10 unified memory reports a
        # smaller physical transition. Zero still identifies a pool that was not
        # paused; pointer, content, and graph checks below prove logical isolation.
        assert delta > 0, delta

    def pause(tag: str) -> int:
        torch.cuda.synchronize()
        free_before = torch.cuda.mem_get_info()[0]
        saver.pause(tag)
        paused_tags.add(tag)
        torch.cuda.synchronize()
        return torch.cuda.mem_get_info()[0] - free_before

    def resume(tag: str) -> int:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_before = torch.cuda.mem_get_info()[0]
        saver.resume(tag)
        paused_tags.remove(tag)
        torch.cuda.synchronize()
        return free_before - torch.cuda.mem_get_info()[0]

    try:
        # A 9 MiB allocation leaves 11 MiB in the native allocator's 20 MiB
        # medium segment. A provider that shares one MemPool across tags will pack
        # both 5 MiB B allocations into A's segment, making pause(B) release zero.
        with saver.region(tag="tag-a", enable_cpu_backup=False):
            tag_a = torch.full(
                (9 * 1024 * 1024 // 4,),
                11.0,
                device="cuda",
                dtype=torch.float32,
            )
        with saver.region(tag="tag-b", enable_cpu_backup=False):
            tag_b_input = torch.full(
                (5 * 1024 * 1024 // 4,),
                7.0,
                device="cuda",
                dtype=torch.float32,
            )
            tag_b_output = torch.empty_like(tag_b_input)

        pointers = {
            "tag_a": (tag_a.data_ptr(), tag_a.untyped_storage().data_ptr()),
            "tag_b_input": (
                tag_b_input.data_ptr(),
                tag_b_input.untyped_storage().data_ptr(),
            ),
            "tag_b_output": (
                tag_b_output.data_ptr(),
                tag_b_output.untyped_storage().data_ptr(),
            ),
        }
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            warmup_scratch = tag_b_input + 1.0
            tag_b_output.copy_(warmup_scratch)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()
        del warmup_scratch

        graph = torch.cuda.CUDAGraph()
        with saver.cuda_graph(graph, tag="graph"):
            graph_scratch = tag_b_input + 1.0
            tag_b_output.copy_(graph_scratch)
        graph.replay()
        torch.cuda.synchronize()
        assert float(tag_b_output[12345]) == 8.0

        tag_b_release_bytes = pause("tag-b")
        assert_physical_transition(tag_b_release_bytes)
        assert float(tag_a[12345]) == 11.0
        assert (tag_a.data_ptr(), tag_a.untyped_storage().data_ptr()) == pointers["tag_a"]

        tag_b_restore_bytes = resume("tag-b")
        assert_physical_transition(tag_b_restore_bytes)
        assert (
            tag_b_input.data_ptr(),
            tag_b_input.untyped_storage().data_ptr(),
        ) == pointers["tag_b_input"]
        assert (
            tag_b_output.data_ptr(),
            tag_b_output.untyped_storage().data_ptr(),
        ) == pointers["tag_b_output"]
        tag_b_input.fill_(3.0)
        graph.replay()
        torch.cuda.synchronize()
        assert float(tag_b_output[12345]) == 4.0

        tag_a_release_bytes = pause("tag-a")
        assert_physical_transition(tag_a_release_bytes)
        tag_b_input.fill_(5.0)
        graph.replay()
        torch.cuda.synchronize()
        assert float(tag_b_output[12345]) == 6.0

        tag_a_restore_bytes = resume("tag-a")
        assert_physical_transition(tag_a_restore_bytes)
        assert (tag_a.data_ptr(), tag_a.untyped_storage().data_ptr()) == pointers["tag_a"]
        tag_a.fill_(13.0)
        torch.cuda.synchronize()
        assert float(tag_a[12345]) == 13.0

        graph_release_bytes = pause("graph")
        assert graph_release_bytes > 0
        _ = resume("graph")
        tag_b_input.fill_(17.0)
        graph.replay()
        torch.cuda.synchronize()
        assert float(tag_b_output[12345]) == 18.0

        # Backup policy is part of physical ownership even when the logical tag is
        # identical. The shared-pool build packs both backup tensors into the
        # no-backup segment, so their contents cannot survive the transition.
        with saver.region(tag="policy", enable_cpu_backup=False):
            policy_no_backup = torch.full(
                (9 * 1024 * 1024 // 4,),
                19.0,
                device="cuda",
                dtype=torch.float32,
            )
        with saver.region(tag="policy", enable_cpu_backup=True):
            policy_backup_a = torch.full(
                (5 * 1024 * 1024 // 4,),
                23.0,
                device="cuda",
                dtype=torch.float32,
            )
            policy_backup_b = torch.full_like(policy_backup_a, 29.0)
        policy_pointers = (
            policy_no_backup.data_ptr(),
            policy_backup_a.data_ptr(),
            policy_backup_b.data_ptr(),
        )
        policy_release_bytes = pause("policy")
        assert_physical_transition(policy_release_bytes)
        policy_restore_bytes = resume("policy")
        assert_physical_transition(policy_restore_bytes)
        assert (
            policy_no_backup.data_ptr(),
            policy_backup_a.data_ptr(),
            policy_backup_b.data_ptr(),
        ) == policy_pointers
        assert float(policy_backup_a[12345]) == 23.0
        assert float(policy_backup_b[12345]) == 29.0
        policy_no_backup.fill_(31.0)
    finally:
        for tag in tuple(paused_tags):
            saver.resume(tag)

    direct_url_text = distribution("torch-memory-saver").read_text("direct_url.json")
    assert direct_url_text is not None
    direct_url = json.loads(direct_url_text)
    print(
        json.dumps(
            {
                "allocator": torch.cuda.get_allocator_backend(),
                "commit": direct_url["vcs_info"]["commit_id"],
                "graph": "pass",
                "tag_a_release_bytes": tag_a_release_bytes,
                "tag_a_restore_bytes": tag_a_restore_bytes,
                "tag_b_release_bytes": tag_b_release_bytes,
                "tag_b_restore_bytes": tag_b_restore_bytes,
                "policy_release_bytes": policy_release_bytes,
                "policy_restore_bytes": policy_restore_bytes,
            }
        )
    )


def _component_command(component: str) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "tests.gpu.integration.test_environment_stack",
        "--component",
        component,
    )


def test_locked_tms_cuda13_tag_isolation_and_graph(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "tms")
    site_packages = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    preload = site_packages / "torch_memory_saver_hook_mode_preload_cu13.abi3.so"
    assert preload.is_file()

    env["LD_PRELOAD"] = str(preload)
    env.pop("PYTORCH_ALLOC_CONF", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    output = run_isolated_process(
        _component_command("tms"), log_root / "tms.log", env=env, cwd=REPO_ROOT
    )

    receipt = json.loads(output.strip().splitlines()[-1])
    assert receipt["allocator"] == "native"
    assert receipt["commit"] == TorchMemorySaverLock.load().commit
    assert receipt["graph"] == "pass"
    for field in (
        "tag_a_release_bytes",
        "tag_a_restore_bytes",
        "tag_b_release_bytes",
        "tag_b_restore_bytes",
        "policy_release_bytes",
        "policy_restore_bytes",
    ):
        assert receipt[field] > 0


def _model_snapshot() -> Path:
    return QWEN3_0_6B.snapshot().resolve()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_json(url: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        assert response.status == 200
        return json.loads(response.read())


def _read_sse_json(url: str, *, payload: dict[str, Any]) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks = []
    with urllib.request.urlopen(request, timeout=180) as response:
        assert response.status == 200
        for raw_line in response:
            if not raw_line.startswith(b"data: "):
                continue
            data = raw_line.removeprefix(b"data: ").strip()
            if data == b"[DONE]":
                break
            chunks.append(json.loads(data))
    return chunks


def _assert_usage(
    response: dict[str, Any],
    *,
    prompt_tokens: int,
    completion_tokens: int | None = None,
) -> None:
    usage = response["usage"]
    assert usage["prompt_tokens"] == prompt_tokens
    assert usage["completion_tokens"] > 0
    if completion_tokens is not None:
        assert usage["completion_tokens"] == completion_tokens
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def _wait_for_server(process: subprocess.Popen[bytes], port: int, log_path: Path) -> dict[str, Any]:
    deadline = time.monotonic() + 180
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(
                f"API server exited before readiness with {process.returncode}\n"
                f"{log_path.read_text(encoding='utf-8')}"
            )
        try:
            return _read_json(url)
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(0.25)
    pytest.fail(f"API server did not become ready\n{log_path.read_text(encoding='utf-8')}")


def _cross_token_stop(decode: Any, token_ids: list[int]) -> tuple[str, int]:
    """A stop string of two characters on each side of a token boundary in the decoded
    ``token_ids``, found nowhere earlier, and the index of the token completing it."""

    text = decode(token_ids)
    for boundary in range(2, len(token_ids) - 1):
        start = len(decode(token_ids[:boundary])) - 2
        stop = text[start : start + 4]
        if len(stop) == 4 and "\ufffd" not in stop and text.find(stop) == start:
            end = next(k for k in range(len(token_ids)) if stop in decode(token_ids[: k + 1]))
            return stop, end
    pytest.fail(f"no cross-token stop string in {text!r}")


def _check_completions_with_stop(url: str, served_model: str, decode: Any) -> dict[str, Any]:
    """Greedy /v1/completions with a cross-token stop: whole == stream == decode(ids) cut
    at the stop, which ends the engine request at the token completing it."""

    prompt = "The capital of France is"
    base = {
        "model": served_model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 24,
        "ignore_eos": True,
    }
    reference = _read_json(url, payload={**base, "return_token_ids": True})
    stop, end = _cross_token_stop(decode, reference["choices"][0]["token_ids"])
    ids = reference["choices"][0]["token_ids"][: end + 1]
    body = {**base, "stop": [stop], "return_token_ids": True, "echo": True}
    whole = _read_json(url, payload=body)["choices"][0]
    chunks = _read_sse_json(url, payload={**body, "stream": True})
    included = _read_json(url, payload={**body, "include_stop_str_in_output": True})["choices"][0]
    # A request's seed repeats its sampled output.
    seeded = {**base, "temperature": 0.7, "seed": 1234, "return_token_ids": True}
    first, again = (_read_json(url, payload=seeded)["choices"][0]["token_ids"] for _ in range(2))

    cut = decode(ids)[: decode(ids).index(stop)]
    assert (whole["text"], whole["token_ids"], whole["finish_reason"]) == (
        prompt + cut,
        ids,
        "stop",
    )
    choices = [chunk["choices"][0] for chunk in chunks]
    assert "".join(choice["text"] for choice in choices) == whole["text"]
    assert [t for choice in choices for t in choice["token_ids"]] == ids
    assert choices[-1]["finish_reason"] == "stop"
    assert included["text"] == prompt + cut + stop
    assert first == again
    return {"stop": stop, "whole": whole, "included": included, "seeded": first}


def test_qwen_api_flashinfer_graph_and_clean_shutdown(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "api")
    log_path = log_root / "qwen-api-smoke.log"
    models_path = log_root / "models.json"
    chat_path = log_root / "chat.json"
    sampling_chat_path = log_root / "chat-sampling.json"
    stream_chat_path = log_root / "chat-stream.json"
    completions_path = log_root / "completions-stop.json"
    chat_recipe_path = log_root / "chat-recipe.log"
    port = _free_port()
    model_snapshot = _model_snapshot()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_snapshot, local_files_only=True)
    chat_messages = [{"role": "user", "content": "Reply with exactly OK."}]
    sampling_messages = [{"role": "user", "content": "Name one color."}]
    stream_messages = [{"role": "user", "content": "Say hi."}]

    def prompt_tokens(messages: list[dict[str, str]]) -> int:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert isinstance(prompt, str)
        return len(tokenizer.encode(prompt, add_special_tokens=False))

    model_receipt = {
        "repo_id": QWEN3_0_6B.repo_id,
        "revision": QWEN3_0_6B.revision,
        "snapshot": str(model_snapshot),
    }
    (log_root / "model-receipt.json").write_text(
        json.dumps(model_receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    env["PYTHONPATH"] = source_pythonpath()

    command = (
        sys.executable,
        "-m",
        "cti",
        "--model.path",
        str(model_snapshot),
        "--model.max-seq-len",
        "4096",
        "--cache.prefix",
        "naive",
        "--scheduling.max-running-requests",
        "8",
        "--scheduling.max-batch-tokens",
        "4096",
        "--memory.inference-ratio",
        "0.8",
        "--memory.peak-ratio",
        "0.8",
        "--server.host",
        "127.0.0.1",
        "--server.port",
        str(port),
    )

    with ApiServerProcess(command, env, log_path) as server:
        process = server.process
        models = _wait_for_server(process, port, log_path)
        served_model = models["data"][0]["id"]
        chat_recipe = subprocess.run(
            (
                sys.executable,
                str(REPO_ROOT / "cookbook/chat.py"),
                "--base-url",
                f"http://127.0.0.1:{port}/v1",
                "--max-tokens",
                "2",
                "--temperature",
                "0",
            ),
            cwd=REPO_ROOT,
            env=env,
            input="Say hi.\n/exit\n",
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        chat_recipe_log = f"stdout:\n{chat_recipe.stdout}\nstderr:\n{chat_recipe.stderr}"
        chat_recipe_path.write_text(chat_recipe_log, encoding="utf-8")
        assert chat_recipe.returncode == 0, chat_recipe_log
        assert chat_recipe.stdout.count("$ ") == 2, chat_recipe_log
        assert chat_recipe.stdout.replace("$ ", "").strip(), chat_recipe_log
        assert not find_forbidden_log_patterns(chat_recipe_log), chat_recipe_log
        chat = _read_json(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            payload={
                "model": served_model,
                "messages": chat_messages,
                "temperature": 0,
                "top_k": 1,
                "max_tokens": 8,
                "stream": False,
            },
        )
        # The first token comes from prefill; forcing two tokens exercises bs=1 graph replay.
        sampling_chat = _read_json(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            payload={
                "model": served_model,
                "messages": sampling_messages,
                "temperature": 1.0,
                "top_k": 2,
                "top_p": 1.0,
                "ignore_eos": True,
                "max_tokens": 2,
                "stream": False,
            },
        )
        stream_chunks = _read_sse_json(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            payload={
                "model": served_model,
                "messages": stream_messages,
                "temperature": 0,
                "top_k": 1,
                "ignore_eos": True,
                "max_tokens": 2,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        completions = _check_completions_with_stop(
            f"http://127.0.0.1:{port}/v1/completions",
            served_model,
            lambda ids: tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ),
        )
        completions_path.write_text(json.dumps(completions, indent=2) + "\n", encoding="utf-8")
        models_path.write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
        chat_path.write_text(json.dumps(chat, indent=2) + "\n", encoding="utf-8")
        sampling_chat_path.write_text(
            json.dumps(sampling_chat, indent=2) + "\n",
            encoding="utf-8",
        )
        stream_chat_path.write_text(
            json.dumps(stream_chunks, indent=2) + "\n",
            encoding="utf-8",
        )

    log = log_path.read_text(encoding="utf-8")
    assert server.clean_shutdown, log
    assert process.returncode == 0, log
    assert models["data"][0]["id"] == str(model_snapshot)
    assert chat["object"] == "chat.completion"
    assert len(chat["choices"]) == 1
    choice = chat["choices"][0]
    assert choice["finish_reason"] in {"stop", "length"}
    message = choice["message"]
    assert message["role"] == "assistant"
    assert isinstance(message["content"], str) and message["content"]
    _assert_usage(chat, prompt_tokens=prompt_tokens(chat_messages))
    assert sampling_chat["object"] == "chat.completion"
    assert len(sampling_chat["choices"]) == 1
    sampled_choice = sampling_chat["choices"][0]
    assert sampled_choice["finish_reason"] in {"stop", "length"}
    sampled_message = sampled_choice["message"]
    assert sampled_message["role"] == "assistant"
    assert isinstance(sampled_message["content"], str)
    _assert_usage(
        sampling_chat,
        prompt_tokens=prompt_tokens(sampling_messages),
        completion_tokens=2,
    )
    assert stream_chunks
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in stream_chunks)
    assert all("usage" in chunk for chunk in stream_chunks)
    assert all(chunk["usage"] is None for chunk in stream_chunks[:-1])
    stream_usage_chunks = [chunk for chunk in stream_chunks if chunk["usage"] is not None]
    assert len(stream_usage_chunks) == 1
    stream_usage = stream_usage_chunks[0]
    assert stream_usage["choices"] == []
    _assert_usage(
        stream_usage,
        prompt_tokens=prompt_tokens(stream_messages),
        completion_tokens=2,
    )
    assert capture_line("decode") in log
    assert "selected=fi" in log or "Selected attention backend fi" in log
    assert "Inference engine stopped" in log
    assert "Application shutdown complete" in log
    assert not find_forbidden_log_patterns(log)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", required=True, choices=("tms",))
    parser.parse_args()
    _tms_component()


if __name__ == "__main__":
    main()
