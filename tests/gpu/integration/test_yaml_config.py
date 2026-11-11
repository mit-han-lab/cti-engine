from __future__ import annotations

import json
import math
import re
import socket
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from cti.utils import get_device_capability
from tests.support.checkpoints import QWEN3_0_6B
from tests.support.gpu import (
    assert_ranks_reclaimed,
    engine_rank_pids,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.graph_captures import graph_captures
from tests.support.inference_correctness import GOLDEN_CASES
from tests.support.paths import REPO_ROOT, source_pythonpath

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)

RECEIPT_PREFIX = "CTI_YAML_CONFIG="
_CAPACITY_LOG = re.compile(r"Inference capacity: (\d+) KV pages")
_SELECTION_LOG = re.compile(r"Selected attention backend (\S+): ([^\n;]+)")
_TRAINED_LORA_NAMES = (
    "model.layers.0.self_attn.qkv_proj.q.lora_A",
    "model.layers.0.self_attn.qkv_proj.q.lora_B",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_json(url: str, *, payload: dict[str, object] | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(
        url,
        data=body,
        headers={} if body is None else {"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urlopen(request, timeout=60) as response:
        assert response.status == 200
        document = json.load(response)
    assert isinstance(document, dict)
    return document


def _write_config(path: Path, *, model_path: str, port: int) -> None:
    path.write_text(
        f"""
model:
  path: {json.dumps(model_path)}
  max_seq_len: 512
  initialization_seed: 17
memory:
  inference_ratio: 0.4
  peak_ratio: 0.5
scheduling:
  max_running_requests: 2
  max_batch_tokens: 64
cache:
  prefix: naive
lora:
  targets: [attention, mlp]
  rank: 4
  alpha: 8.0
training:
  trainable_plan:
    selected_names:
      - {_TRAINED_LORA_NAMES[0]}
      - {_TRAINED_LORA_NAMES[1]}
  optimizer:
    lr: 0.001
    weight_decay: 0.0
    max_grad_norm: 1.0
  batching:
    layout: padded
    policy:
      kind: static_batching
      microbatch_size: 2
      max_tokens_per_sample: 8
server:
  host: 127.0.0.1
  port: {port}
  num_tokenizers: 1
""".lstrip(),
        encoding="utf-8",
    )


def _run_child(tmp_path: Path) -> None:
    import torch

    from cti.llm import LLM
    from cti.protocol import SamplingParams
    from cti.server import OnlineServer
    from cti.tokenizer.text import load_tokenizer
    from tests.support.training import PAD_TOKEN_ID, RAW_SAMPLES, padded_sft_batch

    assert not torch.cuda.is_initialized()
    model_path = str(QWEN3_0_6B.snapshot())
    config_path = tmp_path / "shared-runtime.yaml"
    _write_config(config_path, model_path=model_path, port=_free_port())
    prompts = [list(case.prompt_ids) for case in GOLDEN_CASES]
    sampling = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_new_tokens=4,
    )
    messages = [{"role": "user", "content": "Reply briefly."}]
    tokenizer = load_tokenizer(model_path)
    rendered_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(rendered_chat, str)
    chat_prompt_ids = tokenizer.encode(rendered_chat, add_special_tokens=False)
    assert isinstance(chat_prompt_ids, list) and chat_prompt_ids

    offline = LLM.from_yaml(config_path)
    offline_ranks = engine_rank_pids()
    assert len(offline_ranks) == 1
    try:
        offline_outputs = offline.generate(prompts, sampling)
        offline_tokens = [output["token_ids"] for output in offline_outputs]
        assert offline_tokens == [list(case.output_ids[:4]) for case in GOLDEN_CASES]
        [offline_chat] = offline.generate([chat_prompt_ids], sampling)
        offline_chat_text = offline_chat["text"]
        offline_chat_token_count = len(offline_chat["token_ids"])
        assert offline_chat_text
        assert offline_chat_token_count == 4
    finally:
        offline.shutdown()
        offline.shutdown()
    assert_ranks_reclaimed(offline_ranks)
    assert not torch.cuda.is_initialized()

    chat_contents: list[str] = []
    server = OnlineServer.from_yaml(config_path)
    with server:
        online_ranks = engine_rank_pids()
        assert len(online_ranks) == 1

        base_url = f"http://{server.host}:{server.port}/v1"
        models = _read_json(f"{base_url}/models")
        assert [model["id"] for model in models["data"]] == [model_path]

        def chat() -> dict:
            response = _read_json(
                f"{base_url}/chat/completions",
                payload={
                    "model": model_path,
                    "messages": messages,
                    "temperature": 0.0,
                    "top_k": 1,
                    "ignore_eos": True,
                    "max_tokens": 4,
                    "stream": False,
                },
            )
            assert response["object"] == "chat.completion"
            choices = response["choices"]
            assert len(choices) == 1
            content = choices[0]["message"]["content"]
            assert isinstance(content, str) and content
            assert response["usage"]["completion_tokens"] == 4
            chat_contents.append(content)
            return response

        initial_chat = chat()
        assert initial_chat["choices"][0]["message"]["content"] == offline_chat_text
        assert initial_chat["usage"] == {
            "prompt_tokens": len(chat_prompt_ids),
            "completion_tokens": offline_chat_token_count,
            "total_tokens": len(chat_prompt_ids) + offline_chat_token_count,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
        training_batch = padded_sft_batch(RAW_SAMPLES[:2], PAD_TOKEN_ID)
        with server.training_session() as session:
            completion_future = session.forward_backward(training_batch)
            step = session.optim_step().result()
            session.publish()
            completion = completion_future.result()
        assert completion.normalization_count == training_batch.num_targets == 4
        assert math.isfinite(completion.loss_sum)
        assert step.step == server.model_revision == 1
        assert step.normalization_count == training_batch.num_targets
        assert math.isfinite(step.grad_norm)

        # Closing the local training session must restore serving residency.
        resumed_chat = chat()
        assert resumed_chat["usage"]["completion_tokens"] == 4
    server.close()
    assert_ranks_reclaimed(online_ranks)
    assert not torch.cuda.is_initialized()

    print(
        RECEIPT_PREFIX
        + json.dumps(
            {
                "chat_output_count": len(chat_contents),
                "chat_outputs_nonempty": all(chat_contents),
                "model": {"repo_id": QWEN3_0_6B.repo_id, "revision": QWEN3_0_6B.revision},
                "model_revision": 1,
                "offline_golden_tokens": offline_tokens,
                "offline_online_chat_match": chat_contents[0] == offline_chat_text,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def test_yaml_entrypoints_share_one_bounded_runtime_contract(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "yaml-config")
    env["PYTHONPATH"] = source_pythonpath(REPO_ROOT)
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    output = run_isolated_process(
        (
            sys.executable,
            "-m",
            "tests.gpu.integration.test_yaml_config",
            str(tmp_path),
        ),
        log_root / "yaml-config.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=420,
    )
    receipts = [
        json.loads(line.removeprefix(RECEIPT_PREFIX))
        for line in output.splitlines()
        if line.startswith(RECEIPT_PREFIX)
    ]
    assert len(receipts) == 1
    receipt = receipts[0]
    # Launch files name no backend or graph buckets: both entry points take the backend this
    # device's own rule picks (FlashInfer on SM120, FlashAttention ragged with FlashInfer
    # decode on SM90), not an explicit one.
    selections = _SELECTION_LOG.findall(output)
    assert len(selections) == 2 and len(set(selections)) == 1, selections
    assert selections[0][1].startswith(f"{get_device_capability()} uses "), selections
    assert graph_captures(output, "decode", [1, 2]) == 2
    # Both entry points resolve the launch file's naive cache and plan more than one page.
    assert output.count("Using the naive prefix cache") == 2
    capacities = [int(pages) for pages in _CAPACITY_LOG.findall(output)]
    assert len(capacities) == 2 and min(capacities) > 1
    # Only the online supervisor emits this message; both child exits are checked above.
    assert output.count("Inference engine stopped") == 1
    assert receipt["model"] == {"repo_id": QWEN3_0_6B.repo_id, "revision": QWEN3_0_6B.revision}
    assert receipt["offline_golden_tokens"] == [list(case.output_ids[:4]) for case in GOLDEN_CASES]
    assert receipt["offline_online_chat_match"] is True
    assert receipt["chat_output_count"] == 2
    assert receipt["chat_outputs_nonempty"] is True
    assert receipt["model_revision"] == 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_yaml_config.py TMP_PATH")
    _run_child(Path(sys.argv[1]))
