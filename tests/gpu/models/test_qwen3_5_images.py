"""Qwen3.5 image prompts through the public ``LLM`` against the FP32 Transformers truth, on
Qwen3.5-9B and Qwen3.5-35B-A3B. Two chat sets: a long prompt whose first chunk ends inside its
image, two images in one prompt, a wide and a noise image; and one image asked several
questions, the later ones loading the encoder's cached rows, then each on an image of its own.
Each set free-runs greedy in engine processes of its own: under the naive prefix cache, and
under the radix cache serving every batch twice, whose second pass must hit the prefixes the
first published. With an encoder pool of about one image, five images under three questions in
one batch, some repeats loading held rows and others encoded again after eviction, must follow
the same prompts with uncached images. A large image split across prefill chunks must follow it
unsplit. The truth, teacher-forced on every run's own sequences, judges them as the text prefix
gate judges a cache hit: no cached, radix or split pass leaves its top-1 where it is decided
more than ``DISAGREEMENT_SLACK`` times beyond the plain pass, and every pass's teacher-forced
top-1 log-probabilities are finite. Online, the same image as a data URL and as an http URL
must give the offline prompt and greedy tokens. Under the pinned Qwen3.5-9B
DFlash draft, greedy outputs must follow the plain engine's under the DFlash parity rule, with
the truth's margins on the plain engine's outputs. Skip without CUDA or the cached checkpoint;
a missing DFlash pair fails.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.request import Request, urlopen

import pytest

from tests.support.checkpoints import QWEN3_5_9B, QWEN3_5_9B_DFLASH, Checkpoint
from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    offline_child_env,
    require_device_memory,
    run_isolated_process,
)
from tests.support.hf_judge import DISAGREEMENT_SLACK, NEAR_TIE_BF16_SPACINGS, NEAR_TIE_NATS
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5_gpu import (
    PINNED_CHECKPOINTS,
    max_running_requests,
    receipts_dir,
    selected_checkpoint,
    write_receipt,
)
from tests.support.qwen3_5_prompts import (
    MAX_IMAGE_TOKENS,
    eviction_recipes,
    image_chats,
    processed_image_prompts,
    question_recipes,
)
from tests.support.reference_truth import (
    SequenceTruth,
    free_run_flips,
    teacher_forced_top1,
)

pytestmark = pytest.mark.checkpoints(QWEN3_5_9B, QWEN3_5_9B_DFLASH)

if TYPE_CHECKING:
    from cti.multimodal.processor import ProcessedPrompt

RECEIPT_PREFIX = "CTI_QWEN35_IMAGES="
GREEDY_TOKENS = 32
MAX_SEQ_LEN = 4096
_CHECKPOINTS = [
    checkpoint
    for checkpoint in PINNED_CHECKPOINTS
    if checkpoint.repo_id in ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-35B-A3B")
]
# The BF16 35B-A3B needs an RTX PRO 6000.
_MOE_MIN_GIB = 80
# Chat sets by name; None is ``IMAGE_RECIPES``.
_RECIPES = {
    "images": None,
    "questions": question_recipes(),
    "eviction": eviction_recipes(),
    "split": {"large": ("", ("shapes",), "Describe every shape in this picture and its color.")},
}
# The split chat's image, 1792 x 1792 pixels: 3136 tokens, within the 16384 of Qwen3.5's image
# preprocessing.
_SPLIT_IMAGE_SIZE = 1792
_SPLIT_IMAGE_TOKENS = 16384
# Engine arms by prefix cache, and how many times each serves every chat.
_PASSES = {"naive": 1, "radix": 2}
_CHILD_TIMEOUT_SECONDS = 1800
_INFERENCE_MEMORY_RATIO = 0.85
# Fixed batches: greedy near ties flip with batch composition.
_BATCHES = (("image-after-long-text",), ("one-image", "two-images"), ("wide-image", "noise-image"))
# DFlash rows, as the DFlash parity check runs them.
_DFLASH_SLOTS = 4


# ============================== Engine ==============================


def _chats(recipes: str) -> dict[str, Any]:
    """The chats of ``recipes``; the split chat's image enlarged to ``_SPLIT_IMAGE_SIZE``."""

    chats = image_chats(_RECIPES[recipes])
    if recipes == "split":
        [message] = chats["large"]
        image = message["content"][0]
        image["image"] = image["image"].resize((_SPLIT_IMAGE_SIZE,) * 2)
    return chats


def _image_tokens(recipes: str) -> int:
    return _SPLIT_IMAGE_TOKENS if recipes == "split" else MAX_IMAGE_TOKENS


def _processed(snapshot: Path, recipes: str) -> dict[str, ProcessedPrompt]:
    from transformers import AutoConfig

    from cti.multimodal import ImageLimits
    from cti.tokenizer import PromptEncoder

    if recipes != "split":
        return processed_image_prompts(snapshot, _RECIPES[recipes])
    encoder = PromptEncoder.from_checkpoint(
        str(snapshot),
        hf_config=AutoConfig.from_pretrained(snapshot, local_files_only=True),
        limits=ImageLimits(max_images=1, max_image_tokens=_SPLIT_IMAGE_TOKENS),
    )
    prompts = {
        name: encoder.chat(messages, template_kwargs={"enable_thinking": False})
        for name, messages in _chats(recipes).items()
    }
    for prompt in prompts.values():
        encoder.check(prompt)  # as a request is checked before it reaches the engine
    return prompts


def _schedule(
    recipes: str, prefix: str, prompts: dict[str, ProcessedPrompt], split_tokens: int | None
) -> tuple[list[dict[str, ProcessedPrompt]], tuple[tuple[str, ...], ...], int]:
    """What one arm serves: each pass's prompts, the batches of every pass, and the step's
    token budget (``split_tokens`` for the split chat)."""

    max_running_req = max_running_requests()
    largest = max(span.num_tokens for prompt in prompts.values() for span in prompt.images.spans)
    if recipes == "eviction":
        # Uncached, then cached, in one overlapped batch each, under an encoder pool of about
        # one image: rows = max(max_batch_tokens, max_image_tokens), the image bound being the
        # sequence bound ``_run_arm`` sets.
        uncached = {name: _uncached(name, prompt) for name, prompt in prompts.items()}
        return [uncached, prompts], (tuple(prompts),), largest + max_running_req
    if recipes == "split":
        if largest != (_SPLIT_IMAGE_SIZE // 32) ** 2 or split_tokens is None:
            raise RuntimeError(
                f"the split arm needs a step budget and a 3136-token image: {largest}"
            )
        return [prompts], (tuple(prompts),), split_tokens
    passes = [prompts] * _PASSES[prefix]
    if recipes == "questions":
        # The first question encodes the image and the next three load it; then images of
        # their own make room in the encoder's rows.
        names = tuple(prompts)
        return passes, (names[:1], names[1:4], names[4:]), 1024
    long_image = prompts["image-after-long-text"].images.spans[0]
    # The first chunk of the long prompt would end half way through its image.
    max_batch_tokens = long_image.offset + long_image.num_tokens // 2
    # The long text must put its image far enough in that this budget still encodes every
    # image whole beside a decode token per running request.
    if largest > max_batch_tokens - max_running_req:
        raise RuntimeError(
            f"the long prompt's image starts at {long_image.offset}, too early for a "
            f"{max_batch_tokens}-token step to hold a {largest}-token image beside "
            f"{max_running_req} running requests; lengthen _LONG_TEXT"
        )
    return passes, _BATCHES, max_batch_tokens


def _uncached(name: str, prompt: ProcessedPrompt) -> ProcessedPrompt:
    """``prompt`` with its images keyed by ``name``, so no cache holds them; its pixels and so
    its truth are unchanged."""

    import dataclasses
    import hashlib

    images = prompt.images
    spans = tuple(
        dataclasses.replace(span, digest=hashlib.sha256(f"{name}/{i}".encode()).digest())
        for i, span in enumerate(images.spans)
    )
    return dataclasses.replace(prompt, images=dataclasses.replace(images, spans=spans))


def _run_arm(arguments: argparse.Namespace) -> None:
    """One engine arm: greedy over one chat set, every pass of ``_schedule`` on one engine."""

    import torch

    from cti.config import (
        CacheOptions,
        EngineOptions,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    prompts = _processed(arguments.snapshot, arguments.recipes)
    served, batches, max_batch_tokens = _schedule(
        arguments.recipes, arguments.prefix, prompts, arguments.split_tokens
    )
    # Every prompt and its continuation must fit the sequence.
    longest = max(len(prompt.input_ids) for prompt in prompts.values()) + GREEDY_TOKENS
    if longest > MAX_SEQ_LEN:
        raise RuntimeError(f"a prompt and its continuation exceed MAX_SEQ_LEN ({MAX_SEQ_LEN})")
    # max_image_tokens = min(the processor's limit, max_seq_len): the eviction arm's longest
    # prompt and its continuation, about one image, bounds its encoder pool.
    max_seq_len = longest if arguments.recipes == "eviction" else MAX_SEQ_LEN
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=str(arguments.snapshot), max_seq_len=max_seq_len),
            memory=MemoryOptions(inference_ratio=_INFERENCE_MEMORY_RATIO),
            scheduling=SchedulingOptions(
                max_running_requests=max_running_requests(), max_batch_tokens=max_batch_tokens
            ),
            cache=CacheOptions(prefix=arguments.prefix),
        )
    )
    passes = []
    try:
        params = SamplingParams(
            temperature=0, ignore_eos=True, max_new_tokens=GREEDY_TOKENS, logprobs=True
        )
        for pass_prompts in served:
            samples = {}
            for names in batches:
                result = llm.sample([pass_prompts[name] for name in names], params)
                samples.update(zip(names, result.samples, strict=True))
            passes.append(
                {
                    name: {
                        "prompt_ids": list(sample.prompt_token_ids),
                        "tokens": list(sample.sampled_token_ids),
                        "logprobs": list(sample.sampling_logprobs),
                        "cached": sample.cached_prompt_tokens,
                    }
                    for name, sample in samples.items()
                }
            )
    finally:
        llm.shutdown()
    receipt = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "passes": passes,
    }
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))


def _arm(
    snapshot: Path,
    tmp_path: Path,
    recipes: str,
    prefix: str,
    prompts: dict[str, Any],
    split_tokens: int | None = None,
) -> dict[str, Any]:
    """An engine arm's receipt, its passes checked against the chat set's ``_image_prompts``
    and their samples given their prompt's key."""

    component = f"qwen3_5-{recipes}-{prefix}"
    budget = ()
    if split_tokens is not None:
        component += f"-{split_tokens}"
        budget = ("--split-tokens", str(split_tokens))
    env, log_root = offline_child_env(tmp_path, component)
    output = run_isolated_process(
        (
            sys.executable,
            "-m",
            "tests.gpu.models.test_qwen3_5_images",
            *("--snapshot", str(snapshot), "--recipes", recipes, "--prefix", prefix),
            *budget,
        ),
        log_root / "engine.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=_CHILD_TIMEOUT_SECONDS,
    )
    run = child_receipt(output, RECEIPT_PREFIX)
    assert_cuda_device_metadata(run)
    for samples in run["passes"]:
        _check(samples, prompts, (recipes, prefix))
    return run


# ============================== Truth ==============================


def _image_prompts(
    snapshot: Path, recipes: str = "images"
) -> tuple[dict[str, Any], dict[Any, dict[str, Any]]]:
    """Each chat's prompt key, its ids and its images' digest, since same-sized images give
    the same ids; and per key Transformers' own image inputs for the truth."""

    from tests.support.hf_truth import content_digest
    from tests.support.qwen3_5_hf_reference import prepare_image_inputs

    chats = _chats(recipes)
    prompts = _processed(snapshot, recipes)
    images = {
        name: prepare_image_inputs(
            snapshot, chats[name], prompt, max_image_tokens=_image_tokens(recipes)
        )
        for name, prompt in prompts.items()
    }
    keys = {
        name: (tuple(prompt.input_ids.tolist()), content_digest(images[name]["originals"]))
        for name, prompt in prompts.items()
    }
    return keys, {keys[name]: inputs for name, inputs in images.items()}


def _check(samples: dict[str, Any], prompts: dict[str, Any], arm: Any) -> None:
    """One pass of an arm served every chat of ``prompts`` in full; each sample takes its
    prompt's key."""

    assert samples.keys() == prompts.keys(), arm
    for name, sample in samples.items():
        assert tuple(sample["prompt_ids"]) == prompts[name][0], (arm, name)
        assert len(sample["tokens"]) == GREEDY_TOKENS, (arm, name)
        sample["prompt"] = prompts[name]


def _key(sample: dict[str, Any]) -> tuple[Any, tuple[int, ...]]:
    return sample["prompt"], tuple(sample["tokens"])


def _judge(
    checkpoint: Checkpoint,
    snapshot: Path,
    tmp_path: Path,
    name: str,
    runs: dict[str, dict[str, Any]],
    passes: dict[str, tuple[dict[str, Any], str]],
    images: dict[tuple[int, ...], Any],
) -> None:
    """The truth on every sequence of ``passes``, required once, judged as the text gates
    are: each pass leaves its decided top-1 at most ``DISAGREEMENT_SLACK`` times more than
    its reference pass (``passes`` maps a pass to its samples and its reference's name; the
    plain engine's BF16 rounding sets the reference), with finite teacher-forced top-1
    log-probabilities, whose statistics go in the receipt."""

    samples = {label: list(chats.values()) for label, (chats, _) in passes.items()}
    sequences = {_key(sample) for arm in samples.values() for sample in arm}
    job = (receipts_dir() or tmp_path) / f"qwen3_5-{name}-truth-{checkpoint.slug}.pt"
    truth, truth_report = SequenceTruth(checkpoint, snapshot, tmp_path).require(
        sequences, job, images
    )
    flips = {
        arm: free_run_flips(truth, map(_key, arm_samples)) for arm, arm_samples in samples.items()
    }
    top1 = {
        arm: teacher_forced_top1(
            [truth[_key(sample)] for sample in arm_samples],
            [token for sample in arm_samples for token in sample["tokens"]],
            [logprob for sample in arm_samples for logprob in sample["logprobs"]],
        )
        for arm, arm_samples in samples.items()
    }
    write_receipt(
        f"qwen3_5_{name}-{checkpoint.slug}",
        {
            "near_tie_nats": NEAR_TIE_NATS,
            "near_tie_bf16_spacings": NEAR_TIE_BF16_SPACINGS,
            "truth": truth_report,
            "flips": flips,
            "teacher_forced_top1": top1,
            "runs": runs,
        },
    )

    for label, (_, reference) in passes.items():
        bound = flips[reference]["decided_disagreements"] + DISAGREEMENT_SLACK
        assert flips[label]["decided_disagreements"] <= bound, (label, flips)
        assert top1[label]["finite"], (label, top1)


# ============================== Online ==============================


def _openai_chat(chat: str, urls: list[str]) -> list[dict[str, Any]]:
    """An image chat with its images, in order, as OpenAI image URL parts."""

    [message] = image_chats()[chat]
    images = iter(urls)
    content = [
        {"type": "image_url", "image_url": {"url": next(images)}}
        if part["type"] == "image"
        else part
        for part in message["content"]
    ]
    return [{"role": "user", "content": content}]


def _pngs(chat: str) -> list[bytes]:
    [message] = image_chats()[chat]
    pngs = []
    for part in message["content"]:
        if part["type"] == "image":
            buffer = io.BytesIO()
            part["image"].save(buffer, format="PNG")
            pngs.append(buffer.getvalue())
    return pngs


def _data_url(png: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(png).decode()}"


def _serve_image(png: bytes) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


_ONLINE_CHATS = ("one-image", "two-images")


def _online_answers(snapshot: Path) -> dict[str, Any]:
    """The one- and two-image chats offline through ``LLM``, then online with data URLs, and
    the one-image chat with an http URL: each run's prompt and greedy tokens."""

    from cti.config import (
        CacheOptions,
        EngineOptions,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
        ServerOptions,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams
    from cti.server import OnlineServer

    options = EngineOptions(
        model=ModelOptions(path=str(snapshot), max_seq_len=MAX_SEQ_LEN),
        memory=MemoryOptions(inference_ratio=_INFERENCE_MEMORY_RATIO),
        scheduling=SchedulingOptions(max_running_requests=max_running_requests()),
        cache=CacheOptions(prefix="naive"),
    )
    chats = {chat: [_data_url(png) for png in _pngs(chat)] for chat in _ONLINE_CHATS}

    llm = LLM(options)
    offline = {}
    try:
        params = SamplingParams(temperature=0, ignore_eos=True, max_new_tokens=GREEDY_TOKENS)
        for chat, urls in chats.items():
            prompt = llm.prompt_encoder().chat(_openai_chat(chat, urls))
            [output] = llm.generate([prompt], params)
            offline[chat] = {
                "prompt_ids": prompt.input_ids.tolist(),
                "tokens": list(output["token_ids"]),
            }
    finally:
        llm.shutdown()

    [png] = _pngs("one-image")
    stub = _serve_image(png)
    server = ServerOptions(port=_free_port(), allowed_media_domains=("127.0.0.1",))
    online = {}
    try:
        with OnlineServer(options, server):
            http_url = f"http://127.0.0.1:{stub.server_port}/shapes.png"
            requests = {f"{chat}/data-url": (chat, urls) for chat, urls in chats.items()}
            requests["one-image/http-url"] = ("one-image", [http_url])
            for name, (chat, urls) in requests.items():
                body = {
                    "model": str(snapshot),
                    "messages": _openai_chat(chat, urls),
                    "temperature": 0,
                    "ignore_eos": True,
                    "max_tokens": GREEDY_TOKENS,
                    "return_token_ids": True,
                }
                request = Request(
                    f"http://127.0.0.1:{server.port}/v1/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(request, timeout=600) as response:
                    answer = json.load(response)
                online[name] = {
                    "prompt_ids": answer["prompt_token_ids"],
                    "tokens": answer["choices"][0]["token_ids"],
                }
    finally:
        stub.shutdown()
        stub.server_close()
    return {"offline": offline, "online": online}


# ============================== DFlash ==============================


def _dflash_arms(target: str, draft: str) -> dict[str, Any]:
    """Greedy image chats on the plain engine, then on the same engine with the DFlash draft;
    both arms on FlashInfer, which DFlash's other batches use."""

    from cti.config import (
        CacheOptions,
        EngineOptions,
        EngineTuning,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
        SpeculativeOptions,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    chats = image_chats()
    params = SamplingParams(temperature=0, ignore_eos=True, max_new_tokens=GREEDY_TOKENS)
    arms = {}
    for arm, speculative in (("baseline", None), ("dflash", SpeculativeOptions(model_path=draft))):
        llm = LLM(
            EngineOptions(
                model=ModelOptions(path=target, max_seq_len=MAX_SEQ_LEN),
                memory=MemoryOptions(inference_ratio=_INFERENCE_MEMORY_RATIO),
                scheduling=SchedulingOptions(max_running_requests=_DFLASH_SLOTS),
                cache=CacheOptions(prefix="naive"),
                speculative=speculative,
            ),
            tuning=EngineTuning(attention_backend="fi"),
        )
        try:
            encoder = llm.prompt_encoder()
            prompts = {
                name: encoder.chat(messages, template_kwargs={"enable_thinking": False})
                for name, messages in chats.items()
            }
            samples = {}
            for names in _BATCHES:
                result = llm.sample([prompts[name] for name in names], params)
                samples.update(zip(names, result.samples, strict=True))
        finally:
            llm.shutdown()
        arms[arm] = {
            name: {
                "prompt_ids": list(sample.prompt_token_ids),
                "tokens": list(sample.sampled_token_ids),
            }
            for name, sample in samples.items()
        }
    return arms


# ============================== Test ==============================


@pytest.mark.parametrize("checkpoint", _CHECKPOINTS, ids=lambda c: c.repo_id)
def test_qwen3_5_image_prompts_follow_the_fp32_truth(
    checkpoint: Checkpoint, tmp_path: Path
) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    snapshot = checkpoint.snapshot()
    if checkpoint.moe:
        require_device_memory(_MOE_MIN_GIB)
    runs, passes, images = {}, {}, {}
    for recipes in ("images", "questions"):
        prompts, recipe_images = _image_prompts(snapshot, recipes)
        images |= recipe_images
        for prefix in _PASSES:
            run = _arm(snapshot, tmp_path, recipes, prefix, prompts)
            cached = [sample["cached"] for samples in run["passes"] for sample in samples.values()]
            assert any(cached) == (prefix == "radix"), (recipes, prefix, cached)
            runs[f"{recipes}/{prefix}"] = run
        (naive,), (first, second) = (runs[f"{recipes}/{prefix}"]["passes"] for prefix in _PASSES)
        reference = f"{recipes}/naive"
        passes[reference] = (naive, reference)
        passes[f"{recipes}/radix-1"] = (first, reference)
        passes[f"{recipes}/radix-2"] = (second, reference)
    _judge(checkpoint, snapshot, tmp_path, "images", runs, passes, images)


@pytest.mark.parametrize("checkpoint", _CHECKPOINTS, ids=lambda c: c.repo_id)
def test_qwen3_5_cached_images_match_uncached_ones_under_eviction(
    checkpoint: Checkpoint, tmp_path: Path
) -> None:
    """An encoder pool of about one image serves five images under three questions in one
    overlapped batch: repeats load held rows or encode evicted ones again. Each must follow
    its prompt with uncached images under the truth."""

    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    snapshot = checkpoint.snapshot()
    if checkpoint.moe:
        require_device_memory(_MOE_MIN_GIB)
    prompts, images = _image_prompts(snapshot, "eviction")
    run = _arm(snapshot, tmp_path, "eviction", "naive", prompts)
    uncached, cached = run["passes"]
    passes = {"uncached": (uncached, "uncached"), "cached": (cached, "uncached")}
    _judge(checkpoint, snapshot, tmp_path, "image_eviction", {"eviction": run}, passes, images)


@pytest.mark.parametrize("max_batch_tokens", (1024, 2048))
def test_qwen3_5_an_image_split_across_chunks_matches_it_unsplit(
    max_batch_tokens: int, tmp_path: Path
) -> None:
    """A 3136-token image split across prefill chunks of ``max_batch_tokens`` is encoded
    whole by its first chunk; each chunk loads its part. Split, it must follow the truth as
    the unsplit step does."""

    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    checkpoint = selected_checkpoint()
    snapshot = checkpoint.snapshot()
    if checkpoint.moe:
        require_device_memory(_MOE_MIN_GIB)
    prompts, images = _image_prompts(snapshot, "split")
    runs = {
        arm: _arm(snapshot, tmp_path, "split", "naive", prompts, budget)
        for arm, budget in (("unsplit", MAX_SEQ_LEN), ("split", max_batch_tokens))
    }
    [unsplit], [split] = (run["passes"] for run in runs.values())
    passes = {"unsplit": (unsplit, "unsplit"), "split": (split, "unsplit")}
    name = f"image_split-{max_batch_tokens}"
    _judge(checkpoint, snapshot, tmp_path, name, runs, passes, images)


def test_qwen3_5_online_images_match_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    checkpoint = selected_checkpoint()
    snapshot = checkpoint.snapshot()
    # The image server is local; the API server's fetcher honours proxy variables.
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("NO_PROXY", "*")
    receipt = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        **_online_answers(snapshot),
    }
    write_receipt(f"qwen3_5_online_images-{checkpoint.slug}", receipt)

    assert_cuda_device_metadata(receipt)
    for chat, offline in receipt["offline"].items():
        assert len(offline["tokens"]) == GREEDY_TOKENS, chat
    for name, online in receipt["online"].items():
        assert online == receipt["offline"][name.split("/")[0]], name


def test_qwen3_5_dflash_image_prompts_follow_the_plain_engine(tmp_path: Path) -> None:
    import torch

    from tests.support.checkpoints import QWEN3_5_9B
    from tests.support.dflash import PINNED_CHECKPOINTS as DFLASH_PAIRS
    from tests.support.dflash import preflight
    from tests.support.hf_judge import compare_greedy

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    [pair] = [pair for pair in DFLASH_PAIRS if pair.target.repo_id == QWEN3_5_9B.repo_id]
    # Missing pinned checkpoints raise QualificationBlocked: they fail the run, never skip.
    snapshots = preflight(pair)["snapshots"]
    target = Path(snapshots["target"])
    arms = _dflash_arms(snapshots["target"], snapshots["draft"])
    prompts, images = _image_prompts(target)
    for arm in ("baseline", "dflash"):
        _check(arms[arm], prompts, arm)
    # The truth teacher-forced on the plain engine's own outputs gives its margins on that path.
    plain = {name: _key(sample) for name, sample in arms["baseline"].items()}
    checkpoint = pair.target
    job = (receipts_dir() or tmp_path) / f"qwen3_5-dflash-images-truth-{checkpoint.slug}.pt"
    truth, truth_report = SequenceTruth(checkpoint, target, tmp_path).require(
        set(plain.values()), job, images
    )
    comparisons = {
        name: compare_greedy(
            {**truth[key], "greedy_ids": torch.tensor(key[1])},
            {"greedy_ids": torch.tensor(arms["dflash"][name]["tokens"])},
        )
        for name, key in plain.items()
    }
    write_receipt(
        f"qwen3_5_dflash_images-{pair.target.repo_id.split('/')[-1]}",
        {"comparisons": comparisons, "truth": truth_report, "arms": arms},
    )

    for name, comparison in comparisons.items():
        assert comparison["agrees_until_truth_near_tie"], (name, comparison)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--recipes", choices=tuple(_RECIPES), required=True)
    parser.add_argument("--prefix", choices=tuple(_PASSES), required=True)
    parser.add_argument("--split-tokens", type=int)
    _run_arm(parser.parse_args())


if __name__ == "__main__":
    main()
