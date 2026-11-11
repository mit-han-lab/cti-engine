"""Tensor-parallel inference of compact checkpoints on 2-4 local GPUs; opt in with
CTI_TP_QUALIFY=1. Each case runs in its own process; observers record replays and
completions without replacing any provider. Real checkpoints: test_tensor_parallel_real_models."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
from functools import partial
from pathlib import Path

import pytest

from tests.support.gpu import assert_ranks_reclaimed, engine_rank_pids, stall_rank_after
from tests.support.multimodal import save_qwen3_5_preprocessor_config
from tests.support.replies import per_token
from tests.support.tp_capacity import assert_published_capacity, install_capacity_recorder
from tests.support.tp_cases import run_case

pytestmark = pytest.mark.checkpoints()

_QUALIFY = "CTI_TP_QUALIFY"
_OBSERVE_DIR = "CTI_TP_OBSERVE_DIR"
_STALL = "CTI_TP_STALL"  # a file whose existence stalls rank 1, alive, in its loop
_HANG_TIMEOUT = 20.0  # the distributed timeout a hung-rank case runs under
_AUTO_CAPACITY = "CTI_TP_AUTO_CAPACITY"
_FIXED_PAGES = 2048


def _require_gpus(count: int) -> None:
    if os.environ.get(_QUALIFY) != "1":
        pytest.skip(f"requires an exclusive multi-GPU allocation and {_QUALIFY}=1")
    import torch

    if torch.cuda.device_count() < count:
        pytest.skip(f"requires {count} visible GPUs, found {torch.cuda.device_count()}")


def _make_checkpoint(path: Path, *, num_kv_heads: int = 2) -> None:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    # Four query heads. TP4 replicates each of two KV heads on two ranks; TP2 replicates
    # one KV head, leaving each rank two query heads per KV head instead of four.
    config = Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=num_kv_heads,
        head_dim=64,
        vocab_size=259,
        max_position_embeddings=1024,
        bos_token_id=1,
        eos_token_id=3,
        pad_token_id=2,
        tie_word_embeddings=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1107)
        Qwen3ForCausalLM(config).to(torch.bfloat16).save_pretrained(path)
    _save_tokenizer(path)


def _make_qwen3_5_checkpoint(path: Path) -> None:
    """An image-text Qwen3.5 checkpoint: three Gated DeltaNet layers, then attention."""

    import torch
    from transformers import (
        Qwen2VLImageProcessorPil,
        Qwen3_5Config,
        Qwen3_5ForConditionalGeneration,
    )

    # Two key and four value Gated DeltaNet heads, four query and two KV heads: each of
    # two ranks owns one key head, the two value heads that read it and one KV head.
    config = Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        tie_word_embeddings=False,
        # Vision placeholders inside the compact vocabulary.
        image_token_id=250,
        video_token_id=251,
        vision_start_token_id=252,
        vision_end_token_id=253,
        text_config={
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 259,
            "max_position_embeddings": 1024,
            "bos_token_id": 1,
            "eos_token_id": 3,
            "pad_token_id": 2,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "num_heads": 2,
            "intermediate_size": 16,
            "out_hidden_size": 256,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 1,
            "num_position_embeddings": 4,
        },
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1107)
        Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16).save_pretrained(path)
    # Images of at most 64 tokens: 256 pixels a side over 2-pixel patches without a merge.
    Qwen2VLImageProcessorPil(
        size={"shortest_edge": 16, "longest_edge": 256}, patch_size=2, merge_size=1
    ).save_pretrained(path)
    _save_tokenizer(path)
    save_qwen3_5_preprocessor_config(path, config.vision_config)


def _save_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocabulary = {"[UNK]": 0, "[BOS]": 1, "[PAD]": 2, "[EOS]": 3}
    vocabulary.update({f"word{i}": i for i in range(4, 259)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        bos_token="[BOS]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        chat_template="{% for message in messages %}{{ message['content'] }} {% endfor %}",
    ).save_pretrained(path)


# ============================== Engine-process observers ==============================


def _tensor_cores(backend: object) -> bool | None:
    """FlashInfer's native tensor-core default, which it resolves from the rank's own heads
    (its device dispatch may still pick per batch); None for another backend."""

    from cti.attention.flashinfer import FlashInferBackend

    return backend.use_tensor_cores if isinstance(backend, FlashInferBackend) else None


def _install_rank_observers() -> None:
    """Run in every spawned engine process that re-imports this module as its main."""

    import torch

    import cti.inference as inference_package
    from cti.inference._control._engine import InferenceEngine
    from cti.inference._execution import runner as runner_module
    from cti.inference._execution._breakable_graph import BreakableGraph
    from cti.inference._execution._graphs import DecodeGraph
    from cti.protocol import DetokenizeMsg
    from cti.worker import DeviceMemory, GPUWorker

    evidence: dict[str, object] = {
        "pid": os.getpid(),
        "decode": [],
        "piecewise": 0,
        "admissions": [],
        "completions": [],
        "idle": [],
    }
    observe_dir = Path(os.environ[_OBSERVE_DIR])

    def save() -> None:
        evidence["cuda_device"] = torch.cuda.get_device_name()
        (observe_dir / f"rank-{evidence['rank']}.json").write_text(json.dumps(evidence) + "\n")

    build = runner_module.build_executor

    def observe_build(*args, **kwargs):
        # No fixed-page config knob exists. Patch the pure planner so the assembly still
        # assigns and validates both explicit capacities on every rank.
        from contextlib import nullcontext

        from tests.support.gpu import fixed_kv_page_capacity

        capacity = (
            nullcontext()
            if os.environ.get(_AUTO_CAPACITY) == "1"
            else fixed_kv_page_capacity(_FIXED_PAGES)
        )
        with capacity:
            runner = build(*args, **kwargs)
        evidence.update(
            rank=kwargs["ranks"].rank,
            device=str(runner.device),
            usable_kv_pages=runner.limits.capacity.num_pages,
            # The target's attention backend leads the mixer backends.
            tensor_cores=_tensor_cores(runner._participants[0]),
        )
        (observe_dir / f"rank-{evidence['rank']}.ready").write_text(str(os.getpid()))
        return runner

    # The composition root reads it off the inference facade, which may have cached it.
    runner_module.build_executor = inference_package.build_executor = observe_build
    allocate = DeviceMemory.allocate

    def observe_allocation(self, *args, **kwargs):
        # Device bytes the KV pools, recurrent rows and snapshot rows take.
        device = torch.cuda.current_device()
        torch.cuda.synchronize(device)
        free_before = torch.cuda.mem_get_info(device)[0]
        state = allocate(self, *args, **kwargs)
        torch.cuda.synchronize(device)
        evidence["state_device_bytes"] = free_before - torch.cuda.mem_get_info(device)[0]
        return state

    DeviceMemory.allocate = observe_allocation
    if os.environ.get(_AUTO_CAPACITY) == "1":
        install_capacity_recorder(evidence)

    decode_replay = DecodeGraph.replay

    def observe_decode(self, rows, batch):
        result = decode_replay(self, rows, batch)
        evidence["decode"].append([rows.size, rows.padded_size])
        return result

    DecodeGraph.replay = observe_decode
    piecewise_replay = BreakableGraph.replay

    def observe_piecewise(self, batch):
        result = piecewise_replay(self, batch)
        evidence["piecewise"] += 1
        return result

    BreakableGraph.replay = observe_piecewise
    settle_oldest = InferenceEngine._settle_oldest

    def observe_completion(self):
        replies = settle_oldest(self)
        evidence["completions"].append(
            [
                [reply.uid, reply.next_token, reply.logprobs and reply.logprobs.logprob]
                for reply in per_token(r for r in replies if isinstance(r, DetokenizeMsg))
            ]
        )
        return replies

    InferenceEngine._settle_oldest = observe_completion
    handle = InferenceEngine.handle

    def observe_admission(self, messages, *, admitting):
        if messages:
            evidence["admissions"].append(
                [[type(message).__name__, getattr(message, "uid", None)] for message in messages]
            )
        return handle(self, messages, admitting=admitting)

    InferenceEngine.handle = observe_admission
    step = InferenceEngine.step

    def observe_idle(self) -> None:
        # Once a step leaves nothing in flight or runnable: the requests finished so far, the
        # live and reserved allocator bytes, and the pages held, cached and their size.
        step(self)
        if self.busy:
            return
        occupancy = self._cache.occupancy
        evidence["idle"].append(
            {
                "finished": self._scheduler.totals.num_requests,
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "used_pages": occupancy.num_used_pages,
                "cached_pages": occupancy.num_cached_pages,
                "page_size": self._cache.page_size,
            }
        )

    InferenceEngine.step = observe_idle
    quiesce = GPUWorker.quiesce

    def observe_quiesce(self) -> None:
        # Provider-bound shutdown quiesces every rank, then reclaims processes; it
        # never unwinds managed graphs or groups in Python.
        quiesce(self)
        evidence["quiesced"] = True
        save()

    GPUWorker.quiesce = observe_quiesce


# =================================== Case entry points ===================================


def _rank_evidence(observe_dir: Path, size: int) -> list[dict]:
    return [json.loads((observe_dir / f"rank-{rank}.json").read_text()) for rank in range(size)]


def _assert_reclaimed(pids: list[int]) -> None:
    assert_ranks_reclaimed(tuple(pids))


def _run_bounded(args: argparse.Namespace) -> None:
    """Compact bounded sampling at one TP size, eager or with decode/piecewise graphs."""

    from cti.config import (
        EngineOptions,
        EngineTuning,
        MemoryOptions,
        ModelOptions,
        ParallelOptions,
        SchedulingOptions,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    observe_dir = Path(args.receipt).with_suffix("")
    if args.mode == "auto-capacity":
        os.environ[_AUTO_CAPACITY] = "1"
    graphs = args.mode == "graphs"
    piecewise = graphs
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=args.model, max_seq_len=128),
            parallel=ParallelOptions(num_ranks=args.tp),
            memory=MemoryOptions(inference_ratio=0.1 if args.mode == "auto-capacity" else 0.8),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=64),
        ),
        tuning=EngineTuning(
            cuda_graph_bs=(1, 2, 4) if graphs else (),
            graph_row_buckets=(16, 32) if piecewise else (),
        ),
    )
    samples = []
    try:
        # Greedy, then seeded sampling with selected log-probabilities.
        for temperature, count in ((0.0, 3), (0.7, 4)):
            prompts = [[5 + index, 9, 17, 23, 31][: 3 + index % 3] for index in range(count)]
            policies = [
                SamplingParams(
                    temperature=temperature,
                    max_new_tokens=5 + index,
                    ignore_eos=True,
                    seed=None if temperature == 0 else 101 + index,
                    logprobs=temperature > 0,
                )
                for index in range(count)
            ]
            first = llm.sample(prompts, policies)
            repeated = llm.sample(prompts, policies)
            # The repeat reuses cached prefixes; its tokens must not change.
            assert [s.sampled_token_ids for s in repeated.samples] == [
                s.sampled_token_ids for s in first.samples
            ]
            samples.append([list(s.sampled_token_ids) for s in first.samples])
    finally:
        llm.shutdown()
        llm.shutdown()

    ranks = _rank_evidence(observe_dir, args.tp)
    _assert_reclaimed([evidence["pid"] for evidence in ranks])
    # Rank 0's plan is the one every rank must have agreed on.
    published_pages = ranks[0]["usable_kv_pages"]
    for rank, evidence in enumerate(ranks):
        assert evidence["rank"] == rank and evidence["device"] == f"cuda:{rank}"
        assert evidence["quiesced"]
        assert evidence["usable_kv_pages"] == published_pages
        if graphs:
            assert (evidence["piecewise"] > 0) is piecewise
            assert {size for size, _padded in evidence["decode"]} >= {1, 2, 3, 4}
        else:
            assert not evidence["decode"] and evidence["piecewise"] == 0
        # Every rank replays the same graphs and settles the primary's samples.
        assert evidence["decode"] == ranks[0]["decode"]
        assert evidence["completions"] == ranks[0]["completions"]
    if args.mode == "auto-capacity":
        assert_published_capacity(ranks, published_pages, ranks[0]["plan"][2])
        # The planner sizes every pool by its native allocation, so the state each rank
        # allocated fits the agreed available memory, less the image encoder's peak beyond
        # the reserve.
        for evidence in ranks:
            budget = evidence["capacity"][1][1][0] - max(
                0, evidence["image_encoder_peak_bytes"] - evidence["peak_reserve_bytes"]
            )
            assert 0 < evidence["state_device_bytes"] <= budget, evidence
    Path(args.receipt).write_text(json.dumps({"samples": samples, "ranks": ranks}) + "\n")


def _run_radix(args: argparse.Namespace) -> None:
    """Explicit radix prefix reuse at one TP size: a repeated prompt, one sharing a prefix
    and a disjoint one, each greedy; every rank settles the same completions."""

    from cti.config import (
        CacheOptions,
        EngineOptions,
        EngineTuning,
        MemoryOptions,
        ModelOptions,
        ParallelOptions,
        SchedulingOptions,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=args.model, max_seq_len=128),
            parallel=ParallelOptions(num_ranks=args.tp),
            memory=MemoryOptions(inference_ratio=0.8),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=64),
            cache=CacheOptions(prefix="radix"),
        ),
        tuning=EngineTuning(cuda_graph_bs=(1, 2, 4)),
    )
    # Past the first snapshot point (64) of a model with recurrent layers.
    shared = [5 + index % 7 for index in range(100)]
    prompts = {
        "first": shared,
        "repeat": shared,
        "prefix": shared[:80] + [101, 102],
        "other": [200, 201, 202],
    }
    policy = SamplingParams(temperature=0.0, max_new_tokens=6, ignore_eos=True)
    outputs = {}
    try:
        for name, prompt in prompts.items():
            (sample,) = llm.sample([prompt], [policy]).samples
            outputs[name] = {
                "tokens": list(sample.sampled_token_ids),
                "cached": sample.cached_prompt_tokens,
            }
    finally:
        llm.shutdown()

    assert outputs["first"]["cached"] == 0 and outputs["other"]["cached"] == 0
    assert (
        outputs["repeat"]["cached"] > 0
        and outputs["repeat"]["tokens"] == outputs["first"]["tokens"]
    )
    assert 0 < outputs["prefix"]["cached"] <= 80
    ranks = _rank_evidence(Path(args.receipt).with_suffix(""), args.tp)
    _assert_reclaimed([evidence["pid"] for evidence in ranks])
    assert all(evidence["completions"] == ranks[0]["completions"] for evidence in ranks)
    Path(args.receipt).write_text(json.dumps({"outputs": outputs, "ranks": ranks}) + "\n")


def _run_images(args: argparse.Namespace) -> None:
    """One image under three questions, then two others that evict it from the 64 encoder
    rows, then the first question again: each greedy, every prompt twice, the second pass
    loading or re-encoding rows; every rank settles the same completions (the lockstep checks
    compare each step's rows)."""

    import torch

    from cti.config import (
        CacheOptions,
        EngineOptions,
        MemoryOptions,
        ModelOptions,
        ParallelOptions,
        SchedulingOptions,
    )
    from cti.llm import LLM
    from cti.multimodal import ImageInputs, ImageSpan, ProcessedPrompt
    from cti.multimodal.qwen3_5 import Qwen3_5RopeLayout
    from cti.protocol import SamplingParams

    generator = torch.Generator().manual_seed(7)
    # Patch grids of 16, 24 and 36 tokens (no spatial merge); 3 x 2 x 2 uint8 levels a patch.
    images = [
        (
            torch.randint(256, (h * w, 12), generator=generator, dtype=torch.uint8),
            (1, h, w),
            bytes([h, w]) * 16,
        )
        for h, w in ((4, 4), (4, 6), (6, 6))
    ]

    def prompt(image: int, question: int) -> ProcessedPrompt:
        pixels, grid, digest = images[image]
        span = ImageSpan(3, grid[1] * grid[2], grid, digest)
        ids = [5 + question, 9, 17, *[250] * span.num_tokens, 23 + question, 31]
        layout = Qwen3_5RopeLayout.from_spans((span,), 1)
        return ProcessedPrompt(
            torch.tensor(ids, dtype=torch.int32), ImageInputs(pixels, (span,)), layout
        )

    prompts = [prompt(0, q) for q in range(3)] + [prompt(1, 0), prompt(2, 0), prompt(0, 0)]
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=args.model, max_seq_len=128),
            parallel=ParallelOptions(num_ranks=args.tp),
            memory=MemoryOptions(inference_ratio=0.8),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=64),
            cache=CacheOptions(prefix="naive"),
        )
    )
    policy = SamplingParams(temperature=0.0, max_new_tokens=4, ignore_eos=True)
    try:
        first, _ = (
            [list(s.sampled_token_ids) for s in llm.sample(prompts, policy).samples]
            for _ in range(2)
        )
    finally:
        llm.shutdown()

    ranks = _rank_evidence(Path(args.receipt).with_suffix(""), args.tp)
    _assert_reclaimed([evidence["pid"] for evidence in ranks])
    assert all(evidence["completions"] == ranks[0]["completions"] for evidence in ranks)
    Path(args.receipt).write_text(json.dumps({"tokens": first, "ranks": ranks}) + "\n")


def _run_rank_failure(args: argparse.Namespace) -> None:
    """Kill one rank: the next call fails and every rank is reclaimed. Its remote traceback
    keeps this case out of log-scanned qualification."""

    from cti.config import (
        EngineOptions,
        EngineTuning,
        ModelOptions,
        ParallelOptions,
        SchedulingOptions,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams
    from cti.runtime import EngineProcessError

    observe_dir = Path(args.receipt).with_suffix("")
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=args.model, max_seq_len=128),
            parallel=ParallelOptions(num_ranks=args.tp),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=64),
        ),
        tuning=EngineTuning(cuda_graph_bs=()),
    )
    pids = [int((observe_dir / f"rank-{rank}.ready").read_text()) for rank in range(args.tp)]
    try:
        assert engine_rank_pids() == tuple(pids)
        os.kill(pids[-1], signal.SIGKILL)
        with pytest.raises(EngineProcessError) as failure:
            llm.sample([[5, 9, 17]], [SamplingParams(temperature=0, max_new_tokens=4)])
        # A failed engine stays failed: the next call raises the same error.
        with pytest.raises(EngineProcessError) as again:
            llm.sample([[5, 9, 17]], [SamplingParams(temperature=0, max_new_tokens=4)])
        assert again.value is failure.value
    finally:
        llm.shutdown()
    _assert_reclaimed(pids)
    Path(args.receipt).write_text(json.dumps({"reclaimed": pids}) + "\n")


def _run_rank_hang(args: argparse.Namespace) -> None:
    """Once warm, stop rank 1 (rank-freeze) or stall it alive (rank-stall): the next call fails
    as hung within the distributed timeout, and every rank is reclaimed. Without checks, so no
    gloo comparison times out first: rank 0 waits in its NCCL collective, as across hosts."""

    import time

    from cti.config import (
        EngineOptions,
        EngineTuning,
        ModelOptions,
        ParallelOptions,
        SchedulingOptions,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams
    from cti.runtime import EngineProcessError

    os.environ["CTI_CHECKS"] = "0"
    trigger = Path(args.receipt).with_suffix(".stall")
    os.environ[_STALL] = str(trigger)
    observe_dir = Path(args.receipt).with_suffix("")
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=args.model, max_seq_len=128),
            parallel=ParallelOptions(num_ranks=args.tp),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=64),
        ),
        tuning=EngineTuning(cuda_graph_bs=(), distributed_timeout=_HANG_TIMEOUT),
    )
    pids = [int((observe_dir / f"rank-{rank}.ready").read_text()) for rank in range(args.tp)]
    params = [SamplingParams(temperature=0, max_new_tokens=4)]
    try:
        assert engine_rank_pids() == tuple(pids)
        llm.sample([[5, 9, 17]], params)  # the first work's longer allowance is spent
        if args.mode == "rank-freeze":
            os.kill(pids[1], signal.SIGSTOP)
        else:
            trigger.touch()
        started = time.monotonic()
        with pytest.raises(EngineProcessError, match="the engine hung") as failure:
            llm.sample([[5, 9, 17]], params)
        elapsed = time.monotonic() - started
    finally:
        llm.shutdown()
    _assert_reclaimed(pids)
    Path(args.receipt).write_text(
        json.dumps({"elapsed": elapsed, "error": str(failure.value), "reclaimed": pids}) + "\n"
    )


def _run_online(args: argparse.Namespace) -> None:
    """Idle past the gloo timeout, then serve concurrent and cancelled requests."""

    from cti.config import (
        EngineOptions,
        EngineTuning,
        ModelOptions,
        ParallelOptions,
        SchedulingOptions,
        ServerOptions,
    )
    from cti.server import OnlineServer

    observe_dir = Path(args.receipt).with_suffix("")
    piecewise = True
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with OnlineServer(
        EngineOptions(
            model=ModelOptions(path=args.model, max_seq_len=1024),
            parallel=ParallelOptions(num_ranks=args.tp),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=64),
        ),
        ServerOptions(port=port),
        tuning=EngineTuning(
            cuda_graph_bs=(1, 2, 4), graph_row_buckets=(16, 32) if piecewise else ()
        ),
    ):
        asyncio.run(_online_requests(args.model, port))
    ranks = _rank_evidence(observe_dir, args.tp)
    _assert_reclaimed([evidence["pid"] for evidence in ranks])
    for evidence in ranks:
        assert evidence["quiesced"] and evidence["decode"]
        assert (evidence["piecewise"] > 0) is piecewise
        assert any(
            kind == "AbortBackendMsg" for batch in evidence["admissions"] for kind, _ in batch
        )
        # One ordered request stream: every rank admits and settles the same work.
        assert evidence["admissions"] == ranks[0]["admissions"]
        assert evidence["completions"] == ranks[0]["completions"]
    Path(args.receipt).write_text(json.dumps({"ranks": ranks}) + "\n")


async def _online_requests(model: str, port: int) -> None:
    from openai import AsyncOpenAI

    async with AsyncOpenAI(
        base_url=f"http://127.0.0.1:{port}/v1", api_key="test", max_retries=0, timeout=60
    ) as client:
        # Longer than the default 60 s collective timeout: peers wait inside the
        # admission broadcast only for the primary's bounded idle poll.
        await asyncio.sleep(65)

        async def completion(index: int, *, tokens: int = 8) -> None:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"word{5 + index} word9 word17"}],
                temperature=0.7,
                max_tokens=tokens,
                extra_body={"ignore_eos": True},
            )
            assert response.usage.completion_tokens == tokens

        await asyncio.gather(*(completion(index) for index in range(4)))
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "word31 word41 word51"}],
            temperature=0.7,
            max_tokens=900,
            stream=True,
            extra_body={"ignore_eos": True},
        )
        async for chunk in stream:
            if chunk.choices:
                break
        await stream.close()
        # Let the frontend's abort enter the ordered stream before more work.
        await asyncio.sleep(0.25)
        await asyncio.gather(completion(0), completion(1, tokens=11))


def _run_training(args: argparse.Namespace) -> None:
    """LoRA SFT at one TP size: the steps after ``--resume``'s, if any, to three, each one's
    loss and gradient norm; the checkpoint after two is saved at ``--save``. Every rank then
    settles the same greedy completions."""

    from cti.config import (
        AdamWConfig,
        BatchLayout,
        EngineOptions,
        LoRAConfig,
        ModelOptions,
        ParallelOptions,
        SchedulingOptions,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )
    from cti.llm import LLM
    from cti.protocol import SamplingParams
    from cti.training import SFTDatum, TrainingBatch

    observe_dir = Path(args.receipt).with_suffix("")
    llm = LLM(
        EngineOptions(
            model=ModelOptions(path=args.model, max_seq_len=128),
            parallel=ParallelOptions(num_ranks=args.tp),
            lora=LoRAConfig(targets=("attention", "mlp"), rank=8, seed=3),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=64),
            training=TrainingConfig(
                TrainablePlan.all_declared(),
                AdamWConfig(lr=1e-2, max_grad_norm=1.0),
                TrainingBatching(BatchLayout.PADDED, StaticBatching(2, 16)),
            ),
        ),
        resume_from_checkpoint=args.resume,
    )
    trace = []
    try:
        with llm.training_session() as session:
            for step in range(llm.model_revision, 3):
                data = [
                    SFTDatum(tokens, (False,) + (True,) * (len(tokens) - 1))
                    for tokens in ((5 + step, 9, 17, 23, 31, 7), (11, 13 + step, 19, 29))
                ]
                future = session.forward_backward(
                    TrainingBatch.sft(data, BatchLayout.PADDED, pad_token_id=0)
                )
                optimizer = session.optim_step().result()
                session.publish()
                trace.append([future.result().loss_sum, optimizer.grad_norm])
                if step == 1 and args.save:
                    session.save_checkpoint(args.save)
        prompts = [[5, 9, 17], [11, 13, 19, 29]]
        policy = SamplingParams(temperature=0.0, max_new_tokens=6, ignore_eos=True)
        samples = [list(s.sampled_token_ids) for s in llm.sample(prompts, [policy] * 2).samples]
    finally:
        llm.shutdown()

    ranks = _rank_evidence(observe_dir, args.tp)
    _assert_reclaimed([evidence["pid"] for evidence in ranks])
    for evidence in ranks:
        assert evidence["completions"] == ranks[0]["completions"]
    Path(args.receipt).write_text(json.dumps({"trace": trace, "samples": samples}) + "\n")


def _device_memory_mib() -> tuple[dict[str, int], dict[int, tuple[str, int]]]:
    """The driver's used MiB of every GPU by UUID, and each compute process's GPU and MiB by
    PID."""

    import subprocess

    def query(*fields: str) -> list[list[str]]:
        output = subprocess.run(
            ("nvidia-smi", *fields, "--format=csv,noheader,nounits"),
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
        return [[cell.strip() for cell in line.split(",")] for line in output.splitlines() if line]

    devices = {uuid: int(used) for uuid, used in query("--query-gpu=uuid,memory.used")}
    processes = {
        int(pid): (uuid, int(used))
        for pid, uuid, used in query("--query-compute-apps=pid,gpu_uuid,used_memory")
    }
    return devices, processes


def _run_case(tmp_path: Path, component: str, model: Path, tp: int, mode: str, **options):
    return run_case(
        Path(__file__), tmp_path, component, model, tp, mode, observe=_OBSERVE_DIR, **options
    )


_COMPACT_MODELS = {
    "qwen3": _make_checkpoint,
    "qwen3_kv1": partial(_make_checkpoint, num_kv_heads=1),
    "qwen3_5": _make_qwen3_5_checkpoint,
}


def _compact_model(tmp_path: Path, name: str) -> Path:
    model = tmp_path / name
    _COMPACT_MODELS[name](model)
    return model


# One rank's bounded sampling is the single-GPU tiers'; qwen3 keeps it beside TP4, which skips
# on a pair. The replicated KV head (qwen3_kv1) runs graphs only; its prefill still runs eager.
@pytest.mark.parametrize(
    ("name", "sizes", "modes"),
    [
        ("qwen3", (1, 2, 4), ("eager", "graphs")),
        ("qwen3_kv1", (2,), ("graphs",)),
        ("qwen3_5", (2,), ("eager", "graphs")),
    ],
    ids=["qwen3", "qwen3_kv1", "qwen3_5"],
)
def test_compact_bounded_tp_matches_across_ranks_in_eager_and_graph_modes(
    tmp_path: Path, name: str, sizes: tuple[int, ...], modes: tuple[str, ...]
) -> None:
    _require_gpus(max(sizes))
    model = _compact_model(tmp_path, name)
    for tp in sizes:
        for mode in modes:
            receipt = _run_case(tmp_path, f"tensor-parallel-{name}", model, tp, mode)
            assert len(receipt["ranks"]) == tp
            if name == "qwen3_kv1":
                # FlashInfer reads the rank's query/KV ratio (2), not the model's (4).
                assert {rank["tensor_cores"] for rank in receipt["ranks"]} == {False}


def test_compact_qwen3_5_tp2_cached_images_match_across_ranks(tmp_path: Path) -> None:
    """Under the lockstep checks, both ranks' encoder caches plan the same loads and stores."""
    _require_gpus(2)
    model = _compact_model(tmp_path, "qwen3_5")
    receipt = _run_case(tmp_path, "tensor-parallel-qwen3_5-images", model, 2, "images")
    assert len(receipt["ranks"]) == 2


@pytest.mark.parametrize("name", ["qwen3", "qwen3_5"])
def test_compact_tp2_auto_capacity_publishes_the_minimum_across_ranks(
    tmp_path: Path, name: str
) -> None:
    _require_gpus(2)
    model = _compact_model(tmp_path, name)
    receipt = _run_case(tmp_path, f"tensor-parallel-{name}-capacity", model, 2, "auto-capacity")

    assert all(rank["usable_kv_pages"] != _FIXED_PAGES for rank in receipt["ranks"])
    # Each Qwen3.5 rank reserves recurrent state for its own heads.
    assert all((rank["recurrent_bytes"] > 0) is (name == "qwen3_5") for rank in receipt["ranks"])
    if name == "qwen3_5":
        assert all(
            rank["image_encoder"] and rank["image_encoder"][-1][0] > 0 for rank in receipt["ranks"]
        )


@pytest.mark.parametrize("name", ["qwen3", "qwen3_5"])
def test_compact_radix_reuses_prefixes_alike_on_one_and_two_ranks(
    tmp_path: Path, name: str
) -> None:
    _require_gpus(2)
    model = _compact_model(tmp_path, name)
    receipts = [
        _run_case(tmp_path, f"tensor-parallel-{name}-radix", model, tp, "radix") for tp in (1, 2)
    ]

    assert [len(receipt["ranks"]) for receipt in receipts] == [1, 2]
    # Both sizes hit at the same depths.
    cached = [
        {prompt: output["cached"] for prompt, output in receipt["outputs"].items()}
        for receipt in receipts
    ]
    assert cached[0] == cached[1]


@pytest.mark.parametrize(("name", "tp"), [("qwen3", 2), ("qwen3", 4), ("qwen3_5", 2)])
def test_compact_online_tp_survives_idle_and_orders_concurrent_and_cancelled_requests(
    tmp_path: Path, name: str, tp: int
) -> None:
    _require_gpus(tp)
    model = _compact_model(tmp_path, name)
    _run_case(tmp_path, f"tensor-parallel-{name}-online", model, tp, "online")


@pytest.mark.parametrize("hang", ["rank-freeze", "rank-stall"])
def test_compact_tp2_hung_rank_fails_the_runtime_within_the_timeout(
    tmp_path: Path, hang: str
) -> None:
    _require_gpus(2)
    model = _compact_model(tmp_path, "qwen3")
    receipt = _run_case(tmp_path, f"tensor-parallel-{hang}", model, 2, hang)

    # Rank 0 waits in a collective with its work in hand; a stalled rank 1 holds its own.
    assert _HANG_TIMEOUT < receipt["elapsed"] < _HANG_TIMEOUT + 15, receipt
    assert "cti-rank-0 for" in receipt["error"], receipt
    if hang == "rank-freeze":
        assert "(cti-rank-1 stopped)" in receipt["error"], receipt
    else:
        assert "cti-rank-1 for" in receipt["error"], receipt
    assert len(receipt["reclaimed"]) == 2


def test_compact_tp2_rank_death_fails_the_runtime_and_reclaims_every_rank(tmp_path: Path) -> None:
    _require_gpus(2)
    model = _compact_model(tmp_path, "qwen3")
    # The surviving rank logs the failure it meets, traceback and all.
    receipt = _run_case(
        tmp_path, "tensor-parallel-failure", model, 2, "rank-failure", expected=("traceback",)
    )

    assert len(receipt["reclaimed"]) == 2


# BF16 partial sums reduce in a different order on two ranks.
_BF16_TP = {"rtol": 2e-2, "atol": 2e-2}


def test_compact_qwen3_5_lora_training_tp2_matches_one_rank_and_resumes_on_one_rank(
    tmp_path: Path,
) -> None:
    """Two ranks train LoRA through both mixers and save after two steps; one rank resumes
    there and its third step's loss and gradient norm match theirs. Both ranks then serve the
    trained adapter alike. The gloo tests cover the sharded optimizer and one rank's checkpoint
    on two."""

    import torch

    _require_gpus(2)
    # Checkpoints name the model by its Hugging Face snapshot, an immutable revision.
    snapshot = tmp_path / "hub" / "models--compact--qwen3_5" / "snapshots" / ("0" * 40)
    snapshot.parent.mkdir(parents=True)
    model = _compact_model(tmp_path, "qwen3_5").rename(snapshot)
    saved = tmp_path / "tp2.ckpt"
    trained = _run_case(
        tmp_path, "tensor-parallel-qwen3_5", model, 2, "training", arguments=("--save", str(saved))
    )
    resumed = _run_case(
        tmp_path / "resume",
        "tensor-parallel-qwen3_5",
        model,
        1,
        "training",
        arguments=("--resume", str(saved)),
    )
    torch.testing.assert_close(
        torch.tensor(resumed["trace"]), torch.tensor(trained["trace"][2:]), **_BF16_TP
    )


if __name__ == "__mp_main__" and os.environ.get(_OBSERVE_DIR):
    _install_rank_observers()
if __name__ == "__mp_main__" and os.environ.get(_STALL):
    stall_rank_after(os.environ[_STALL], rank=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "eager",
            "graphs",
            "auto-capacity",
            "rank-failure",
            "rank-freeze",
            "rank-stall",
            "online",
            "radix",
            "images",
            "training",
        ),
        required=True,
    )
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--save")
    parser.add_argument("--resume")
    arguments = parser.parse_args()
    if arguments.mode == "online":
        _run_online(arguments)
    elif arguments.mode == "rank-failure":
        _run_rank_failure(arguments)
    elif arguments.mode in ("rank-freeze", "rank-stall"):
        _run_rank_hang(arguments)
    elif arguments.mode == "radix":
        _run_radix(arguments)
    elif arguments.mode == "images":
        _run_images(arguments)
    elif arguments.mode == "training":
        _run_training(arguments)
    else:
        _run_bounded(arguments)
