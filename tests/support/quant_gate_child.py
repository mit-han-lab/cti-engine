"""The engine child of the quantized checkpoints' gates 1, 4 and 5
(``tests/hf_parity/test_quant_gates.py``), through the public ``LLM`` API.

``graph-eager``: greedy decoding of the first B parity prompts, B = 1..4, by an engine that
replays decode graphs and by one that runs eagerly; token ids and chosen-token logprobs.
``switching``: a LoRA training engine serves, enters and leaves training sessions that run a
forward/backward and discard it, and serves again; the outputs of every serving phase. Its rank
processes run this module as ``__mp_main__`` (spawn imports the parent's main module), which
installs the residency probe: around each switch every quantized tensor's ``data_ptr`` and
checksum, one JSON line per event in ``--probe`` (``rank-<device>.jsonl``).
``lora``: the base engine's outputs, a LoRA engine's with its zero-initialized adapter, then after
one committed LoRA training step.

Prints one ``CTI_QUANT_GATE=`` JSON line."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest.mock import patch

RECEIPT_PREFIX = "CTI_QUANT_GATE="
PROBE_VARIABLE = "CTI_QUANT_GATE_PROBE"
_MAX_RUNNING_REQUESTS = 4
_MAX_SEQ_LEN = 2048
_INFERENCE_MEMORY_RATIO = 0.85
_LORA_RANK = 8
_SWITCH_CYCLES = 3
_TRAINING_TOKENS = 128
_CHECKSUM_CHUNK = 1 << 26


# =============================== The residency probe ===============================


def _quantized_tensors(model: Any) -> dict[str, Any]:
    """Every tensor a ``layers/quant`` method declared on its layer, and any FP8 or packed
    FP4 tensor elsewhere, by state-dict path."""

    import torch

    tensors = {}
    for prefix, op in model.named_ops():
        method = getattr(op, "method", None)
        if method is not None and type(method).__module__.startswith("cti.layers.quant"):
            for name in getattr(op, "_declared", ()):
                if isinstance(tensor := getattr(op, name, None), torch.Tensor):
                    tensors[f"{prefix}.{name}"] = tensor
    for name, tensor in model.state_dict().items():
        if tensor.dtype in (torch.float8_e4m3fn, torch.uint8):
            tensors.setdefault(name, tensor)
    return tensors


def _checksum(tensor: Any) -> list[int]:
    """The byte sum and a position-weighted byte sum of ``tensor``."""

    import torch

    data = tensor.detach().contiguous().view(torch.uint8).flatten()
    plain = weighted = 0
    for first in range(0, data.numel(), _CHECKSUM_CHUNK):
        chunk = data[first : first + _CHECKSUM_CHUNK].to(torch.int64)
        weights = torch.arange(first, first + chunk.numel(), device=chunk.device) % 65521 + 1
        plain += int(chunk.sum())
        weighted += int((chunk * weights).sum())
    return [plain, weighted]


def _install_probe(folder: Path) -> None:
    import torch

    from cti.worker import GPUWorker

    release, acquire = GPUWorker.release, GPUWorker.acquire

    def record(worker: Any, event: str, mode: Any) -> None:
        tensors = _quantized_tensors(worker.host.model)
        line = {
            "event": event,
            "mode": str(mode),
            "tensors": {
                name: [tensor.data_ptr(), *_checksum(tensor)]
                for name, tensor in sorted(tensors.items())
            },
        }
        path = folder / f"rank-{torch.cuda.current_device()}.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(line) + "\n")

    def probed_release(self: Any, mode: Any) -> None:
        record(self, "before_release", mode)
        release(self, mode)

    def probed_acquire(self: Any, mode: Any) -> None:
        acquire(self, mode)
        record(self, "after_acquire", mode)

    # For the rank process's lifetime.
    patch.object(GPUWorker, "release", probed_release).start()
    patch.object(GPUWorker, "acquire", probed_acquire).start()


if __name__ == "__mp_main__" and PROBE_VARIABLE in os.environ:
    _install_probe(Path(os.environ[PROBE_VARIABLE]))


# ==================================== The engine ====================================


def _options(arguments: argparse.Namespace, *, lora: bool, training: bool) -> Any:
    from cti.config import (
        CacheOptions,
        EngineOptions,
        LoRAConfig,
        MemoryOptions,
        ModelOptions,
        ParallelOptions,
        SchedulingOptions,
    )
    from cti.models import get_model_class

    lora_config = None
    if lora:
        config = json.loads((arguments.snapshot / "config.json").read_text(encoding="utf-8"))
        model_class = get_model_class(config["architectures"][0])
        lora_config = LoRAConfig(targets=model_class.LORA_TARGETS, rank=_LORA_RANK, alpha=16.0)
    return EngineOptions(
        model=ModelOptions(path=str(arguments.snapshot), max_seq_len=_MAX_SEQ_LEN),
        parallel=ParallelOptions(
            num_ranks=arguments.ranks, enable_expert_parallel=not arguments.expert_tensor_parallel
        ),
        memory=MemoryOptions(inference_ratio=_INFERENCE_MEMORY_RATIO),
        scheduling=SchedulingOptions(
            max_running_requests=_MAX_RUNNING_REQUESTS, max_batch_tokens=_MAX_SEQ_LEN
        ),
        cache=CacheOptions(prefix="naive"),
        lora=lora_config,
        training=_training_config() if training else None,
    )


def _training_config() -> Any:
    from cti.config import (
        AdamWConfig,
        BatchLayout,
        StaticBatching,
        TrainablePlan,
        TrainingBatching,
        TrainingConfig,
    )

    return TrainingConfig(
        trainable_plan=TrainablePlan.all_declared(),
        optimizer=AdamWConfig(lr=1e-3, weight_decay=0.0, max_grad_norm=1.0),
        batching=TrainingBatching(
            BatchLayout.PADDED,
            StaticBatching(microbatch_size=1, max_tokens_per_sample=_TRAINING_TOKENS),
        ),
    )


def _training_batch(ids: list[int]) -> Any:
    """One SFT row of a parity prompt's first tokens; a single row has no padding, so the pad
    id is never read."""

    from cti.config import BatchLayout
    from cti.training import SFTDatum, TrainingBatch

    tokens = tuple(ids[:_TRAINING_TOKENS])
    datum = SFTDatum(token_ids=tokens, target_mask=(False,) + (True,) * (len(tokens) - 1))
    return TrainingBatch.sft((datum,), BatchLayout.PADDED, pad_token_id=0)


def _tuning(*, graphs: bool) -> Any:
    from cti.config import EngineTuning

    if graphs:
        return EngineTuning(cuda_graph_bs=tuple(range(1, _MAX_RUNNING_REQUESTS + 1)))
    return EngineTuning(cuda_graph_max_bs=0)


def _prompts(arguments: argparse.Namespace) -> list[list[int]]:
    from tests.support.hf_truth import load_prompts

    _, prompts = load_prompts(arguments.prompts, arguments.model_key)
    return [ids[: arguments.prompt_tokens] for _, ids in prompts]


def _decode(llm: Any, prompts: list[list[int]], tokens: int) -> dict[str, list[list[Any]]]:
    """Greedy ids and their logprobs of ``prompts`` decoded together."""

    from cti.protocol import SamplingParams

    params = SamplingParams(temperature=0, ignore_eos=True, max_new_tokens=tokens, logprobs=True)
    samples = llm.sample(prompts, params).samples
    return {
        "ids": [list(sample.sampled_token_ids) for sample in samples],
        "logprobs": [[float(value) for value in sample.sampling_logprobs] for sample in samples],
    }


def _graph_eager(arguments: argparse.Namespace) -> dict[str, Any]:
    from cti.llm import LLM

    prompts = _prompts(arguments)
    outputs: dict[str, dict[str, Any]] = {}
    for mode in ("graph", "eager"):
        llm = LLM(
            _options(arguments, lora=False, training=False), tuning=_tuning(graphs=mode == "graph")
        )
        try:
            outputs[mode] = {
                str(batch): _decode(llm, prompts[:batch], arguments.tokens)
                for batch in range(1, _MAX_RUNNING_REQUESTS + 1)
            }
        finally:
            llm.shutdown()
    return outputs


def _switching(arguments: argparse.Namespace) -> dict[str, Any]:
    from cti.llm import LLM

    prompts = _prompts(arguments)[:_MAX_RUNNING_REQUESTS]
    llm = LLM(_options(arguments, lora=True, training=True), tuning=_tuning(graphs=True))
    phases, losses = [], []
    try:
        batch = _training_batch(prompts[0])
        phases.append(_decode(llm, prompts, arguments.tokens))
        for _ in range(_SWITCH_CYCLES):
            with llm.training_session() as session:
                losses.append(session.forward_backward(batch).result().loss_sum)
                session.discard_accumulation()
            phases.append(_decode(llm, prompts, arguments.tokens))
        revision = llm.model_revision
    finally:
        llm.shutdown()
    return {"phases": phases, "loss_sums": losses, "model_revision": revision}


def _lora(arguments: argparse.Namespace) -> dict[str, Any]:
    from cti.llm import LLM

    prompts = _prompts(arguments)[:_MAX_RUNNING_REQUESTS]
    llm = LLM(_options(arguments, lora=False, training=False), tuning=_tuning(graphs=True))
    try:
        base = _decode(llm, prompts, arguments.tokens)
    finally:
        llm.shutdown()
    llm = LLM(_options(arguments, lora=True, training=True), tuning=_tuning(graphs=True))
    try:
        zero = _decode(llm, prompts, arguments.tokens)
        batch = _training_batch(prompts[0])
        with llm.training_session() as session:
            completion = session.forward_backward(batch).result()
            step = session.optim_step().result()
            assert llm.model_revision == 0
            assert _decode(llm, prompts, arguments.tokens) == zero
            session.publish()
        trained = _decode(llm, prompts, arguments.tokens)
        revision = llm.model_revision
    finally:
        llm.shutdown()
    return {
        "base": base,
        "zero_adapter": zero,
        "trained": trained,
        "loss_sum": completion.loss_sum,
        "grad_norm": step.grad_norm,
        "step": step.step,
        "model_revision": revision,
    }


_GATES = MappingProxyType({"graph-eager": _graph_eager, "switching": _switching, "lora": _lora})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("gate", choices=sorted(_GATES))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--expert-tensor-parallel", action="store_true")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    arguments = parser.parse_args()

    import importlib.metadata

    import torch

    import cti

    result = _GATES[arguments.gate](arguments)
    arguments.output.write_text(json.dumps(result) + "\n", encoding="utf-8")
    receipt = {
        "gate": arguments.gate,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "engine_root": str(Path(cti.__file__).resolve().parent),
        "kernels_version": importlib.metadata.version("cti-kernels"),
        "num_ranks": arguments.ranks,
        "expert_parallel": not arguments.expert_tensor_parallel,
        "output": str(arguments.output),
    }
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    sys.exit(main())
