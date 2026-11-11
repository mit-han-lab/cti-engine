"""A rank's one composition root, ``EngineCore._start``: it rejects what the checkpoint cannot
serve before any device state exists, then binds the rank's device, plans for it and hands
the rank's parts the plan."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

from cti.config import (
    EngineOptions,
    EngineTuning,
    ModelOptions,
    ParallelOptions,
    SpeculativeOptions,
)
from cti.resolution import ResolvedModel
from cti.runtime._core import EngineCore
from cti.runtime._protocol import RankProgress


class _WorkerBuilt(Exception):
    """Raised by the stand-in rank: startup stops once the root has composed it."""


def _core(options: EngineOptions, *, rank: int = 0) -> EngineCore:
    return EngineCore(
        options,
        EngineTuning(),
        None,  # type: ignore[arg-type]
        rank=rank,
        init_method="unused",
        checkpoint=None,
        checks=True,
        overlap=False,
        online=False,
        progress=RankProgress([0.0, 0.0]),
    )


def test_a_checkpoint_it_cannot_serve_is_rejected_before_any_device_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    Qwen3Config(architectures=["LlamaForCausalLM"]).save_pretrained(tmp_path)
    bound: list[int] = []
    monkeypatch.setattr(torch.cuda, "set_device", bound.append)

    with pytest.raises(ValueError, match="LlamaForCausalLM not supported"):
        _core(EngineOptions(model=ModelOptions(path=str(tmp_path))))._start()

    assert bound == []


@pytest.mark.parametrize("refused", [False, True], ids=["planned", "refused"])
def test_the_root_checks_then_binds_the_rank_plans_and_hands_the_rank_its_plan(
    monkeypatch: pytest.MonkeyPatch, refused: bool
) -> None:
    events: list[str] = []
    plan = SimpleNamespace(model_path="model")
    method = object()

    checked = object()

    class _Checkpoint:
        model_config = "target"

        def check(self, _options: object, _tuning: object, *, speculative: object) -> object:
            assert speculative is method
            events.append("check")
            return checked

        def plan(
            self,
            accepted: object,
            *,
            device: torch.device,
            capability: str,
            device_memory: int,
            online: bool,
        ) -> object:
            assert online is False and accepted is checked
            events.append(f"plan {device} {capability} {device_memory >> 30} GiB")
            if refused:
                raise ValueError('cache prefix "radix" ...; use "naive"')
            return plan

    checkpoint = _Checkpoint()

    def load(_cls: type, path: str, *, hub: str, text_only: bool) -> _Checkpoint:
        assert hub == options.model.source
        events.append(f"load {path}")
        return checkpoint

    def load_speculative(
        path: str, *, target: object, hub: str, speculative_tokens: int | None
    ) -> object:
        assert hub == options.model.source and speculative_tokens is None
        events.append(f"load {path} beside {target}")
        return method

    def build_rank(planned: object, resolved: object, **composition: object) -> None:
        assert planned is plan and resolved is checkpoint
        assert (composition["rank"], composition["memory"]) == (1, "memory")
        events.append("rank")
        raise _WorkerBuilt

    monkeypatch.setattr(ResolvedModel, "load", classmethod(load))
    monkeypatch.setattr("cti.inference.load_speculative", load_speculative)
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: events.append(f"bind {rank}"))
    monkeypatch.setattr("cti.utils.get_device_capability", lambda: "sm120")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=32 << 30, is_integrated=0),
    )
    monkeypatch.setattr("cti.memory.MemoryManager.from_stock_provider", lambda *, unified: "memory")
    monkeypatch.setattr("cti.runtime._core.Rank", build_rank)
    options = EngineOptions(
        model=ModelOptions(path="model"),
        parallel=ParallelOptions(num_ranks=2),
        speculative=SpeculativeOptions(model_path="draft"),
    )

    with pytest.raises(ValueError if refused else _WorkerBuilt):
        _core(options, rank=1)._start()

    # A plan refused for this rank's device (such as a radix cache over recurrent layers
    # whose prefill exports no states) stops startup before any rank part exists.
    planned = [
        "load model",
        "load draft beside target",
        "check",
        "bind cuda:1",
        "plan cuda:1 sm120 32 GiB",
    ]
    assert events == planned if refused else [*planned, "rank"]


def test_the_root_keeps_the_ranks_parts_for_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parts' resources hold what the executor's captured graphs read (the graph pool's
    workspaces): the startup's last collection must not free them."""

    import gc
    import weakref

    class _Parts:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            ranks = SimpleNamespace(size=1, require_same=None)
            self.worker = SimpleNamespace(ranks=ranks, quiesce=lambda: None)
            self.executor = SimpleNamespace(limits=None)
            self.training = None

    class _Plane:
        def publish(self, _replies: object) -> None:
            pass

        def start(self, *, seeds: object) -> None:
            pass

        def reply(self, _reply: object) -> None:
            pass

    built: list[weakref.ref] = []

    def build_rank(*args: object, **kwargs: object) -> _Parts:
        parts = _Parts(*args, **kwargs)
        built.append(weakref.ref(parts))
        return parts

    checkpoint = SimpleNamespace(
        model_config=SimpleNamespace(vocab_size=8),
        facts=SimpleNamespace(hf_config=None),
        check=lambda *_args, **_kwargs: None,
        plan=lambda *_args, **_kwargs: SimpleNamespace(
            model_path="model", image_limits=None, seeds_every_row=False
        ),
    )
    monkeypatch.setattr(ResolvedModel, "load", classmethod(lambda _cls, *_a, **_k: checkpoint))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr("cti.utils.get_device_capability", lambda: "sm120")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=32 << 30, is_integrated=0),
    )
    monkeypatch.setattr("cti.memory.MemoryManager.from_stock_provider", lambda *, unified: None)
    monkeypatch.setattr("cti.runtime._core.Rank", build_rank)
    monkeypatch.setattr(
        "cti.tokenizer.load_text_decoder",
        lambda _path: SimpleNamespace(tokenizer=SimpleNamespace(chat_template=None)),
    )
    monkeypatch.setattr("cti.inference.InferenceEngine", lambda *_a, **_k: object())
    monkeypatch.setattr(torch, "set_num_threads", lambda _n: None)
    core = _core(EngineOptions(model=ModelOptions(path="model")))
    core._plane = _Plane()

    try:
        core._start()
        gc.collect()
        assert built[0]() is not None
    finally:
        gc.unfreeze()
