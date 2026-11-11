from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pytest
import torch

from cti.attention import AttentionRequirement
from cti.attention.paged import KVPair
from cti.config import EngineOptions, EngineTuning, ModelOptions, SpeculativeOptions
from cti.forward import RoundShape
from cti.inference._execution._speculative.method import DFlash, load_speculative
from cti.kvcache import GroupRole, ResolvedSlot, Spec
from cti.models.dflash import DFlashDraftModel
from cti.resolution import ResolvedModel, _state_slots
from cti.utils import DeviceCapability
from tests.support.dflash import TINY_DRAFT_CONFIG as _DRAFT
from tests.support.dflash import TINY_TARGET_CONFIG as _TARGET


def _write(folder: Path, document: dict) -> str:
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(document), encoding="utf-8")
    return str(folder)


def _load(
    target: str, draft: str, speculative_tokens: int | None = None
) -> tuple[ResolvedModel, DFlash]:
    """The target and its draft's method, loaded as the composition root loads them."""

    hub = ModelOptions(path=target).source
    checkpoint = ResolvedModel.load(target, hub=hub)
    method = load_speculative(
        draft, target=checkpoint.model_config, hub=hub, speculative_tokens=speculative_tokens
    )
    return checkpoint, method


@pytest.mark.parametrize("count", [0, 17, -1, True, 4.0, "4"])
def test_speculative_tokens_outside_one_to_sixteen_are_refused(count: object) -> None:
    with pytest.raises(ValueError, match="num_speculative_tokens must be an integer from 1 to 16"):
        SpeculativeOptions(model_path="draft", num_speculative_tokens=count)


@pytest.mark.parametrize(("count", "width"), [(None, 16), (1, 2), (7, 8), (16, 17)])
def test_a_round_is_the_anchor_and_its_speculative_tokens_whatever_the_trained_block(
    tmp_path: Path, count: int | None, width: int
) -> None:
    # The tiny draft is trained on blocks of 16: None keeps its 15 drafts.
    target, draft = _write(tmp_path / "target", _TARGET), _write(tmp_path / "draft", _DRAFT)
    _, method = _load(target, draft, count)

    assert method.config.block_size == 16
    assert method.round_width == width
    shape = method.round_shape(max_running_req=4, max_forward_len=4 * width)
    assert shape == RoundShape(width=width, depth=2, max_rows=4)


def test_the_method_loads_the_draft_beside_its_target(tmp_path: Path) -> None:
    draft = Path(_write(tmp_path / "draft", _DRAFT))
    alias = tmp_path / "alias"
    alias.symlink_to(draft, target_is_directory=True)
    _, method = _load(_write(tmp_path / "target", _TARGET), str(alias))

    # The draft's path is frozen where its config was read.
    assert method.source.folder == str(draft.resolve())
    assert method.model_class is DFlashDraftModel
    shape = method.round_shape(max_running_req=8, max_forward_len=40)
    assert shape == RoundShape(width=16, depth=2, max_rows=2)


def test_a_draft_architecture_outside_the_table_is_refused(tmp_path: Path) -> None:
    target = _write(tmp_path / "target", _TARGET)
    draft = _write(tmp_path / "draft", {**_DRAFT, "architectures": ["Qwen3ForCausalLM"]})

    with pytest.raises(ValueError, match="Qwen3ForCausalLM not supported"):
        _load(target, draft)


@pytest.mark.parametrize("spelling", ("{tmp}/missing", "./missing-draft", "~/missing-draft"))
def test_a_missing_local_draft_directory_is_named(tmp_path: Path, spelling: str) -> None:
    """A path spelled as one is never taken for a Hugging Face repository id."""

    draft = spelling.format(tmp=tmp_path)
    with pytest.raises(FileNotFoundError, match=f"model directory {draft} does not exist"):
        _load(_write(tmp_path / "target", _TARGET), draft)


def _options(target: str, draft: str, **sections: object) -> EngineOptions:
    return EngineOptions(
        model=ModelOptions(path=target),
        speculative=SpeculativeOptions(model_path=draft),
        **sections,
    )


def test_a_sequence_longer_than_the_draft_rope_is_rejected(tmp_path: Path) -> None:
    options = _options(
        _write(tmp_path / "target", _TARGET),
        _write(tmp_path / "draft", {**_DRAFT, "max_position_embeddings": 1024}),
    )
    checkpoint, method = _load(options.model.path, options.speculative.model_path)

    with pytest.raises(ValueError, match="max_position_embeddings"):
        checkpoint.check(options, EngineTuning(), speculative=method)


@pytest.mark.parametrize("window", [1024, 4096])
def test_sequences_past_a_draft_sliding_window_are_served(tmp_path: Path, window: int) -> None:
    layer_types = ["sliding_attention"] * 4 + ["full_attention"]
    draft = {**_DRAFT, "layer_types": layer_types, "sliding_window": window}
    options = _options(
        _write(tmp_path / "target", {**_TARGET, "max_position_embeddings": 8192}),
        _write(tmp_path / "draft", {**draft, "max_position_embeddings": 8192}),
    )
    checkpoint, method = _load(options.model.path, options.speculative.model_path)

    checkpoint.check(options, EngineTuning(), speculative=method)
    assert checkpoint.model_config.max_position == 8192
    # Resolution reads the draft's KV slot once per distinct window.
    model = method.config.model
    heads = (model.num_qo_heads, model.num_kv_heads, model.head_dim)
    requirements = method.attention_requirements()
    assert requirements == (
        AttentionRequirement.mha(*heads, window=window),
        AttentionRequirement.mha(*heads),
    )
    # The draft's one KV group reads up to its widest window: all keys.
    layout = KVPair(model.num_kv_heads, model.head_dim, torch.bfloat16)
    slots = _state_slots({GroupRole.DRAFT: requirements}, {(GroupRole.DRAFT, "kv"): layout})
    assert slots == {(GroupRole.DRAFT, "kv"): ResolvedSlot("mha", Spec(layout))}


def test_a_slot_of_windowed_requirements_reads_up_to_the_widest() -> None:
    layout = KVPair(1, 64, torch.bfloat16)
    windows = (AttentionRequirement.mha(2, 1, 64, window=w) for w in (512, 2048, 1024))
    slots = _state_slots({GroupRole.DRAFT: tuple(windows)}, {(GroupRole.DRAFT, "kv"): layout})
    assert slots == {(GroupRole.DRAFT, "kv"): ResolvedSlot("mha", Spec(layout, window=2048))}


@pytest.mark.parametrize("num_slots", [1, 7, 256])
def test_round_slot_rows_are_planned_per_slot_and_allocated_once(num_slots: int) -> None:
    from types import SimpleNamespace

    from cti.inference._execution._speculative.dflash import RoundModels, _slot_rows

    # The rounds of a target without recurrent layers, four rows a round.
    models = RoundModels(
        config=None,
        masks=(),
        rounds=RoundShape(width=16, depth=2, max_rows=4),
        max_kv_tokens=0,
        target=SimpleNamespace(iter_recurrent_layers=lambda: ()),
        draft=None,
        head=None,
        sampling=None,
        proposal=None,
        vocab_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    (planned,) = models.state_allocations(num_slots, 0)
    frontiers, done = _slot_rows(num_slots, 4, torch.device("cpu"))

    # A row per request slot and up to a round's padding rows, in the one allocation the
    # plan counted.
    assert frontiers.shape == done.shape == (num_slots + 3,)
    assert frontiers.dtype == torch.int64 and done.dtype == torch.bool
    assert frontiers.untyped_storage().data_ptr() == done.untyped_storage().data_ptr()
    assert frontiers.untyped_storage().nbytes() == planned


def test_rounds_refuse_an_attention_backend_without_fixed_query(tmp_path: Path) -> None:
    target = _write(tmp_path / "target", _TARGET)
    checkpoint, method = _load(target, _write(tmp_path / "draft", _DRAFT))

    # Refused before any provider check, so on any device.
    with pytest.raises(ValueError, match="fa does not run beside fixed-query rounds"):
        checkpoint.plan(
            checkpoint.check(
                _options(target, "d"), EngineTuning(attention_backend="fa"), speculative=method
            ),
            device=torch.device("cpu"),
            capability=DeviceCapability(9, 0),
            device_memory=80 << 30,
            online=False,
        )


def _capturing_runner(monkeypatch: pytest.MonkeyPatch, max_batch_size: int, recurrent=None):
    """A runner whose eager rounds record ``(size, (verification, full, steers), slot,
    captured)``, with fake graphs that remember what was captured into them."""

    from contextlib import contextmanager
    from types import SimpleNamespace

    from cti.inference._execution import _graphs
    from cti.inference._execution._speculative._records import RoundArgs
    from cti.inference._execution._speculative.dflash import SpeculativeRuntime

    capturing: list = []

    class _Graph:
        slot = None

    class _Pool:
        @contextmanager
        def workspace(self, max_rows):
            yield

        @contextmanager
        def capture(self, captured, stream=None):
            capturing.append(captured)
            yield
            capturing.pop()

    stream = SimpleNamespace(synchronize=lambda: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _Graph)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)
    for name in ("synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *_args: None)
    monkeypatch.setattr(_graphs, "get_free_memory", lambda _device: 0)
    runner = object.__new__(SpeculativeRuntime)
    # The pool the test captures through; the runtime itself holds none.
    runner.test_pool = _Pool()
    runner.block_size, runner.max_batch_size, runner._cache_tokens = 16, max_batch_size, 1 << 20
    runner.vocab_size = 32
    runner._device = torch.device("cpu")
    runner._target_backend = SimpleNamespace(map_warm_up=lambda rows: None)
    runner._padding_rows = range(max_batch_size, max_batch_size + 3)
    runner._frontiers = torch.zeros(max_batch_size + 3, dtype=torch.int64)
    runner._done = torch.zeros(max_batch_size + 3, dtype=torch.bool)
    runner._checks = False
    runner._max_seq_len = 64
    runner._plans = {size: size for size in range(1, max_batch_size + 1)}
    runner._draft_graphs, runner._target_graphs, runner._recurrent = {}, {}, recurrent
    runner._scratch = SimpleNamespace(args=RoundArgs.allocate(max_batch_size))
    runner._steering = lambda size, sampling: "steering"
    runner._block_logits = lambda plan: "logits"
    runner._finish = lambda plan, verification, slot, local, processing: None
    rounds: list = []

    def round_(size, verification, slot, full, steering) -> None:
        # Only a full round steers.
        assert full or steering is None
        rounds.append((size, (verification, full, steering is not None), slot, bool(capturing)))

    def draft(plan) -> None:
        capturing[-1].captured = ("draft", plan)

    def target(plan, slot) -> None:
        capturing[-1].captured = ("target", plan, slot)

    runner._round, runner._draft, runner._target = round_, draft, target
    return runner, rounds


def test_a_capture_keys_the_draft_by_size_and_the_target_by_status_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from cti.inference._execution._speculative.dflash import Verification

    # After an odd number of rounds, the next invocation takes status slot 1.
    # Three warm-up kinds (plain, full, full and steering) per verification.
    slots = iter([1, 0] * 3 * len(Verification))
    recurrent = SimpleNamespace(
        depth=2,
        begin=lambda *args: next(slots),
        mark_stale=lambda rows: None,
        clear=lambda: None,
        capture_handoffs=lambda *args: None,
        end=lambda *args: SimpleNamespace(copies=(), settle=lambda: None),
    )
    runner, rounds = _capturing_runner(monkeypatch, 1, recurrent)
    runner.capture([1], stream=None, pool=runner.test_pool)

    # One draft graph per size, whatever the verification or status slot; one target graph
    # per status slot; the tail, verification and all, runs eagerly.
    assert runner._draft_graphs[1].captured == ("draft", 1)
    assert [graph.captured for graph in runner._target_graphs[1]] == [
        ("target", 1, 0),
        ("target", 1, 1),
    ]
    assert not any(captured for *_, captured in rounds)


@pytest.mark.parametrize(
    ("listed", "cache_tokens", "warmed", "captured"),
    [
        # Size 3 runs eagerly, so its rounds compile here, uncaptured; size 8 exceeds the round capacity.
        ([1, 2, 4, 8], 1 << 20, [1, 2, 3, 4], [1, 2, 4]),
        # Without graphs, every size still warms up.
        ([], 1 << 20, [1, 2, 3, 4], []),
        # 32 cache tokens hold the warm-up blocks of at most two Q16 rows.
        ([1, 2, 3, 4], 32, [1, 2], [1, 2]),
        ([1], 15, [], []),
    ],
    ids=["listed", "no-graphs", "small-pools", "nothing-fits"],
)
def test_every_round_size_that_fits_warms_up_and_only_listed_ones_are_captured(
    monkeypatch: pytest.MonkeyPatch,
    listed: list[int],
    cache_tokens: int,
    warmed: list[int],
    captured: list[int],
) -> None:
    from cti.inference._execution._speculative.dflash import Verification

    runner, rounds = _capturing_runner(monkeypatch, 4)
    runner._cache_tokens = cache_tokens
    runner.capture(listed, stream=None, pool=runner.test_pool)

    # Plain, full and full-and-steering rounds: model-dtype and FP32 processing both compile.
    kinds = ((False, False), (True, False), (True, True))
    assert {(size, *kind) for size, kind, _, _ in rounds} == {
        (size, verification, *kind)
        for size, verification, kind in product(warmed, Verification, kinds)
    }
    assert set(runner._draft_graphs) == set(runner._target_graphs) == set(captured)


@pytest.mark.parametrize(("size", "bucket"), [(1, 1), (3, 4), (4, 4)])
def test_a_round_replays_the_smallest_captured_size_that_holds_it(
    monkeypatch: pytest.MonkeyPatch, size: int, bucket: int
) -> None:
    from cti.inference._execution._speculative._records import (
        RoundArgs,
        RoundInput,
        Verification,
    )

    runner, _ = _capturing_runner(monkeypatch, 4)
    runner.capture([1, 2, 4], stream=None, pool=runner.test_pool)
    replayed = []
    for graph in (*runner._draft_graphs.values(), *sum(runner._target_graphs.values(), ())):
        graph.replay = lambda graph=graph: replayed.append(graph.captured)
    finished = []
    runner._finish = lambda plan, *args: finished.append(plan)
    staged = RoundInput(
        RoundArgs.allocate(4), size, Verification.GREEDY, tuple(range(size)), False, ()
    )

    runner.run_round(staged)

    # The bucket's draft and target replay, and its plan runs the eager tail.
    assert replayed == [("draft", bucket), ("target", bucket, 0)]
    assert finished == [bucket]


def test_staging_pads_a_round_with_distinct_done_rows_greedy_and_unfiltered() -> None:
    from cti.inference import Round
    from cti.inference._execution._speculative.dflash import RoundInput
    from cti.inference._step import RequestSpec
    from cti.sampling import TokenSampling

    sampling = TokenSampling(temperature=1.0, top_k=5)
    specs = [RequestSpec(slot, 4, 64, sampling, seed=0) for slot in (2, 0)]
    staged = RoundInput.stage(
        Round((2, 0)), specs, capacity=4, vocab_size=32, padding=range(8, 11), pin_memory=False
    )

    args = staged.args
    assert args.request_indices.tolist() == [2, 0, 8, 9]
    assert args.temperatures[2:].tolist() == [0.0, 0.0]
    assert args.top_k.tolist() == [5, 5, 32, 32]
    assert args.top_p[2:].tolist() == [1.0, 1.0]
    assert args.snapshot_rows.tolist() == [-1] * 4


def test_every_round_size_warms_up_each_verification_at_each_status_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from cti.inference._execution._speculative.dflash import Verification

    class _Rounds:
        """Two status slots that successive rounds alternate, as K2's do."""

        depth = 2
        invocations = 0

        def begin(self, *args: object) -> int:
            self.invocations += 1
            return (self.invocations - 1) % self.depth

        def mark_stale(self, rows: object) -> None:
            pass

        def clear(self) -> None:
            pass

        def capture_handoffs(self, pool: object, stream: object) -> None:
            self.captured_handoffs = True

        def end(self, *args: object) -> SimpleNamespace:
            return SimpleNamespace(copies=(), settle=lambda: None)

    recurrent = _Rounds()
    runner, rounds = _capturing_runner(monkeypatch, 4, recurrent)
    runner.capture([1, 2, 4], stream=None, pool=runner.test_pool)

    # Uncaptured size 3 runs as often as the captured ones: K2 specializes invocation numbers.
    kinds = ((False, False), (True, False), (True, True))
    every_slot = sorted(
        (v.name, *kind, slot) for v in Verification for kind in kinds for slot in (0, 1)
    )
    for size in range(1, 5):
        eager = [
            (v.name, full, steers, slot)
            for s, (v, full, steers), slot, graphed in rounds
            if s == size and not graphed
        ]
        assert sorted(eager) == every_slot
    assert recurrent.invocations == 4 * len(every_slot)
    assert recurrent.captured_handoffs
