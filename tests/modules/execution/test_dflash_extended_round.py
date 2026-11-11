"""A DFlash round whose target forward runs a step's prompt extensions first."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cti.attention import MixedQuery
from cti.forward import ForwardInputs
from cti.inference import Sampled
from cti.inference._execution._graphs import forward_rows
from cti.inference._execution._sampling import DeviceSampled
from cti.inference._execution._speculative._records import RoundResult, Verification
from cti.inference._execution._speculative.dflash import (
    SpeculativeRuntime,
    TargetForward,
    _RoundPlan,
    round_write_fault,
)
from cti.inference._step import QueryKind, Row
from cti.recurrent import RecurrentMetadata

_V = 7


class _Head:
    """An LM head that records its entries: logits are each row's hidden sum + token id."""

    def __init__(self, calls: list) -> None:
        self.calls = calls

    def local_logits(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append(("local", x.clone()))
        return x.sum(-1, keepdim=True) + torch.arange(_V)

    def gather_logits(self, local: torch.Tensor) -> torch.Tensor:
        self.calls.append(("logits", len(local)))
        return local

    def gather_argmax(self, local: torch.Tensor) -> torch.Tensor:
        self.calls.append(("argmax", len(local)))
        return local.argmax(-1)


def _runner(
    calls: list, *, greedy_rows: int = 8, hidden: torch.Tensor | None = None
) -> tuple[SpeculativeRuntime, _RoundPlan]:
    runner = object.__new__(SpeculativeRuntime)
    runner._recurrent = None
    verify = ForwardInputs(
        torch.full((greedy_rows,), 9, dtype=torch.int32),
        torch.arange(50, 50 + greedy_rows),
        SimpleNamespace(
            cache_write_indices=torch.arange(100, 100 + greedy_rows, dtype=torch.int32)
        ),
        2,
        ragged=False,
    )
    plan = _RoundPlan(2, (), (), verify, None, torch.arange(40, 40 + greedy_rows))
    if hidden is None:
        hidden = torch.arange(5 + greedy_rows, dtype=torch.float32)[:, None].expand(-1, 3)
    runner.block_size = greedy_rows // plan.size
    draft = SimpleNamespace(
        store_context=lambda features, positions, batch: calls.append(
            ("context", features, positions, batch)
        )
    )
    runner._models = SimpleNamespace(target=SimpleNamespace(), head=_Head(calls), draft=draft)
    runner._scratch = SimpleNamespace(taps=torch.empty(16, 2), verify_hidden=torch.empty(16, 3))
    runner._forward_hidden = lambda rows: calls.append(("forward", rows)) or hidden
    runner._draft_graphs = {}
    runner._draft = lambda plan: calls.append("head")
    runner._verify_rows = lambda plan, slot: plan.verify
    runner._commit = lambda plan, verification, slot, ids: calls.append(("commit", ids))
    runner._advance = lambda forward: calls.append(("advance", forward))
    runner._sample = lambda size, logits, filters, *, min_p: calls.append(
        ("sample", size, len(logits))
    )
    return runner, plan


def _extension(calls: list, *, axes: int = 1) -> TargetForward:
    batch = ForwardInputs(
        torch.arange(5, dtype=torch.int32),
        torch.arange(5) if axes == 1 else torch.arange(15).view(3, 5),
        SimpleNamespace(
            cache_write_indices=torch.tensor([14, 15, 30, 31, 32], dtype=torch.int32),
        ),
        2,
        ragged=True,
        recurrent_metadata=RecurrentMetadata(torch.zeros(2, dtype=torch.int32), 2, 5),
        logit_rows=torch.tensor([1, 4]),
    )
    rows = forward_rows((Row(1, QueryKind.EXTEND, 4, 6), Row(3, QueryKind.EXTEND, 0, 3)))
    sampled = DeviceSampled(torch.tensor([5, 6], dtype=torch.int32))

    def sample(logits: torch.Tensor, hidden: torch.Tensor):
        calls.append(("sample_extension", logits.clone()))
        return sampled

    positions = torch.tensor([4, 5, 0, 1, 2])
    return TargetForward(batch, rows, positions, torch.tensor([1, 1, 3, 3, 3]), sample)


@pytest.mark.parametrize("verification", list(Verification))
def test_one_head_serves_the_extensions_and_the_blocks_and_context_is_stored_once(
    verification: Verification,
) -> None:
    calls: list = []
    runner, plan = _runner(calls)
    extension = _extension(calls)

    sampled = runner._extended_round(plan, verification, 1, extension, False, None)

    names = [call if isinstance(call, str) else call[0] for call in calls]
    verify = "argmax" if verification is Verification.GREEDY else "sample"
    expected = ["head", "forward", "local", "logits", "sample_extension", "context"]
    expected += ["argmax"] if verify == "argmax" else ["logits", "sample"]
    assert names == [*expected, "commit", "advance"]
    assert sampled.token_ids.tolist() == [5, 6]
    # One LM head over each extension's last row, then the block rows.
    (local,) = [call[1] for call in calls if call[0] == "local"]
    assert local[:, 0].tolist() == [1, 4, *range(5, 13)]
    (logits,) = [call[1] for call in calls if call[0] == "sample_extension"]
    assert logits.shape == (2, _V) and logits[:, 0].tolist() == [3.0, 12.0]
    # The draft context of the extension tokens, then of the blocks, in one store.
    # The joint forward's batch writes them at its slots, the extension's then the blocks'.
    (context,) = [call for call in calls if call[0] == "context"]
    _, features, positions, batch = context
    assert positions.tolist() == [4, 5, 0, 1, 2, *range(40, 48)]
    assert len(features) == 13 and features.data_ptr() == runner._scratch.taps.data_ptr()
    assert [batch] == [call[1] for call in calls if call[0] == "forward"]
    writes = batch.attn_metadata.cache_write_indices
    assert writes.tolist() == [14, 15, 30, 31, 32, *range(100, 108)]
    (commit,) = [call for call in calls if call[0] == "commit"]
    if verification is Verification.GREEDY:
        assert commit[1].tolist() == [_V - 1] * 8
    else:
        assert commit[1] is None
    assert calls[-1] == ("advance", extension)


@pytest.mark.parametrize("axes", [1, 3])
def test_the_extension_runs_ahead_of_the_rows_in_one_target_forward(axes: int) -> None:
    calls: list = []
    runner, plan = _runner(calls)
    extension = _extension(calls, axes=axes)

    runner._extended_round(plan, Verification.GREEDY, 0, extension, False, None)

    (joint,) = [call[1] for call in calls if call[0] == "forward"]
    # Ragged, so a breakable graph replays it when its tokens fit a bucket.
    assert joint.ragged and joint.images is None
    # Its images encode as the step's forward would, through the step's vision graphs.
    assert joint.image_encoding is extension.batch.image_encoding
    assert joint.input_ids.tolist() == [0, 1, 2, 3, 4] + [9] * 8
    writes = joint.attn_metadata.cache_write_indices
    assert writes.tolist() == [14, 15, 30, 31, 32, *range(100, 108)]
    if axes == 1:
        assert joint.rope_positions.tolist() == [*range(5), *range(50, 58)]
    else:
        # Each block row's one position on every M-RoPE axis.
        assert joint.rope_positions.shape == (3, 13)
        assert all(row[5:].tolist() == list(range(50, 58)) for row in joint.rope_positions)
    assert isinstance(joint.attn_metadata, MixedQuery)
    assert joint.attn_metadata.paged is extension.batch.attn_metadata
    assert joint.attn_metadata.fixed is plan.verify.attn_metadata
    assert joint.attn_metadata.paged_tokens == 5
    assert joint.recurrent_metadata.num_extend_tokens == 5


def test_an_extended_round_replays_its_sizes_captured_draft_at_any_verification() -> None:
    calls: list = []
    runner, plan = _runner(calls)
    graph = Mock()
    graph.replay.side_effect = lambda: calls.append("replay")

    runner._extended_round(plan, Verification.GREEDY, 0, _extension(calls), False, None)
    runner._draft_graphs = {plan.size: graph}
    runner._extended_round(plan, Verification.GREEDY, 0, _extension(calls), False, None)
    runner._extended_round(plan, Verification.SAMPLED, 1, _extension(calls), False, None)

    heads = [call for call in calls if call in ("replay", "head")]
    assert heads == ["head", "replay", "replay"]


def test_the_verify_rows_are_their_status_slots_verify() -> None:
    runner = object.__new__(SpeculativeRuntime)
    ids, positions = torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int64)
    verify = ForwardInputs(ids, positions, Mock(), 1, ragged=False)
    plain = _RoundPlan(1, (), (), verify, None, (positions, ids))
    slots = (replace(verify, recurrent_metadata=Mock()), replace(verify, recurrent_metadata=Mock()))
    recurrent = _RoundPlan(1, (), (), verify, slots, (positions, ids))

    assert runner._verify_rows(plain, 1) is verify
    assert runner._verify_rows(recurrent, 1) is slots[1]


def test_a_target_forward_wider_than_the_taps_is_refused() -> None:
    runner = object.__new__(SpeculativeRuntime)
    runner._plans = {2: object()}
    runner.block_size = 16
    runner._scratch = SimpleNamespace(taps=torch.empty(40, 4))
    extension = SimpleNamespace(token_rows=torch.zeros(9, dtype=torch.int64))

    with pytest.raises(ValueError, match="more rows than max_batch_tokens"):
        runner.run_round(SimpleNamespace(size=2), extension)


def test_an_extended_round_completes_with_its_sampled_tokens() -> None:
    settled = []
    records = SimpleNamespace(settle=lambda slot, size, sampled: settled.append(sampled) or size)
    sampled = DeviceSampled(torch.tensor([5, 6], dtype=torch.int32))

    assert RoundResult(Mock(), 2, 0, records, None, sampled, ()).output() == 2
    assert RoundResult(Mock(), 3, 1, records, None, None, ()).output() == 3

    tokens = torch.tensor([5, 6], dtype=torch.int32).numpy().tobytes()
    assert settled == [Sampled(tokens), None]


@pytest.mark.parametrize("verification", list(Verification))
def test_a_plain_round_keeps_its_order_draft_target_context_head_verify_commit(
    verification: Verification,
) -> None:
    calls: list = []
    runner, plan = _runner(calls)
    runner._models.target.forward_hidden = lambda rows: (
        calls.append(("forward", rows)) or torch.ones(8, 3)
    )

    runner._round(plan, verification, 0, False, None)

    # The rows pick their tokens as beside prompt extensions: from the head's shard logits,
    # of the hidden rows the target forward left in ``verify_hidden``.
    names = [call if isinstance(call, str) else call[0] for call in calls]
    middle = ["argmax"] if verification is Verification.GREEDY else ["logits", "sample"]
    assert names == ["head", "forward", "context", "local", *middle, "commit"]
    (local,) = [call[1] for call in calls if call[0] == "local"]
    assert torch.equal(local, torch.ones(8, 3))
    # The blocks' context is stored at the verify rows' write slots.
    (context,) = [call for call in calls if call[0] == "context"]
    assert context[2] is plan.context_positions and context[3] is plan.verify
    assert len(context[1]) == 8


def test_a_full_rounds_logit_rows_read_the_frontiers_its_draft_placed() -> None:
    calls: list = []
    runner, plan = _runner(calls)
    runner._target = lambda plan, slot: calls.append("target")
    runner._block_logits = lambda plan: torch.zeros(8, _V)
    runner._logit_rows = lambda size, steering: calls.append(("rows", size, steering))
    runner._verify = lambda plan, verification, local, processing: (
        calls.append("verify") or SimpleNamespace(target_ids=None, logits=None)
    )

    runner._round(plan, Verification.SAMPLED, 0, True, "steering")

    assert calls[:4] == ["head", "target", ("rows", plan.size, "steering"), "verify"]


def test_a_round_writing_unmapped_or_shared_slots_is_a_fault() -> None:
    def fault(writes, done=(False, False)) -> bool:
        return bool(
            round_write_fault(
                torch.tensor(writes, dtype=torch.int32),
                torch.tensor([0, 1, 2, 3]),
                torch.tensor(done),
                cache_tokens=64,
                max_seq_len=3,
            )
        )

    # A live row writes mapped slots; past max_seq_len and in a done row, guard slots.
    assert not fault([10, 11, 20, 70])
    assert not fault([10, 11, 64, 64], done=(False, True))
    # Unmapped table entries name the sink slot (cache_tokens); two rows may not share one.
    assert fault([10, 64, 20, 21])
    assert fault([10, 11, 11, 21])


def test_a_faulty_round_fails_where_it_settles() -> None:
    records = SimpleNamespace(settle=lambda *args: None)
    fault = ("a round wrote unmapped or shared cache slots", torch.tensor(True))
    result = RoundResult(Mock(), 1, 0, records, None, None, (), (fault,))

    with pytest.raises(RuntimeError, match="unmapped or shared"):
        result.output()


def test_the_extended_warm_up_runs_one_greedy_round_of_row_zero_between_row_resets() -> None:
    from cti.inference._execution._speculative._records import RoundArgs

    calls: list = []
    runner = object.__new__(SpeculativeRuntime)
    runner._scratch = SimpleNamespace(args=RoundArgs.allocate(4))
    runner._target_backend = SimpleNamespace(map_warm_up=lambda rows: calls.append(("map", rows)))
    runner.vocab_size, runner._max_seq_len, runner._recurrent = 32, 64, None
    runner._plans = {1: "plan-1", 2: "plan-2"}
    runner.reset_rows = lambda: calls.append("reset")

    def extended_round(plan, verification, slot, extension, full, steering) -> None:
        assert runner._scratch.args.request_indices[0] == 0
        calls.append(("round", plan, verification, slot, extension, full, steering))

    runner._extended_round = extended_round
    extension = object()

    runner.warm_up_extended(extension)

    assert calls == [
        ("map", 1),
        "reset",
        ("round", "plan-1", Verification.GREEDY, 0, extension, False, None),
        "reset",
    ]
