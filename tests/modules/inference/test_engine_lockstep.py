"""Checked ranks compare every planned step before any of them runs it."""

from __future__ import annotations

import pickle
from dataclasses import replace
from unittest.mock import ANY, Mock, call

import pytest

from cti.inference import Forward
from cti.inference._control._engine import step_digest
from cti.inference._step import QueryKind, RequestSpec, Row
from tests.support.inference.inference_steps import engine, user_msg


@pytest.mark.parametrize("diverged", (False, True), ids=("agree", "diverge"))
def test_ranks_compare_each_planned_step_before_running_it(diverged: bool) -> None:
    divergence = RuntimeError("tensor-parallel ranks diverged at planned step 0 (Forward)")
    agree_on_step = Mock(side_effect=divergence if diverged else None)
    harness = engine(agree_on_step=agree_on_step)
    harness.engine.handle([user_msg(7, [11, 12, 13], max_tokens=1)], admitting=True)

    if diverged:
        with pytest.raises(RuntimeError) as raised:
            harness.engine.step()
        assert raised.value is divergence
        assert harness.executor.forwards == []
    else:
        harness.engine.step()
        assert len(harness.executor.forwards) == 1
    assert agree_on_step.call_args_list == [call(ANY, "planned step 0 (Forward)")]


def test_equal_steps_digest_alike_however_their_values_are_shared() -> None:
    """A rank's step may share one bytes object where a peer's holds equal copies."""

    tokens = bytes(range(8))
    row = Row(0, QueryKind.EXTEND, 0, 2, write_position=2)
    shared = Forward((row, row), extend_token_ids=tokens, extend_rope_positions=tokens)
    copies = Forward(
        (Row(0, QueryKind.EXTEND, 0, 2, write_position=2), replace(row)),
        extend_token_ids=bytes(bytearray(tokens)),
        extend_rope_positions=bytes(bytearray(tokens)),
    )
    spec = RequestSpec(0, 2, 4)

    assert pickle.dumps((shared, (spec, spec))) != pickle.dumps((copies, (spec, replace(spec))))
    assert step_digest(shared, (spec, spec)) == step_digest(copies, (spec, replace(spec)))
    assert step_digest(shared, (spec,)) != step_digest(shared, (replace(spec, seed=1),))
