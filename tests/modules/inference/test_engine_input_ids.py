"""An image's prefix-cache key ids, negative, never leave the scheduler: the model reads 0
there, and under checks the engine refuses any step whose input ids hold one."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from cti.inference import Forward, Round
from cti.inference._step import QueryKind, Row
from cti.multimodal import ImageInputs, ImageSpan
from cti.multimodal.qwen3_5 import Qwen3_5RopeLayout
from tests.support.inference.inference_steps import engine, user_msg


def test_a_checked_image_step_reaches_the_model_with_its_placeholder_ids() -> None:
    span = ImageSpan(2, 4, (1, 2, 2), bytes(32))
    message = user_msg(7, [11, 12, 99, 99, 99, 99, 13], max_tokens=1)
    message.images = ImageInputs(torch.zeros(4, 4, dtype=torch.uint8), (span,))
    message.rope_layout = Qwen3_5RopeLayout.from_spans((span,), 1)
    harness = engine()

    harness.engine.handle([message], admitting=True)
    harness.engine.step()

    (forward,) = harness.executor.forwards
    assert list(memoryview(forward.extend_token_ids).cast("i")) == [11, 12, 99, 99, 99, 99, 13]


@pytest.mark.parametrize("ids", [[5, 0], [5, -1]], ids=["vocabulary", "key"])
@pytest.mark.parametrize("in_round", [False, True], ids=["forward", "round"])
def test_forwards_and_rounds_check_their_input_ids(ids: list[int], in_round: bool) -> None:
    forward = Forward(
        (Row(0, QueryKind.EXTEND, 0, 2, write_position=2),),
        extend_token_ids=torch.tensor(ids, dtype=torch.int32).numpy().tobytes(),
    )
    op = Round(slots=(1,), extend=forward) if in_round else forward

    refused = pytest.raises(RuntimeError, match="image key id")
    with refused if min(ids) < 0 else nullcontext():
        op.require_vocabulary_ids()
