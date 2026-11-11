"""Scheduling prompts that hold images: positions and images frozen per step, an image
encoded whole once and loaded by the chunks it is split across.

Images here merge no patches (merge size 1): ``"image:HxW"`` is an image of grid
``(1, H, W)``, which fills ``H * W`` placeholder tokens and has as many pixel rows, and
``"image:n"`` one of grid ``(1, 1, n)``; a ``"#tag"`` suffix names its content.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from unittest.mock import Mock

import pytest
import torch

import cti.inference._control._scheduler as request_scheduler_module
from cti.forward import ImageBatch
from cti.inference._control._cache import CacheManager
from cti.inference._control._engine import step_digest
from cti.inference._control._scheduler import InferenceScheduler
from cti.inference._control._scheduling.encoder_cache import EncoderCache
from cti.inference._control._scheduling.request import Request
from cti.inference._control._scheduling.strategy import SchedulingLimits, TokenSteps
from cti.inference._step import QueryKind, Row
from cti.multimodal import ImageInputs, ImageSpan
from cti.multimodal.qwen3_5 import Qwen3_5RopeLayout
from cti.protocol import (
    AbortBackendMsg,
    RequestRejectedMsg,
    SamplingParams,
    UserMsg,
)
from tests.support.inference.cache import new_cache_manager
from tests.support.inference.inference_steps import engine
from tests.support.inference.prefix_cache import reference_radix_comparison  # noqa: F401
from tests.support.inference.scheduling import request_ids, settle_forward

_PATCH_DIM = 4
_IMAGE_TOKEN_ID = 99


@pytest.fixture(autouse=True)
def _isolate_rank_aware_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(request_scheduler_module, "logger", Mock())


def _prompt(
    *pieces: int | str,
) -> tuple[torch.Tensor, ImageInputs | None, Qwen3_5RopeLayout | None]:
    """Token ids from text runs (ints) and images, with the images' inputs and layout.
    Image i's pixel rows all hold ``i + 1``."""

    ids: list[int] = []
    spans: list[ImageSpan] = []
    for piece in pieces:
        if isinstance(piece, int):
            ids.extend(range(len(ids), len(ids) + piece))
            continue
        size, _, tag = piece.removeprefix("image:").partition("#")
        height, _, width = size.rpartition("x")
        grid = (1, int(height or 1), int(width))
        num_tokens = grid[1] * grid[2]
        spans.append(ImageSpan(len(ids), num_tokens, grid, hashlib.sha256(tag.encode()).digest()))
        ids.extend([_IMAGE_TOKEN_ID] * num_tokens)
    token_ids = torch.tensor(ids, dtype=torch.int32)
    if not spans:
        return token_ids, None, None
    pixels = torch.cat(
        [
            torch.full((span.num_patches, _PATCH_DIM), index + 1, dtype=torch.uint8)
            for index, span in enumerate(spans)
        ]
    )
    return token_ids, ImageInputs(pixels, tuple(spans)), Qwen3_5RopeLayout.from_spans(spans, 1)


def _message(uid: int, *pieces: int | str, max_tokens: int = 4) -> UserMsg:
    token_ids, images, layout = _prompt(*pieces)
    return UserMsg(
        uid=uid,
        input_ids=token_ids,
        sampling=SamplingParams(
            temperature=0, max_new_tokens=max_tokens, ignore_eos=True
        ).snapshot(),
        images=images,
        rope_layout=layout,
    )


def _request(*pieces: int | str) -> Request:
    return Request.from_message(_message(0, *pieces))


class _PinnedEncoderCache(EncoderCache):
    """An encoder cache whose ``pins`` (request uid -> image key) a test reads."""

    def __init__(self, rows: int) -> None:
        super().__init__(rows, checks=True)
        self.pins: dict[int, bytes] = {}

    def pin(self, uid: int, key: bytes | None) -> None:
        super().pin(uid, key)
        if key is None:
            self.pins.pop(uid, None)
        else:
            self.pins[uid] = key


def _scheduler(
    *,
    max_running_requests: int,
    max_batch_tokens: int,
    reuse: bool = False,
    snapshot_rows: range = range(0),
    encoder_rows: int | None = None,
    max_seq_len: int = 128,
    cache: CacheManager | None = None,
    encoder_cache: EncoderCache | None = None,
) -> InferenceScheduler:
    """A scheduler over ``cache`` and ``encoder_cache``, by default new ones of the given
    sizes."""

    if cache is None:
        cache = new_cache_manager(
            max_running_requests * max_seq_len,
            1,
            torch.full((max_running_requests, max_seq_len), -1, dtype=torch.int32),
            reuse=reuse,
            snapshot_rows=snapshot_rows,
            checks=True,
        )
    if encoder_cache is None:
        # As the runner sizes it when an image fits one step's tokens.
        encoder_cache = EncoderCache(encoder_rows or max_batch_tokens, checks=True)
    return InferenceScheduler(
        checks=True,
        limits=SchedulingLimits(
            max_running_requests=max_running_requests,
            max_batch_tokens=max_batch_tokens,
            max_sequence_length=max_seq_len,
            max_prompt_tokens=max_batch_tokens,
        ),
        cache_manager=cache,
        stop_token_ids=(),
        vocab_size=1 << 20,
        decode=TokenSteps(),
        encoder_cache=encoder_cache,
    )


@dataclass(frozen=True)
class _Step:
    """A settled forward as the runner reads it: rows, input ids, images, rotary positions
    and each row's rotary offset."""

    rows: tuple[Row, ...]
    token_ids: torch.Tensor | None
    uids: tuple[int, ...]
    images: ImageBatch | None
    extend_rope_positions: torch.Tensor | None
    rope_offsets: tuple[int, ...]


class _Steps:
    """Plan, submit and settle one step at a time; each step samples token 1 everywhere."""

    def __init__(self, scheduler: InferenceScheduler) -> None:
        self.scheduler = scheduler

    def step(self) -> _Step:
        plan = self.scheduler.plan_next()
        assert plan is not None
        self.scheduler.mark_submitted(plan)
        op = plan.op
        sampled = torch.ones(len(op.rows), dtype=torch.int32)
        settle_forward(self.scheduler, plan, sampled)
        positions = op.extend_rope_positions
        ids = op.extend_token_ids
        return _Step(
            op.rows,
            None if ids is None else torch.frombuffer(bytearray(ids), dtype=torch.int32),
            request_ids(plan),
            None if op.images is None else ImageBatch.unpack(op.images),
            None
            if positions is None
            else torch.frombuffer(bytearray(positions), dtype=torch.int64).view(3, -1),
            tuple(
                entry.request.spec.rope_offset if row.query_kind is QueryKind.DECODE else 0
                for row, entry in zip(op.rows, plan.rows, strict=True)
            ),
        )


def _extends(step: _Step) -> list[tuple[int, int, int]]:
    return [
        (uid, row.kv_prefix_len, row.sequence_len)
        for row, uid in zip(step.rows, step.uids, strict=True)
        if row.query_kind is QueryKind.EXTEND
    ]


# Text [0, 3), a 2 x 3 image [3, 9), text [9, 11), a 1 x 2 image [11, 13), text [13, 14).
_PIECES = (3, "image:2x3", 2, "image:1x2", 1)


def test_an_extension_may_end_inside_an_image_it_then_names() -> None:
    request = _request(*_PIECES)
    first, second = request._image_spans
    request.admit(Mock(), cached_len=0)

    assert request.split_image is None
    assert request.plan_prompt_extension(5) == 5
    assert [span.offset for span, _ in request.planned_images()] == [3]
    assert request.split_image == first.key
    request.mark_submitted()
    assert request.plan_prompt_extension(7) == 7
    assert [span.offset for span, _ in request.planned_images()] == [3, 11]
    assert request.split_image == second.key


def test_a_text_request_extends_by_its_whole_budget() -> None:
    request = _request(13)
    request.admit(Mock(), cached_len=0)

    assert [request.plan_prompt_extension(budget) for budget in (1, 7, 20)] == [1, 7, 5]
    assert request.split_image is None
    assert request.decode_rope_offset == 0
    assert request.rope_layout is None


def test_pixels_are_held_until_the_request_finishes() -> None:
    """A retracted request may prefill its images again, so it keeps their pixels."""

    request = _request(*_PIECES)
    request.admit(Mock(), cached_len=0)
    for budget in (9, 4, 1):
        request.plan_prompt_extension(budget)
        request.mark_submitted()
    assert request.prompt_fully_scheduled and request._pixel_values is not None

    request.mark_finished()
    with pytest.raises(RuntimeError, match="released"):
        request._device_len, request._cached_len = 13, 11
        list(request.planned_images())


def _image_rows(steps: list[_Step]) -> dict[int, int]:
    """Each image token's prompt position and the encoder row it loads, over steps that
    extend one request each."""

    rows = {}
    for step in steps:
        if step.images is not None:
            ((_, start, _),) = _extends(step)
            positions = (start + row for row in step.images.dst.tolist())
            rows.update(zip(positions, step.images.src_rows.tolist(), strict=True))
    return rows


def _prefill(max_batch_tokens: int, *pieces: int | str) -> list[_Step]:
    scheduler = _scheduler(
        max_running_requests=2,
        max_batch_tokens=max_batch_tokens,
        encoder_rows=4096,
        max_seq_len=4096,
    )
    scheduler.handle_message(_message(0, *pieces, max_tokens=1))
    steps = _Steps(scheduler)
    return [steps.step() for _ in range(-(-sum(_prompt(*pieces)[0].shape) // max_batch_tokens))]


@pytest.mark.parametrize("max_batch_tokens", [1024, 2048])
def test_an_image_split_across_chunks_is_encoded_once_and_loaded_as_unsplit(
    max_batch_tokens: int,
) -> None:
    pieces = (10, "image:56x56", 10)
    split = _prefill(max_batch_tokens, *pieces)
    (whole,) = _prefill(4096, *pieces)

    assert [_extends(step) for step in split] == [
        [(0, start, min(start + max_batch_tokens, 3156))]
        for start in range(0, 3156, max_batch_tokens)
    ]
    # The first chunk encodes the whole image; the rest load their part of its rows.
    assert split[0].images.grids == ((1, 56, 56),) == whole.images.grids
    assert all(step.images.grids == () for step in split[1:])
    assert split[0].images.store_rows.tolist() == whole.images.store_rows.tolist()
    assert (
        _image_rows(split)
        == _image_rows([whole])
        == dict(zip(range(10, 3146), range(3136), strict=True))
    )


def test_a_split_image_stays_pinned_until_the_step_past_its_end_is_planned() -> None:
    encoder = _PinnedEncoderCache(4096)
    scheduler = _scheduler(
        max_running_requests=2, max_batch_tokens=1024, encoder_cache=encoder, max_seq_len=4096
    )
    message = _message(0, 10, "image:56x56", 10, max_tokens=1)
    (span,) = message.images.spans
    scheduler.handle_message(message)
    pins = []
    for _ in range(4):
        plan = scheduler.plan_next()
        pins.append(dict(encoder.pins))
        scheduler.mark_submitted(plan)
        sampled = torch.ones(len(plan.op.rows), dtype=torch.int32)
        settle_forward(scheduler, plan, sampled)

    # Chunks end at 1024, 2048 and 3072 inside the image [10, 3146), then past it.
    assert pins == [{0: span.key}] * 3 + [{}]


def test_an_aborted_split_image_unpins_and_its_rows_are_reused() -> None:
    encoder = _PinnedEncoderCache(4096)
    scheduler = _scheduler(
        max_running_requests=2, max_batch_tokens=1024, encoder_cache=encoder, max_seq_len=4096
    )
    steps = _Steps(scheduler)
    scheduler.handle_message(_message(0, 10, "image:56x56#a", 10, max_tokens=1))
    steps.step()
    scheduler.handle_message(AbortBackendMsg(0))
    unpinned = dict(encoder.pins)
    # The pinned image would leave 960 of the 4096 rows; aborted, it makes room.
    message = _message(1, 2, "image:4000#b", 1, max_tokens=1)
    scheduler.handle_message(message)

    first = steps.step()

    assert unpinned == {}
    assert encoder.pins == {1: message.images.spans[0].key}
    assert _extends(first) == [(1, 0, 1024)]
    assert first.images.grids == ((1, 1, 4000),)
    # The 960 free rows, then 3040 of image a's.
    assert first.images.store_rows.tolist() == [*range(3136, 4096), *range(3040)]


def test_a_new_image_waits_for_the_rows_a_split_image_pins() -> None:
    scheduler = _scheduler(max_running_requests=2, max_batch_tokens=8, encoder_rows=20)
    steps = _Steps(scheduler)
    scheduler.handle_message(_message(0, 7, "image:14#a", 1, max_tokens=1))
    scheduler.handle_message(_message(1, "image:8#b", 1, max_tokens=1))

    batches = [steps.step() for _ in range(5)]

    # Image a pins 14 of the 20 rows until the third step's planning: request 1, admitted
    # beside it, waits with its first token on image b, then runs once a is released.
    assert [_extends(batch) for batch in batches] == [
        [(0, 0, 8)],
        [(0, 8, 16)],
        [(0, 16, 22)],
        [(1, 0, 8)],
        [(1, 8, 9)],
    ]
    assert batches[3].images.grids == ((1, 1, 8),)


def test_a_cut_before_an_image_that_must_wait_ends_on_the_snapshot_grid() -> None:
    scheduler = _scheduler(
        max_running_requests=2,
        max_batch_tokens=256,
        reuse=True,
        snapshot_rows=range(2, 4),
        encoder_rows=40,
        max_seq_len=256,
    )
    steps = _Steps(scheduler)
    scheduler.handle_message(_message(0, 10, "image:30#a", 60, "image:30#b", 10, max_tokens=1))

    first, second = steps.step(), steps.step()

    # Image b's rows are not left beside a's: the cut before it, at 100, rounds down to the
    # 64-token grid, and the next forward, starting there, captures at 128.
    assert [_extends(first), _extends(second)] == [[(0, 0, 64)], [(0, 64, 140)]]
    assert [[c.position for c in row.captures] for row in (*first.rows, *second.rows)] == [
        [],
        [128],
    ]


def test_a_cut_that_cannot_reach_the_grid_ends_at_the_image_and_realigns() -> None:
    scheduler = _scheduler(
        max_running_requests=1,
        max_batch_tokens=64,
        reuse=True,
        snapshot_rows=range(2, 4),
        encoder_rows=60,
        max_seq_len=256,
    )
    steps = _Steps(scheduler)
    scheduler.handle_message(_message(0, 10, "image:60#a", 10, "image:30#b", 40, max_tokens=1))

    extends = [_extends(steps.step()) for _ in range(4)]

    # The second forward holds image a's rows, so b waits: its cut at 80 rounds to the
    # forward's own start, so the forward ends at the image instead and the next realigns.
    assert extends == [[(0, 0, 64)], [(0, 64, 80)], [(0, 80, 128)], [(0, 128, 150)]]


def test_a_step_freezes_layout_positions_and_offsets_decodes() -> None:
    scheduler = _scheduler(max_running_requests=4, max_batch_tokens=32)
    steps = _Steps(scheduler)
    image_message = _message(0, *_PIECES, max_tokens=3)
    layout = image_message.rope_layout
    scheduler.handle_message(_message(1, 4, max_tokens=3))
    scheduler.handle_message(image_message)

    prefill = steps.step()
    decode = steps.step()

    assert prefill.uids == (1, 0)
    positions = prefill.extend_rope_positions
    assert positions.dtype == torch.int64 and positions.shape == (3, 4 + 14)
    # The text prompt's tokens sit at their sequence positions on every axis.
    assert positions[:, :4].tolist() == [list(range(4))] * 3
    assert torch.equal(positions[:, 4:], layout.positions(0, 14))
    assert prefill.rope_offsets == (0, 0)
    assert dict(zip(decode.uids, decode.rope_offsets, strict=True)) == {
        0: layout.decode_offset,
        1: 0,
    }
    # Each image advances the positions after it by its longer side, not its tokens.
    assert layout.decode_offset == (3 - 6) + (2 - 2)
    assert decode.extend_rope_positions is None and decode.images is None


def test_text_steps_freeze_no_rope_positions_or_images() -> None:
    scheduler = _scheduler(max_running_requests=2, max_batch_tokens=8)
    scheduler.handle_message(_message(0, 5))

    batch = _Steps(scheduler).step()

    assert batch.extend_rope_positions is None
    assert batch.images is None
    assert batch.rope_offsets == (0,)


def test_a_step_packs_whole_images_with_the_token_rows_they_fill() -> None:
    scheduler = _scheduler(max_running_requests=4, max_batch_tokens=32)
    scheduler.handle_message(_message(1, 4))
    scheduler.handle_message(_message(0, *_PIECES))

    images = _Steps(scheduler).step().images
    assert images.grids == ((1, 2, 3), (1, 1, 2))
    assert images.cu_patches.tolist() == [0, 6, 8]
    assert images.max_patches == 6
    # Image placeholders [3, 9) and [11, 13) of the second extension, after 4 text tokens,
    # take the rows the encoder stores them in.
    assert images.dst.tolist() == [*range(7, 13), *range(15, 17)]
    assert images.store_rows.tolist() == images.src_rows.tolist() == [*range(8)]
    assert images.cached == 0
    assert images.pixel_values[:, 0].tolist() == [1] * 6 + [2] * 2


def test_the_same_image_under_another_question_loads_its_cached_rows() -> None:
    scheduler = _scheduler(max_running_requests=2, max_batch_tokens=32)
    steps = _Steps(scheduler)
    scheduler.handle_message(_message(0, 2, "image:2x3", 3, max_tokens=1))
    first = steps.step()
    # Another question, before and after the image: only the image's content matches.
    scheduler.handle_message(_message(1, 5, "image:2x3", 1, max_tokens=1))

    second = steps.step().images

    assert first.images.store_rows.tolist() == [*range(6)]
    assert second.grids == () and second.pixel_values.shape[0] == 0
    assert second.store_rows.tolist() == []
    assert (second.src_rows.tolist(), second.dst.tolist()) == ([*range(6)], [*range(5, 11)])
    assert second.cached == 6


def test_an_image_twice_in_one_step_is_encoded_once() -> None:
    scheduler = _scheduler(max_running_requests=2, max_batch_tokens=32)
    scheduler.handle_message(_message(0, "image:2x3", 2, "image:2x3", 1))
    scheduler.handle_message(_message(1, 1, "image:2x3", max_tokens=1))

    images = _Steps(scheduler).step().images

    assert images.grids == ((1, 2, 3),) and images.store_rows.tolist() == [*range(6)]
    assert images.src_rows.tolist() == [*range(6)] * 3
    assert images.dst.tolist() == [*range(6), *range(8, 14), *range(16, 22)]


def test_the_least_recently_used_image_makes_room() -> None:
    scheduler = _scheduler(max_running_requests=2, max_batch_tokens=16)
    steps = _Steps(scheduler)
    stored = []
    for uid, image in enumerate(("image:6#a", "image:6#b", "image:6#a", "image:6#c")):
        scheduler.handle_message(_message(uid, 1, image, max_tokens=1))
        stored.append(steps.step().images.store_rows.tolist())
    for uid, image in ((4, "image:6#a"), (5, "image:6#b")):
        scheduler.handle_message(_message(uid, 1, image, max_tokens=1))

    last = steps.step().images

    # Of 16 rows, c takes the 4 free and 2 of b's, which a's load left the least recently
    # used; then b comes back in the 4 left and 2 of c's, a loading first.
    assert stored == [[*range(6)], [*range(6, 12)], [], [12, 13, 14, 15, 6, 7]]
    assert last.store_rows.tolist() == [*range(8, 14)]
    assert last.src_rows.tolist() == [*range(6), *range(8, 14)] and last.cached == 6


@pytest.mark.parametrize("max_batch_tokens", [16, 5], ids=["whole", "split"])
def test_ranks_replaying_one_stream_plan_identical_steps(max_batch_tokens: int) -> None:
    schedulers = [
        _scheduler(max_running_requests=2, max_batch_tokens=max_batch_tokens, encoder_rows=16)
        for _ in range(2)
    ]
    stream = [_message(uid, 2, f"image:6#{uid % 3}", max_tokens=2) for uid in range(6)]
    digests: list[list[bytes]] = []
    for scheduler in schedulers:
        plans = []
        for message in stream:
            scheduler.handle_message(message)
        while (plan := scheduler.plan_next()) is not None:
            plans.append(step_digest(plan.op, plan.admit))
            scheduler.mark_submitted(plan)
            sampled = torch.ones(len(plan.op.rows), dtype=torch.int32)
            settle_forward(scheduler, plan, sampled)
        digests.append(plans)

    assert digests[0] == digests[1] and len(digests[0]) > 3


def test_a_released_engine_encodes_its_images_again() -> None:
    harness = engine(max_batch_tokens=16)
    inference = harness.engine
    for uid in (0, 1):
        inference.handle([_message(uid, 2, "image:6", max_tokens=1)], admitting=True)
        while inference.busy:
            inference.step()
        assert inference.drained()
        inference.release()
        inference.switched(True, None)

    first, second = (ImageBatch.unpack(op.images) for op in harness.executor.forwards)
    assert first.grids == second.grids == ((1, 1, 6),)


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        pytest.param({"rope_layout": None}, "present together", id="no-layout"),
        pytest.param({"images": None}, "present together", id="no-images"),
        pytest.param(
            {"input_ids": torch.arange(10, dtype=torch.int32)}, "within the prompt", id="span"
        ),
    ],
)
def test_malformed_image_requests_are_rejected_at_admission(edit: dict, reason: str) -> None:
    scheduler = _scheduler(max_running_requests=2, max_batch_tokens=32)
    message = _message(7, *_PIECES)
    for name, value in edit.items():
        setattr(message, name, value)

    (rejection,) = scheduler.handle_message(message)

    assert isinstance(rejection, RequestRejectedMsg)
    assert rejection.uid == 7
    assert reason in rejection.reason
    assert not scheduler.has_runnable_work


def test_an_image_larger_than_the_encoder_rows_is_rejected() -> None:
    scheduler = _scheduler(max_running_requests=2, max_batch_tokens=4, encoder_rows=8)

    (rejection,) = scheduler.handle_message(_message(3, 1, "image:9", 1))
    accepted = scheduler.handle_message(_message(4, 1, "image:8", 1))

    assert "an image of 9 tokens exceeds 8 encoder rows" in rejection.reason
    assert accepted == ()
    assert scheduler.has_runnable_work


# The radix alone, and joint with recurrent snapshots on the grid of 64.
_PREFIX_CACHES = pytest.mark.parametrize(
    "snapshot_rows", [range(0), range(3, 5)], ids=["radix", "hybrid"]
)


def _radix(snapshot_rows: range) -> tuple[InferenceScheduler, _Steps, CacheManager]:
    """A scheduler, its steps, and its prefix cache, which a test observes."""

    cache = new_cache_manager(
        3 * 128,
        1,
        torch.full((3, 128), -1, dtype=torch.int32),
        reuse=True,
        snapshot_rows=snapshot_rows,
        checks=True,
    )
    scheduler = _scheduler(max_running_requests=3, max_batch_tokens=128, cache=cache)
    return scheduler, _Steps(scheduler), cache


@_PREFIX_CACHES
def test_the_same_image_prompt_hits_inside_its_image(snapshot_rows: range) -> None:
    scheduler, steps, cache = _radix(snapshot_rows)
    # Text [0, 40), a 6 x 8 image [40, 88), text [88, 98).
    scheduler.handle_message(_message(0, 40, "image:6x8", 10, max_tokens=1))
    steps.step()
    scheduler.handle_message(_message(1, 40, "image:6x8", 10, max_tokens=1))

    step = steps.step()

    # The radix holds all but the final token; the hybrid restores its snapshot at 64,
    # inside the image, whose last 24 tokens load their cached rows.
    hit = 64 if snapshot_rows else 97
    assert _extends(step) == [(1, hit, 98)]
    images = step.images
    loads = None if images is None else (images.grids, images.src_rows.tolist())
    assert loads == (((), [*range(24, 48)]) if snapshot_rows else None)
    assert images is None or images.dst.tolist() == [*range(24)]
    cache.check_integrity()


@_PREFIX_CACHES
def test_a_different_image_of_the_same_size_misses_from_its_first_token(
    snapshot_rows: range,
) -> None:
    scheduler, steps, _ = _radix(snapshot_rows)
    scheduler.handle_message(_message(0, 70, "image:4x8#a", 4, max_tokens=1))
    steps.step()
    scheduler.handle_message(_message(1, 70, "image:4x8#b", 4, max_tokens=1))

    step = steps.step()

    assert _extends(step) == [(1, 64 if snapshot_rows else 70, 106)]
    assert step.images.grids == ((1, 4, 8),) and step.images.cached == 0


@_PREFIX_CACHES
def test_image_and_text_prompts_in_one_step_both_hit_their_shared_text(
    snapshot_rows: range,
) -> None:
    scheduler, steps, _ = _radix(snapshot_rows)
    scheduler.handle_message(_message(0, 100, max_tokens=1))
    steps.step()
    image_message = _message(1, 70, "image:4x8", 4, max_tokens=1)
    scheduler.handle_message(image_message)
    scheduler.handle_message(_message(2, 110, max_tokens=1))

    step = steps.step()

    image_hit, text_hit = (64, 64) if snapshot_rows else (70, 100)
    assert _extends(step) == [(1, image_hit, 106), (2, text_hit, 110)]
    # The model reads the prompt's own ids, placeholders where the image's encoder output
    # goes, not its prefix-cache key ids.
    image_ids = image_message.input_ids[image_hit:]
    assert step.token_ids.tolist() == [*image_ids.tolist(), *range(text_hit, 110)]
    assert int(image_message.input_ids[70]) == _IMAGE_TOKEN_ID


@pytest.mark.parametrize("cached", [True, False], ids=["cached", "evicted"])
def test_a_hit_inside_an_image_loads_its_rows_or_encodes_it_whole(cached: bool) -> None:
    scheduler = _scheduler(
        max_running_requests=3, max_batch_tokens=80, reuse=True, snapshot_rows=range(3, 5)
    )
    steps = _Steps(scheduler)
    scheduler.handle_message(_message(0, 40, "image:6x8", 10, max_tokens=1))
    steps.step(), steps.step()
    if not cached:
        # 40 more encoder rows evict the first image's 48 of the 80.
        scheduler.handle_message(_message(3, "image:5x8#other", 1, max_tokens=1))
        steps.step()
    scheduler.handle_message(_message(1, 40, "image:6x8", 10, max_tokens=1))
    scheduler.handle_message(_message(2, "image:5x6", 1, max_tokens=1))

    first = steps.step()

    # Restored at 64, request 1 extends 34 tokens beside request 2. Cached, it loads its
    # last 24 rows; evicted, it encodes all 48 of its image again and loads the same part.
    assert _extends(first) == [(1, 64, 98), (2, 0, 31)]
    images = first.images
    if cached:
        assert images.grids == ((1, 5, 6),) and images.cached == 24
        assert images.src_rows.tolist()[:24] == [*range(24, 48)]
    else:
        assert images.grids == ((1, 6, 8), (1, 5, 6)) and images.cached == 0
        assert images.src_rows.tolist()[:24] == images.store_rows.tolist()[24:48]
    assert images.dst.tolist()[:24] == [*range(24)]


def test_a_hit_inside_an_image_whose_rows_are_not_left_waits() -> None:
    scheduler = _scheduler(
        max_running_requests=3, max_batch_tokens=64, reuse=True, snapshot_rows=range(3, 5)
    )
    steps = _Steps(scheduler)
    scheduler.handle_message(_message(0, 40, "image:6x8", 10, max_tokens=1))
    while scheduler.has_runnable_work:
        steps.step()
    # 40 more encoder rows evict the first image's 48 of the 64.
    scheduler.handle_message(_message(3, "image:5x8#other", 1, max_tokens=1))
    steps.step()
    scheduler.handle_message(_message(2, "image:1x30#x", 1, max_tokens=1))
    scheduler.handle_message(_message(1, 40, "image:6x8", 10, max_tokens=1))

    first = steps.step()

    # Restored at 64, request 1 encodes its whole image again: its 48 encoder rows do not
    # fit beside request 2's 30 of the 64, so it opens and waits for the next step.
    assert _extends(first) == [(2, 0, 31)]
    assert _extends(steps.step())[0] == (1, 64, 98)


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(temperature=0, max_new_tokens=1, repetition_penalty=1.5),
        SamplingParams(temperature=0, max_new_tokens=1, prompt_logprobs=0),
    ],
    ids=["restored-prompt", "prompt-targets"],
)
def test_logits_processing_and_prompt_scoring_read_vocabulary_ids(
    params: SamplingParams,
) -> None:
    scheduler, steps, _ = _radix(range(0))
    scheduler.handle_message(_message(0, 40, "image:6x8", 10, max_tokens=1))
    steps.step()
    message = _message(1, 40, "image:6x8", 10)
    scheduler.handle_message(replace(message, sampling=params.snapshot()))

    op = scheduler.plan_next().op

    # The prompt's own ids: the image's placeholders, not its prefix-cache key ids.
    ids = message.input_ids
    if params.prompt_logprobs is None:
        ((_, restored),) = op.restored_prompts
        assert (
            torch.frombuffer(bytearray(restored), dtype=torch.int32).tolist() == ids[:97].tolist()
        )
    else:
        targets = torch.frombuffer(bytearray(op.prompt_targets), dtype=torch.int32)
        assert targets.tolist() == [*ids[1:].tolist(), -1]
