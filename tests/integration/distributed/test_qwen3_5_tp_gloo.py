"""Tensor-parallel Qwen3.5 across real processes and gloo groups on CPU.

A tiny image-text checkpoint with two Gated DeltaNet key heads (four value heads) and
two full-attention layers runs a chunked prefill, a mixed batch and decode steps over
the runner's inference state, with the CPU reference operations at every kernel seam.
A second scenario prefills an image prompt beside a text prompt, its images encoded
whole by the vision tower every rank holds, then decodes it at its rotary offset.
A third restores a joint prefix snapshot each rank captured for its own heads.
TP2 must match the single-rank model: logits at every step, and every request's
recurrent state and snapshot once each rank's heads are put back in place.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import transformers

from cti.distributed.runtime import ParallelGroup
from cti.forward import ImageBatch, RecurrentExport
from cti.kvcache import RecurrentStatePool
from cti.multimodal import ImageSpan
from cti.multimodal.qwen3_5 import Qwen3_5RopeLayout
from tests.support.forward_rows import Entry, decode_entry, extend_entry
from tests.support.gloo_ranks import gloo_ranks
from tests.support.multimodal import QWEN3_5_IMAGE_PREPROCESSING
from tests.support.qwen3_5 import TINY_QWEN3_5_CONFIG, load_hf_config, randomize_hf_weights
from tests.support.qwen3_5_engine import (
    TP_TOLERANCE,
    EagerEngine,
    assert_ranks_match_single_rank,
    bind_reference_kernels,
    load_model,
    run_steps,
)

# Gloo ranks are processes of their own: one such test at a time keeps them from starving.
pytestmark = pytest.mark.xdist_group("gloo")

_SIZE = 2
_HEAD_DIM = 128
# Both mixers twice. Each rank owns one key head and the two value heads that read it,
# two query heads with their gates and one KV head; the vocabulary leaves a padded row.
_CONFIG = copy.deepcopy(dict(TINY_QWEN3_5_CONFIG))
_CONFIG["text_config"].update(
    vocab_size=67,
    max_position_embeddings=256,
    num_hidden_layers=4,
    layer_types=["linear_attention", "full_attention"] * 2,
    num_attention_heads=4,
    num_key_value_heads=2,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
)
# The image prompt: text runs around three images whose patches the tower does not merge
# (spatial_merge_size 1), so each image's tokens are its patches.
_IMAGE_TEXT_RUNS = (3, 2, 2, 3)
_IMAGE_GRIDS = ((1, 2, 3), (1, 3, 2), (1, 2, 2))
_IMAGE_PROMPT_LEN = sum(_IMAGE_TEXT_RUNS) + sum(t * h * w for t, h, w in _IMAGE_GRIDS)
_TEXT_PROMPT_LEN = 9


def _image_spans() -> tuple[ImageSpan, ...]:
    spans, offset = [], 0
    for text_len, grid in zip(_IMAGE_TEXT_RUNS, _IMAGE_GRIDS, strict=False):
        offset += text_len
        num_tokens = grid[0] * grid[1] * grid[2]
        spans.append(ImageSpan(offset, num_tokens, grid, bytes(32)))
        offset += num_tokens
    return tuple(spans)


def _image_steps() -> tuple[
    torch.Tensor, list[tuple[tuple[Entry, ...], torch.Tensor, ImageBatch | None]]
]:
    """The token table and each step's entries, rotary rows and images.

    Request 1, a text prompt, prefills first, so the first two images fill rows after its
    tokens; request 0's prefill ends where its second image does. Its last image follows
    in the next step, beside a decode, and then both decode, request 0 at its layout's
    offset.
    """

    generator = torch.Generator().manual_seed(3)
    vocab_size = _CONFIG["text_config"]["vocab_size"]
    token_ids = torch.randint(
        vocab_size, (2, _IMAGE_PROMPT_LEN + 2), generator=generator, dtype=torch.int32
    )
    vision = _CONFIG["vision_config"]
    patch_dim = 3 * vision["patch_size"] ** 2
    spans = _image_spans()
    pixels = [
        torch.randint(256, (span.num_tokens, patch_dim), generator=generator, dtype=torch.uint8)
        for span in spans
    ]
    layout = Qwen3_5RopeLayout.from_spans(spans, vision["spatial_merge_size"])
    assert layout.decode_offset != 0

    def text_rows(start: int, end: int) -> torch.Tensor:
        return torch.arange(start, end).expand(3, -1)

    def images(indices: tuple[int, ...], *, start: int, row: int) -> ImageBatch:
        """Request 0's images ``indices`` in a step whose row ``row`` is its position ``start``."""

        rows = [torch.arange(spans[i].offset, spans[i].end) - start + row for i in indices]
        stored = range(sum(len(image_rows) for image_rows in rows))
        packed = ImageBatch.pack(
            [(pixels[i], spans[i].grid_thw) for i in indices], stored, stored, torch.cat(rows)
        )
        return ImageBatch.unpack(packed)

    first_end = spans[1].end
    steps = [
        (
            (extend_entry(1, 0, _TEXT_PROMPT_LEN), extend_entry(0, 0, first_end)),
            torch.cat([text_rows(0, _TEXT_PROMPT_LEN), layout.positions(0, first_end)], 1),
            images((0, 1), start=0, row=_TEXT_PROMPT_LEN),
        ),
        (
            (extend_entry(0, first_end, _IMAGE_PROMPT_LEN), decode_entry(1, _TEXT_PROMPT_LEN)),
            torch.cat(
                [
                    layout.positions(first_end, _IMAGE_PROMPT_LEN),
                    text_rows(_TEXT_PROMPT_LEN, _TEXT_PROMPT_LEN + 1),
                ],
                1,
            ),
            images((2,), start=first_end, row=0),
        ),
    ]
    for step in range(2):
        position = _IMAGE_PROMPT_LEN + step
        text_position = _TEXT_PROMPT_LEN + 1 + step
        rows = torch.tensor([position + layout.decode_offset, text_position]).expand(3, -1)
        steps.append(((decode_entry(0, position), decode_entry(1, text_position)), rows, None))
    return token_ids, steps


def _joint_hit(
    folder: Path, group: ParallelGroup | None
) -> tuple[RecurrentStatePool, list[torch.Tensor]]:
    """A 193-token prompt prefilled in 64-token chunks exports its state at 192 into row 2,
    and a second request restores it. The pool, then the logits of the hit, of the first
    request's next step and of a cold prefill."""

    generator = torch.Generator().manual_seed(5)
    tokens = torch.randint(
        _CONFIG["text_config"]["vocab_size"], (1, 193), generator=generator, dtype=torch.int32
    ).repeat(2, 1)
    engine = EagerEngine(load_model(folder, group=group), tokens, snapshot_rows=1)
    for start in range(0, 192, 64):
        end = start + 64
        captures = (RecurrentExport(192, 2),) if end == 192 else ()
        engine.execute(extend_entry(0, start, end, captures=captures))
    # The hit reads the first request's KV rows, as a radix hit does.
    engine._cache_index_table[1, :192] = engine._cache_index_table[0, :192]
    hit = engine.execute(extend_entry(1, 192, 193, restore_row=2))
    first = engine.execute(extend_entry(0, 192, 193))
    cold = EagerEngine(load_model(folder, group=group), tokens)
    return engine.recurrent_state, [hit, first, cold.execute(extend_entry(1, 0, 193))]


def _run(folder: Path, group: ParallelGroup | None) -> dict[str, list]:
    """Every step's logits, then each Gated DeltaNet layer's state and history pools (the
    steps' slots, then the joint hit's slots and snapshot row), and every image step's and
    the joint hit's logits."""

    engine, logits = run_steps(folder, group=group)
    with torch.inference_mode():
        prefix_pool, prefix_logits = _joint_hit(folder, group)
    pools = (engine.recurrent_state, prefix_pool)
    token_ids, image_steps = _image_steps()
    image_engine = EagerEngine(
        load_model(folder, group=group, image_preprocessing=QWEN3_5_IMAGE_PREPROCESSING),
        token_ids,
        image_rows=_IMAGE_PROMPT_LEN,
    )
    with torch.inference_mode():
        image_logits = [
            image_engine.execute(*entries, rope_positions=rows, images=images).tolist()
            for entries, rows, images in image_steps
        ]
    # Tensors would cross the Pipe as shared memory owned by an exiting rank.
    return {
        "logits": [step.tolist() for step in logits],
        "states": [p.layer_views(i)[0].tolist() for p in pools for i in range(p.num_layers)],
        "histories": [p.layer_views(i)[1].tolist() for p in pools for i in range(p.num_layers)],
        "image_logits": image_logits,
        "prefix_logits": [step.tolist() for step in prefix_logits],
    }


def _rank(group: ParallelGroup, folder: Path) -> dict[str, list]:
    # The rank process exits after the scenario, so its bindings are never undone.
    bind_reference_kernels(pytest.MonkeyPatch())
    return _run(folder, group)


def _assemble_histories(histories: list[torch.Tensor]) -> torch.Tensor:
    """Each rank holds its heads' q | k | v channels; restore the full channel order."""

    text = _CONFIG["text_config"]
    key = text["linear_num_key_heads"] // _SIZE * _HEAD_DIM
    value = text["linear_num_value_heads"] // _SIZE * _HEAD_DIM
    parts = [history.split((key, key, value), dim=-1) for history in histories]
    return torch.cat([rank_parts[part] for part in range(3) for rank_parts in parts], dim=-1)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Transformers' image-text model with random weights, saved with its prefixes."""

    folder = tmp_path_factory.mktemp("qwen3_5-tp")
    reference = transformers.Qwen3_5ForConditionalGeneration(load_hf_config(_CONFIG, folder))
    # The engine declares a BF16 convolution weight.
    randomize_hf_weights(reference, conv_dtype=torch.bfloat16)
    reference.save_pretrained(folder)
    return folder


@pytest.fixture(scope="module")
def single_rank(checkpoint: Path) -> dict[str, list]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        bind_reference_kernels(monkeypatch)
        return _run(checkpoint, None)


@pytest.fixture(scope="module")
def ranks(checkpoint: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[list[dict]]:
    rendezvous = tmp_path_factory.mktemp("rendezvous")
    with gloo_ranks(_rank, _SIZE, checkpoint, rendezvous) as results:
        yield results


def test_tensor_parallel_logits_match_the_single_rank_model(
    ranks: list[dict], single_rank: dict[str, list]
) -> None:
    reference = [torch.tensor(step) for step in single_rank["logits"]]
    assert all(step.shape == (2, _CONFIG["text_config"]["vocab_size"]) for step in reference)
    assert_ranks_match_single_rank([result["logits"] for result in ranks], reference)


def test_tensor_parallel_image_steps_match_the_single_rank_model(
    ranks: list[dict], single_rank: dict[str, list]
) -> None:
    """Each rank encodes the images whole and writes them after the embedding's reduction;
    a rank that reduced them as well would scale them by the group size."""

    primary = ranks[0]["image_logits"]
    for rank, result in enumerate(ranks):
        assert result["image_logits"] == primary, f"rank {rank} image logits differ from rank 0"
    expected_steps = single_rank["image_logits"]
    assert len(primary) == len(expected_steps) == 4
    for step, (values, reference) in enumerate(zip(primary, expected_steps, strict=True)):
        logits, expected = torch.tensor(values), torch.tensor(reference)
        assert torch.isfinite(logits).all()
        torch.testing.assert_close(logits, expected, **TP_TOLERANCE, msg=f"image step {step}")
        assert torch.equal(logits.argmax(-1), expected.argmax(-1)), f"image step {step} greedy"


def test_tensor_parallel_joint_prefix_hit_matches_its_source_and_a_cold_prefill(
    ranks: list[dict], single_rank: dict[str, list]
) -> None:
    """Each rank restores the snapshot its own heads exported: the hit is the first
    request's next step bit for bit, and a cold prefill's within the chunking tolerance."""

    for rank, result in enumerate(ranks):
        assert result["prefix_logits"] == ranks[0]["prefix_logits"], f"rank {rank} differs"
        hit, first, cold = (torch.tensor(values) for values in result["prefix_logits"])
        assert torch.isfinite(hit).all()
        assert torch.equal(hit, first), f"rank {rank} hit differs from its source"
        torch.testing.assert_close(hit, cold, rtol=1e-4, atol=5e-5)
    hit, expected = ranks[0]["prefix_logits"][0], single_rank["prefix_logits"][0]
    torch.testing.assert_close(torch.tensor(hit), torch.tensor(expected), **TP_TOLERANCE)


def test_each_rank_holds_its_heads_recurrent_state(
    ranks: list[dict], single_rank: dict[str, list]
) -> None:
    num_layers = len(single_rank["states"])
    # Two layers in each of two pools.
    assert num_layers == 4
    for layer in range(num_layers):
        expected_state = torch.tensor(single_rank["states"][layer])
        expected_history = torch.tensor(single_rank["histories"][layer])
        states = [torch.tensor(result["states"][layer]) for result in ranks]
        histories = [torch.tensor(result["histories"][layer]) for result in ranks]
        # [slots, value heads, value_head_dim, key_head_dim]: each rank owns half the heads.
        assert all(state.shape[1] == expected_state.shape[1] // _SIZE for state in states)
        assert torch.isfinite(expected_state).all() and torch.isfinite(expected_history).all()
        torch.testing.assert_close(
            torch.cat(states, dim=1), expected_state, **TP_TOLERANCE, msg=f"layer {layer} state"
        )
        torch.testing.assert_close(
            _assemble_histories(histories),
            expected_history,
            **TP_TOLERANCE,
            msg=f"layer {layer} history",
        )
