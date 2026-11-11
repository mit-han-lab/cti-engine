"""Qwen3.5-MoE over real processes and gloo groups on CPU: tensor-parallel mixers and
expert-parallel sparse blocks.

A tiny image-text checkpoint with both mixers twice, eight experts (two per token) and a
shared expert runs a chunked prefill, a mixed batch and decode steps over the runner's
inference state, with the CPU reference operations at every kernel seam. A second scenario
prefills an image prompt beside a text prompt, its images encoded whole by the vision tower
every rank holds, then decodes it at its rotary offset. TP2 and TP4, with whole experts split
over the ranks (expert parallel) or every expert's width split (expert tensor parallel), must
give the single-rank model's logits at every step, bitwise equal on every rank.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from transformers import AutoConfig, AutoModelForImageTextToText

from cti.distributed import Topology
from cti.distributed.runtime import ParallelGroup
from cti.forward import ImageBatch
from cti.layers import ExpertParallel, ExpertPlacement, ExpertTensorParallel
from cti.multimodal import ImageSpan
from cti.multimodal.qwen3_5 import Qwen3_5RopeLayout
from tests.support.forward_rows import Entry, decode_entry, extend_entry
from tests.support.gloo_ranks import gloo_ranks
from tests.support.moe import ReferenceMoe
from tests.support.multimodal import QWEN3_5_IMAGE_PREPROCESSING
from tests.support.qwen3_5 import load_hf_config, randomize_hf_weights
from tests.support.qwen3_5_engine import (
    STEPS,
    EagerEngine,
    bind_reference_kernels,
)
from tests.support.qwen3_5_moe import TINY_QWEN3_5_MOE_CONFIG
from tests.support.tp_shards import load_sharded

# Gloo ranks are processes of their own: one such test at a time keeps them from starving.
pytestmark = pytest.mark.xdist_group("gloo")

if TYPE_CHECKING:
    from cti.models.base import BaseLLMModel

# FP32 sums reordered by the split (routed experts, the shared expert's channels) and by each
# host's BLAS: measured worst 1.8e-5 (logits up to 4) on another CPU host; without the exit
# reduction the logits miss by >= 2.4 at every step.
_TOLERANCE = {"rtol": 1e-4, "atol": 1e-4}
# At TP4 each rank owns one key head and the two value heads that read it, one query head with
# its gate and a KV head shared with one other rank, and a quarter of the experts (EP) or of
# every expert's and the shared expert's channels (ETP); the vocabulary leaves padded rows at
# both sizes.
_CONFIG = copy.deepcopy(dict(TINY_QWEN3_5_MOE_CONFIG))
_CONFIG["text_config"].update(
    vocab_size=67,
    num_attention_heads=4,
    num_key_value_heads=2,
    linear_num_key_heads=4,
    linear_num_value_heads=8,
)
# The image prompt: text runs around three images whose patches the tower merges 2 x 2, so
# each image's tokens are a quarter of its patches.
_MERGE = _CONFIG["vision_config"]["spatial_merge_size"]
_IMAGE_TEXT_RUNS = (3, 2, 2, 3)
_IMAGE_GRIDS = ((1, 2, 4), (1, 4, 2), (1, 4, 4))
_IMAGE_TOKENS = tuple(t * h * w // _MERGE**2 for t, h, w in _IMAGE_GRIDS)
_IMAGE_PROMPT_LEN = sum(_IMAGE_TEXT_RUNS) + sum(_IMAGE_TOKENS)
_TEXT_PROMPT_LEN = 9


def _image_spans() -> tuple[ImageSpan, ...]:
    spans, offset = [], 0
    for text_len, grid, num_tokens in zip(
        _IMAGE_TEXT_RUNS, _IMAGE_GRIDS, _IMAGE_TOKENS, strict=False
    ):
        offset += text_len
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
        torch.randint(256, (t * h * w, patch_dim), generator=generator, dtype=torch.uint8)
        for t, h, w in _IMAGE_GRIDS
    ]
    layout = Qwen3_5RopeLayout.from_spans(spans, _MERGE)
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


def _run(folder: Path, group: ParallelGroup | None, placement: ExpertPlacement) -> dict[str, list]:
    """Every text step's logits, every image step's logits and the expert share this rank
    holds."""

    def load() -> BaseLLMModel:
        return load_sharded(
            AutoConfig.from_pretrained(folder),
            folder,
            group=group,
            moe_backend=ReferenceMoe(),
            experts=placement,
            image_preprocessing=QWEN3_5_IMAGE_PREPROCESSING,
        )[0]

    model = load()
    generator = torch.Generator().manual_seed(2)
    vocab_size = _CONFIG["text_config"]["vocab_size"]
    token_ids = torch.randint(vocab_size, (2, 24), generator=generator, dtype=torch.int32)
    engine = EagerEngine(model, token_ids)
    image_token_ids, image_steps = _image_steps()
    image_engine = EagerEngine(load(), image_token_ids, image_rows=_IMAGE_PROMPT_LEN)
    with torch.inference_mode():
        logits = [engine.execute(*entries) for entries in STEPS]
        image_logits = [
            image_engine.execute(*entries, rope_positions=rows, images=images)
            for entries, rows, images in image_steps
        ]
    experts = [layer.mlp.experts for layer in model.model.layers.op_list]
    # Tensors would cross the Pipe as shared memory owned by an exiting rank.
    return {
        "logits": [step.tolist() for step in logits],
        "image_logits": [step.tolist() for step in image_logits],
        "experts": [e.placement for e in experts],
    }


def _rank(group: ParallelGroup, argument: tuple[Path, ExpertPlacement]) -> dict[str, list]:
    # The rank process exits after the scenario, so its bindings are never undone.
    bind_reference_kernels(pytest.MonkeyPatch())
    folder, placement = argument
    return _run(folder, group, placement)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("qwen3_5_moe-tp")
    config = load_hf_config(_CONFIG, folder)
    reference = AutoModelForImageTextToText.from_config(config, experts_implementation="eager")
    # The engine declares a BF16 convolution weight.
    randomize_hf_weights(reference, conv_dtype=torch.bfloat16)
    reference.save_pretrained(folder)
    return folder


@pytest.fixture(scope="module")
def single_rank(checkpoint: Path) -> dict[str, list]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        bind_reference_kernels(monkeypatch)
        return _run(checkpoint, None, ExpertParallel())


_PLACEMENTS = {"ep": ExpertParallel(), "etp": ExpertTensorParallel()}


@pytest.fixture(
    scope="module",
    params=[(size, name) for name in _PLACEMENTS for size in (2, 4)],
    ids=lambda param: f"tp{param[0]}-{param[1]}",
)
def ranks(
    request: pytest.FixtureRequest, checkpoint: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[ExpertPlacement, list[dict]]]:
    size, name = request.param
    rendezvous = tmp_path_factory.mktemp("rendezvous")
    placement = _PLACEMENTS[name]
    with gloo_ranks(_rank, size, (checkpoint, placement), rendezvous) as results:
        yield placement, results


@pytest.mark.parametrize("scenario", ["logits", "image_logits"], ids=["text", "image"])
def test_tensor_and_expert_parallel_logits_match_the_single_rank_model(
    ranks: tuple[ExpertPlacement, list[dict]], single_rank: dict[str, list], scenario: str
) -> None:
    """In the image scenario each rank encodes the images whole and writes them after the
    embedding's reduction; a rank that reduced them as well would scale them by the group
    size."""

    _, ranks = ranks
    for rank, result in enumerate(ranks):
        assert result[scenario] == ranks[0][scenario], f"rank {rank} differs from rank 0"
    assert len(single_rank[scenario]) == 4
    for step, (values, reference) in enumerate(
        zip(ranks[0][scenario], single_rank[scenario], strict=True)
    ):
        logits, expected = torch.tensor(values), torch.tensor(reference)
        assert logits.shape == (2, _CONFIG["text_config"]["vocab_size"])
        assert torch.isfinite(logits).all()
        # Without the exit reduction the logits miss by >= 2.4 at every step.
        torch.testing.assert_close(logits, expected, **_TOLERANCE, msg=f"step {step}")
        assert torch.equal(logits.argmax(-1), expected.argmax(-1)), f"step {step} greedy token"


def test_each_rank_holds_its_expert_share(
    ranks: tuple[ExpertPlacement, list[dict]], single_rank: dict[str, list]
) -> None:
    placement, ranks = ranks
    layers = _CONFIG["text_config"]["num_hidden_layers"]
    width = _CONFIG["text_config"]["moe_intermediate_size"]
    assert all(held.is_whole for held in single_rank["experts"])
    for rank, result in enumerate(ranks):
        expected = placement.shard(8, width, Topology(rank, len(ranks)))
        assert result["experts"] == [expected] * layers
