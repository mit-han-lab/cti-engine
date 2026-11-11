"""The Qwen3.5 vision tower and image prefill on CPU, against transformers 5.16.1.

For each model family that encodes images, dense Qwen3.5 and Qwen3.5-MoE, Transformers'
``ForConditionalGeneration`` with random FP32 weights saves a tiny image-text checkpoint:
the same two-block vision tower that merges 2 x 2 patches and resamples a 4 x 4 grid of
learned positions, then the family's tiny text model of Gated DeltaNet and full-attention
layers. The engine loads it through its checkpoint mapping, with the CPU reference
operations bound at every kernel seam (a MoE model routes through the CPU reference
backend), and runs prompts that hold images of several sizes, larger and smaller than the
position grid, over the runner's inference state.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import accumulate, pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import transformers
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PreTrainedModel,
    Qwen2VLImageProcessorPil,
)
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

import ctikernels
from cti.forward import ImageBatch
from cti.layers import BaseOP
from cti.models import create_model, get_model_class
from cti.models.base import BaseLLMModel
from cti.multimodal import ImageInputs, ImageLimits, ImagePreprocessing, ImageSpan
from cti.multimodal.qwen3_5 import Qwen3_5RopeLayout
from cti.multimodal.qwen3_5.processor import Qwen3_5InputProcessor
from cti.protocol import SamplingParams, UserMsg
from tests.support import qwen3_5_moe
from tests.support.checkpoint_facts import checkpoint_facts
from tests.support.forward_rows import decode_entry, extend_entry
from tests.support.inference.prefix_cache import compare_keys
from tests.support.inference.qwen3_5_serving import CpuServing
from tests.support.moe import ReferenceMoe
from tests.support.multimodal import (
    QWEN3_5_IMAGE_PREPROCESSING,
    QWEN3_5_PREPROCESSOR_CONFIG,
    hf_pixel_values,
)
from tests.support.qwen3_5 import (
    TINY_QWEN3_5_VL_CONFIG,
    load_hf_config,
    randomize_hf_weights,
    text_only_config,
)
from tests.support.qwen3_5_engine import EagerEngine, bind_reference_kernels
from tests.support.tp_shards import default_dtype, load_sharded, site

_VISION = TINY_QWEN3_5_VL_CONFIG["vision_config"]
_MERGE = _VISION["spatial_merge_size"]
_FRAMES = _VISION["temporal_patch_size"]
_PATCH_DIM = 3 * _VISION["patch_size"] ** 2
_IMAGE_TOKEN_ID = TINY_QWEN3_5_VL_CONFIG["image_token_id"]
# Patch grids (frames, rows, columns): wider and taller than the 4 x 4 position grid,
# and a single merge window.
_GRIDS = ((1, 4, 6), (1, 2, 2), (1, 6, 4))
# Text tokens before, between and after the images.
_TEXT_RUNS = (5, 3, 2, 4)
_DECODE_STEPS = 3
# The model packages that each hold the vision tower, as models are peers.
_VISION_FAMILIES = ("qwen3_5", "qwen3_5_moe")
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _reference_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    bind_reference_kernels(monkeypatch)


@dataclass(frozen=True)
class _Family:
    """A model family that encodes images: its tiny image-text config, how a text-only export
    of it saves, and the options that Transformers and the engine build it with."""

    config: Mapping[str, Any]
    text_only_config: Callable[[Mapping[str, Any]], dict[str, Any]]
    reference_options: Mapping[str, str]
    engine_options: Mapping[str, object]

    def load(self, folder: Path) -> BaseLLMModel:
        """Build and load the engine model as the runner does, in FP32."""

        return load_sharded(
            AutoConfig.from_pretrained(folder),
            folder,
            image_preprocessing=QWEN3_5_IMAGE_PREPROCESSING,
            **self.engine_options,
        )[0]


# Both share the tiny vision tower and placeholders, so every prompt below serves both. The
# MoE reference runs its own per-expert loop.
_FAMILIES = {
    "dense": _Family(TINY_QWEN3_5_VL_CONFIG, text_only_config, {}, {}),
    "moe": _Family(
        qwen3_5_moe.TINY_QWEN3_5_MOE_CONFIG,
        qwen3_5_moe.text_only_config,
        {"experts_implementation": "eager"},
        {"moe_backend": ReferenceMoe()},
    ),
}


@pytest.fixture(scope="module", params=sorted(_FAMILIES))
def family(request: pytest.FixtureRequest) -> _Family:
    return _FAMILIES[request.param]


@dataclass(frozen=True)
class _Checkpoint:
    folder: Path
    reference: PreTrainedModel
    family: _Family

    def load(self) -> BaseLLMModel:
        return self.family.load(self.folder)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory, family: _Family) -> _Checkpoint:
    folder = tmp_path_factory.mktemp("qwen3_5_vl")
    config = load_hf_config(family.config, folder)
    reference = AutoModelForImageTextToText.from_config(
        config, attn_implementation="eager", **family.reference_options
    )
    randomize_hf_weights(reference.eval(), conv_dtype=torch.bfloat16)
    reference.save_pretrained(folder)
    return _Checkpoint(folder, reference, family)


@dataclass(frozen=True)
class _Prompt:
    """A prompt of text runs around image placeholders, with its continuation."""

    token_ids: torch.Tensor  # int32 [prompt + continuation]
    spans: tuple[ImageSpan, ...]
    pixels: tuple[torch.Tensor, ...]  # uint8 patch rows of each image
    layout: Qwen3_5RopeLayout
    prompt_len: int

    @classmethod
    def build(cls, grids: tuple[tuple[int, int, int], ...] = _GRIDS) -> _Prompt:
        generator = torch.Generator().manual_seed(4)
        pieces, spans, offset = [], [], 0
        for text_len, grid in zip(_TEXT_RUNS, (*grids, None), strict=True):
            # Text ids stay below the vision ids, as a tokenizer's would.
            pieces.append(torch.randint(56, (text_len,), generator=generator, dtype=torch.int32))
            offset += text_len
            if grid is None:
                break
            num_tokens = grid[0] * grid[1] * grid[2] // _MERGE**2
            pieces.append(torch.full((num_tokens,), _IMAGE_TOKEN_ID, dtype=torch.int32))
            digest = hashlib.sha256(bytes(grid)).digest()
            spans.append(ImageSpan(offset, num_tokens, grid, digest))
            offset += num_tokens
        continuation = torch.randint(56, (_DECODE_STEPS,), generator=generator, dtype=torch.int32)
        pixels = tuple(
            torch.randint(256, (t * h * w, _PATCH_DIM), generator=generator, dtype=torch.uint8)
            for t, h, w in grids
        )
        return cls(
            torch.cat([*pieces, continuation]),
            tuple(spans),
            pixels,
            Qwen3_5RopeLayout.from_spans(spans, _MERGE),
            offset,
        )

    def images(self, start: int, end: int) -> ImageBatch | None:
        """The whole images among tokens ``[start, end)``, as a step holding them packs them."""

        inside = [
            index
            for index, span in enumerate(self.spans)
            if start <= span.offset and span.end <= end
        ]
        if not inside:
            return None
        rows = [
            torch.arange(self.spans[index].offset, self.spans[index].end) - start
            for index in inside
        ]
        # The encoder stores the images in rows [0, tokens), and the tokens load them.
        tokens = range(sum(len(image_rows) for image_rows in rows))
        return ImageBatch.unpack(
            ImageBatch.pack(
                [(self.pixels[index], self.spans[index].grid_thw) for index in inside],
                tokens,
                tokens,
                torch.cat(rows),
            )
        )

    @property
    def prompt_ids(self) -> torch.Tensor:
        return self.token_ids[: self.prompt_len]

    def image_inputs(self) -> ImageInputs:
        return ImageInputs(torch.cat(self.pixels), self.spans)

    def reference_pixels(self) -> torch.Tensor:
        """Transformers' pixel values of the images, at the BF16 the engine normalizes to."""

        return hf_pixel_values(torch.cat(self.pixels), _FRAMES).bfloat16()

    def reference_outputs(
        self,
        reference: PreTrainedModel,
        continuation: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transformers' final hidden states and logits over the prompt and a continuation,
        its own unless given."""

        if continuation is not None:
            sequence = torch.cat([self.prompt_ids, torch.tensor(continuation, dtype=torch.int32)])
        else:
            sequence = self.token_ids
        token_ids = sequence[None].long()
        with torch.no_grad():
            output = reference(
                input_ids=token_ids,
                pixel_values=self.reference_pixels(),
                image_grid_thw=torch.tensor([span.grid_thw for span in self.spans]),
                mm_token_type_ids=(token_ids == _IMAGE_TOKEN_ID).int(),
                output_hidden_states=True,
            )
        return output.hidden_states[-1][0], output.logits[0]


def _assert_matches(engine: torch.Tensor, reference: torch.Tensor) -> None:
    # Both run in FP32 end to end, the text model with test_qwen3_5_parity's bound; the
    # tower's position embedding sums its four taps in another order than Transformers.
    torch.testing.assert_close(engine, reference, rtol=1e-4, atol=5e-5)


def _engine(model: BaseLLMModel, prompt: _Prompt) -> EagerEngine:
    return EagerEngine(
        model, prompt.token_ids[None], image_rows=sum(span.num_tokens for span in prompt.spans)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vision_position_addition_matches_batched_addition_bitwise(
    checkpoint: _Checkpoint, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    from cti.models.qwen3_5.vision import _merge_window_coordinates

    tower = checkpoint.load().visual
    prompt = _Prompt.build()
    images = prompt.images(0, prompt.prompt_len)
    patches = tower.patch_embed.forward(images.pixel_values).to(dtype)
    positions = []
    for _, height, width in images.grids:
        rows, columns = _merge_window_coordinates(height, width, _MERGE, CPU)
        positions.append(tower.pos_embed.forward(rows, columns, height, width))
    expected = patches + torch.cat(positions).to(dtype)
    monkeypatch.setattr(tower.patch_embed, "forward", lambda _: patches.clone())
    monkeypatch.setattr(tower.blocks, "op_list", [])
    monkeypatch.setattr(tower.merger, "forward", lambda x: x)

    actual = tower.forward(images)

    assert actual.dtype == dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_vision_tower_matches_transformers(checkpoint: _Checkpoint) -> None:
    model = checkpoint.load()
    prompt = _Prompt.build()
    images = prompt.images(0, prompt.prompt_len)
    assert images is not None

    with torch.no_grad():
        encoded = model.visual.forward(images)
        expected = checkpoint.reference.model.visual(
            prompt.reference_pixels(), grid_thw=torch.tensor(_GRIDS)
        ).pooler_output

    assert encoded.shape == (sum(span.num_tokens for span in prompt.spans), 32)
    _assert_matches(encoded, expected)


def test_the_warm_up_encodes_the_largest_step_of_images(
    checkpoint: _Checkpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = checkpoint.load()
    encoded: list[ImageBatch] = []
    forward = model.visual.forward

    def record(images: ImageBatch) -> torch.Tensor:
        encoded.append(images)
        return forward(images)

    monkeypatch.setattr(model.visual, "forward", record)

    assert model.warm_up_image_encoder(max_tokens=10, max_image_tokens=6, device=CPU)

    # One image of 6 tokens and one of the other 4, each a row of 2 x 2 merge windows,
    # fill the step's 10 token embeddings.
    (images,) = encoded
    assert images.grids == ((1, 2, 12), (1, 2, 8))
    assert images.cu_patches.tolist() == [0, 24, 40]
    assert images.max_patches == 24
    assert images.dst.tolist() == [*range(10)]


def _patch_embed(
    family: str,
    preprocessing: ImagePreprocessing | None,
    *,
    hidden_size: int = 1536,
    dtype: torch.dtype = torch.float32,
) -> BaseOP:
    """The family's embedding of 16-pixel patches of 3 channels and 2 frames, in ``dtype``."""

    package = f"cti.models.{family}"
    config = importlib.import_module(f"{package}.config").Qwen3_5VisionConfig(
        depth=1,
        hidden_size=hidden_size,
        num_heads=2,
        intermediate_size=8,
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=4,
        out_hidden_size=8,
    )
    with default_dtype(dtype):
        return importlib.import_module(f"{package}.vision").Qwen3_5VisionPatchEmbed(
            config, preprocessing, site=site()
        )


def _identity_patch_embed(family: str, preprocessing: ImagePreprocessing | None) -> BaseOP:
    """An FP32 patch embedding with an identity projection: its output is the patches'
    normalized pixels, each frame of transformers' patch layout."""

    embed = _patch_embed(family, preprocessing)
    embed.load_state_dict({"proj.weight": torch.eye(1536), "proj.bias": torch.zeros(1536)})
    return embed


def _normalization(mean: Sequence[float], std: Sequence[float]) -> ImagePreprocessing:
    return replace(QWEN3_5_IMAGE_PREPROCESSING, image_mean=tuple(mean), image_std=tuple(std))


@pytest.mark.parametrize("family", _VISION_FAMILIES)
@pytest.mark.parametrize(
    ("mean", "std"), ((OPENAI_CLIP_MEAN, OPENAI_CLIP_STD), ((0.5,) * 3, (0.5,) * 3))
)
def test_every_level_normalizes_bitwise_as_transformers(
    family: str, mean: Sequence[float], std: Sequence[float]
) -> None:
    image_processor = Qwen2VLImageProcessorPil()
    # Each channel's 256 levels as one CHW image, through the processor's own steps.
    ramp = np.broadcast_to(np.arange(256, dtype=np.uint8), (3, 1, 256))
    expected = image_processor.normalize(image_processor.rescale(ramp, 1 / 255), mean, std)
    rows = torch.arange(256, dtype=torch.uint8)[:, None].expand(-1, 768).contiguous()

    normalized = _identity_patch_embed(family, _normalization(mean, std)).forward(rows)

    # Row v holds level v in all 256 pixels of each channel and frame, as BF16 values.
    levels = torch.from_numpy(expected[:, 0]).bfloat16().float().T
    assert torch.equal(
        normalized.view(256, 3, 2, 256), levels[:, :, None, None].expand(-1, -1, 2, 256)
    )


@pytest.mark.parametrize("family", _VISION_FAMILIES)
def test_a_tower_without_image_preprocessing_takes_no_image(family: str) -> None:
    embed = _identity_patch_embed(family, None)

    assert embed._levels is None
    with pytest.raises(ValueError, match="no image preprocessing"):
        embed.forward(torch.zeros(1, 768, dtype=torch.uint8))
    with pytest.raises(ValueError, match="normalizes 1 channels, the vision tower takes 3"):
        _identity_patch_embed(family, _normalization((0.5,), (0.5,)))


@pytest.mark.parametrize("name", sorted(_FAMILIES))
def test_the_model_builds_its_table_from_the_image_preprocessing_it_is_given(
    name: str, tmp_path: Path
) -> None:
    family = _FAMILIES[name]
    hf_config = load_hf_config(family.config, tmp_path)
    before = hf_config.to_dict()
    model_class = get_model_class(hf_config.architectures[0])

    with_preprocessing = create_model(
        model_class,
        checkpoint_facts(hf_config, image_preprocessing=QWEN3_5_IMAGE_PREPROCESSING),
        site=site(**family.engine_options),
    )
    without = create_model(
        model_class, checkpoint_facts(hf_config), site=site(**family.engine_options)
    )

    assert with_preprocessing.visual.patch_embed._levels.shape == (3 * 256,)
    assert without.visual.patch_embed._levels is None
    assert hf_config.to_dict() == before


def _processed_image(size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """A random image's uint8 patch rows from the engine's processor and its FP32
    ``pixel_values`` from transformers'."""

    image_processor = Qwen2VLImageProcessorPil(**QWEN3_5_PREPROCESSOR_CONFIG)
    hf_config = transformers.Qwen3_5Config()
    processor = Qwen3_5InputProcessor(hf_config, image_processor, limits=ImageLimits(1, 16384))
    generator = torch.Generator().manual_seed(sum(size))
    pixels = torch.randint(256, (size[1], size[0], 3), generator=generator, dtype=torch.uint8)
    image = Image.fromarray(pixels.numpy())
    expected = image_processor(images=[image], return_tensors="pt")["pixel_values"]
    rows = processor.expand([hf_config.image_token_id, 0], [image]).images.pixel_values
    return rows, expected


@pytest.mark.parametrize("size", ((100, 70), (64, 300), (517, 333)))
def test_processor_rows_normalize_to_transformers_pixel_values_bitwise(
    size: tuple[int, int],
) -> None:
    rows, expected = _processed_image(size)

    normalized = _identity_patch_embed("qwen3_5", QWEN3_5_IMAGE_PREPROCESSING).forward(rows)

    # The engine's one frame per patch, repeated in transformers' channel, frame order.
    assert torch.equal(normalized, expected.bfloat16().float())


@pytest.mark.parametrize("family", _VISION_FAMILIES)
@pytest.mark.parametrize("size", ((100, 70), (517, 333)))
def test_the_patch_embedding_is_bitwise_transformers_bf16_pixels_through_the_checkpoint_weight(
    family: str, size: tuple[int, int]
) -> None:
    rows, expected = _processed_image(size)
    flatten = importlib.import_module(f"cti.models.{family}.model")._flatten_patch_conv
    generator = torch.Generator().manual_seed(7)
    conv = (torch.randn(64, 3, 2, 16, 16, generator=generator) / 16).bfloat16()
    bias = torch.randn(64, generator=generator).bfloat16()
    embed = _patch_embed(family, QWEN3_5_IMAGE_PREPROCESSING, hidden_size=64, dtype=torch.bfloat16)
    embed.load_state_dict({"proj.weight": flatten(conv)[0], "proj.bias": bias})

    embedded = embed.forward(rows)

    # The checkpoint's Conv3d weight, unfolded, on transformers' pixel values in BF16.
    reference = embed.proj.forward(expected.bfloat16())
    assert embedded.dtype == torch.bfloat16
    assert torch.equal(embedded, reference)


def test_image_bounds_follow_the_grids() -> None:
    prompt = _Prompt.build()
    images = prompt.images(0, prompt.prompt_len)

    patches = [t * h * w for t, h, w in _GRIDS]
    assert images.cu_patches.tolist() == [0, *accumulate(patches)]
    assert images.max_patches == max(patches)
    assert images.dst.tolist() == [
        position for span in prompt.spans for position in range(span.offset, span.end)
    ]


def test_image_prefill_matches_transformers(checkpoint: _Checkpoint) -> None:
    model = checkpoint.load()
    prompt = _Prompt.build()
    engine = _engine(model, prompt)
    reference_hidden, reference_logits = prompt.reference_outputs(checkpoint.reference)
    end = prompt.prompt_len

    batch = engine.prepare(
        extend_entry(0, 0, end),
        rope_positions=prompt.layout.positions(0, end),
        images=prompt.images(0, end),
    )
    hidden = model.forward_hidden(batch)

    _assert_matches(hidden, reference_hidden[:end])
    _assert_matches(model.compute_logits(hidden, batch), reference_logits[end - 1 : end])


def test_chunks_around_images_then_decode_match_transformers(checkpoint: _Checkpoint) -> None:
    model = checkpoint.load()
    prompt = _Prompt.build()
    engine = _engine(model, prompt)
    _, reference_logits = prompt.reference_outputs(checkpoint.reference)
    first, second, third = prompt.spans
    # Text alone, then the first image with the text after it, then two images in one
    # chunk that ends one token before the prompt does, then that token.
    bounds = (0, first.offset, second.offset, third.end, prompt.prompt_len - 1, prompt.prompt_len)

    logits = [
        engine.execute(
            extend_entry(0, start, end),
            rope_positions=prompt.layout.positions(start, end),
            images=prompt.images(start, end),
        )
        for start, end in pairwise(bounds)
    ]
    decode_offset = prompt.layout.decode_offset
    for position in range(prompt.prompt_len, prompt.prompt_len + _DECODE_STEPS - 1):
        logits.append(
            engine.execute(
                decode_entry(0, position),
                rope_positions=torch.full((3, 1), position + decode_offset),
            )
        )

    positions = [end - 1 for end in bounds[1:]]
    positions += range(prompt.prompt_len, prompt.prompt_len + _DECODE_STEPS - 1)
    _assert_matches(torch.cat(logits), reference_logits[positions])


def test_a_text_only_export_builds_no_vision_tower_and_refuses_images(
    family: _Family, tmp_path: Path
) -> None:
    config = load_hf_config(family.text_only_config(family.config), tmp_path)
    reference = AutoModelForCausalLM.from_config(config)
    randomize_hf_weights(reference, conv_dtype=torch.bfloat16)
    reference.save_pretrained(tmp_path)
    model = family.load(tmp_path)
    prompt = _Prompt.build()
    engine = _engine(model, prompt)
    end = prompt.prompt_len

    assert model.visual is None
    assert not any(name.startswith("visual.") for name in model.state_dict())
    assert not model.warm_up_image_encoder(max_tokens=10, max_image_tokens=6, device=CPU)
    with pytest.raises(ValueError, match="no vision tower"):
        engine.execute(
            extend_entry(0, 0, end),
            rope_positions=prompt.layout.positions(0, end),
            images=prompt.images(0, end),
        )


def test_scheduled_image_and_text_requests_match_transformers(
    checkpoint: _Checkpoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Through the scheduler and the runner's preparation: eight-token steps split the
    image prompt inside its images, and its decodes share steps with a text prompt's
    chunks."""

    model = checkpoint.load()
    prompt = _Prompt.build()
    text_ids = torch.randint(56, (10,), generator=torch.Generator().manual_seed(5)).int()
    serving = CpuServing(model, monkeypatch, max_running_req=2, max_batch_tokens=8, max_seq_len=64)
    first, second, third = prompt.spans

    rejections = serving.submit(
        UserMsg(
            uid=0,
            input_ids=prompt.prompt_ids,
            sampling=SamplingParams(temperature=0, max_new_tokens=4, ignore_eos=True).snapshot(),
            images=prompt.image_inputs(),
            rope_layout=prompt.layout,
        ),
        UserMsg(
            uid=1,
            input_ids=text_ids,
            sampling=SamplingParams(temperature=0, max_new_tokens=3, ignore_eos=True).snapshot(),
        ),
    )
    sampled = serving.run()

    assert rejections == ()
    assert {uid: len(tokens) for uid, tokens in sampled.items()} == {0: 4, 1: 3}
    image_chunks = [
        (row.kv_prefix_len, row.sequence_len)
        for uid, row, _ in serving.scheduled(decode=False)
        if uid == 0
    ]
    # The first chunk ends inside the first image, which it encodes whole; the second
    # loads the image's rest beside the second image's encode.
    assert image_chunks == [(0, 8), (8, 16), (16, 24), (24, prompt.prompt_len)]
    assert first.offset < 8 < first.end
    assert [
        ImageBatch.unpack(batch.images).grids if batch.images else ()
        for batch in serving.batches[:4]
    ] == [
        (first.grid_thw,),
        (second.grid_thw,),
        (third.grid_thw,),
        (),
    ]
    decode_offsets = {(uid, spec.rope_offset) for uid, _, spec in serving.scheduled(decode=True)}
    assert decode_offsets == {(0, prompt.layout.decode_offset), (1, 0)}
    assert prompt.layout.decode_offset != 0

    references = {
        0: prompt.reference_outputs(checkpoint.reference, sampled[0])[1],
        1: checkpoint.reference(input_ids=torch.cat([text_ids, torch.tensor(sampled[1])])[None])
        .logits[0]
        .detach(),
    }
    for step, uids in zip(serving.steps, serving.request_ids, strict=True):
        expected = torch.stack(
            [
                references[uid][entry.sequence_len - 1]
                for entry, uid in zip(step.entries, uids, strict=True)
            ]
        )
        _assert_matches(step.logits, expected)


@pytest.mark.parametrize("cached", [True, False], ids=["cached", "evicted"])
def test_a_hit_inside_an_image_writes_its_rest_and_matches_transformers(
    checkpoint: _Checkpoint,
    monkeypatch: pytest.MonkeyPatch,
    cached: bool,
) -> None:
    """The radix's recurrent snapshot at 64 lies inside the third image, [17, 81): the
    repeated prompt restores there and loads only that image's last 17 rows, cached or
    encoded again. The model only ever reads vocabulary ids."""

    monkeypatch.setattr(ctikernels, "fast_compare_key", compare_keys)
    model = checkpoint.load()
    prompt = _Prompt.build(grids=(*_GRIDS[:2], (1, 16, 16)))
    serving = CpuServing(
        model, monkeypatch, max_running_req=2, max_batch_tokens=128, max_seq_len=96, snapshot_rows=2
    )
    message = UserMsg(
        uid=0,
        input_ids=prompt.prompt_ids,
        sampling=SamplingParams(temperature=0, max_new_tokens=2, ignore_eos=True).snapshot(),
        images=prompt.image_inputs(),
        rope_layout=prompt.layout,
    )

    sampled = []
    for uid in (0, 1):
        assert serving.submit(replace(message, uid=uid)) == ()
        sampled.append(serving.run()[uid])
        if not cached:
            serving.scheduler._encoder_cache.reset()

    extends = [(uid, row.kv_prefix_len) for uid, row, _ in serving.scheduled(decode=False)]
    assert extends == [(0, 0), (1, 64)]
    stored, images = (ImageBatch.unpack(serving.batches[i].images) for i in (0, 2))
    third = stored.store_rows[-64:] if cached else images.store_rows
    assert images.src_rows.tolist() == third[64 - 17 :].tolist()
    assert (images.grids, images.cached) == (
        ((), 17) if cached else ((prompt.spans[2].grid_thw,), 0)
    )
    assert bool((serving.runner.token_id_table >= 0).all())
    assert sampled[0] == sampled[1]
    reference = prompt.reference_outputs(checkpoint.reference, sampled[0])[1]
    for step in serving.steps:
        (entry,) = step.entries
        _assert_matches(step.logits, reference[entry.sequence_len - 1][None])
    serving.scheduler.require_quiescent()
