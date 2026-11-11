from __future__ import annotations

import base64
import io
import json
import logging
import struct
import zlib
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessorPil,
    Qwen3_5Config,
    Qwen3VLProcessor,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from cti.multimodal import ImageLimits
from cti.multimodal.inputs import ImageInputs, ImageSpan
from cti.multimodal.processor import (
    ImagePreprocessing,
    ProcessedPrompt,
    create_input_processor,
    read_image_preprocessing,
)
from cti.multimodal.qwen3_5.processor import Qwen3_5InputProcessor
from cti.protocol import ChatPrompt, SamplingParams, TokenizeMsg
from cti.tokenizer import PromptEncoder
from cti.tokenizer.server import tokenize
from cti.tokenizer.text import TokenizerTextDecoder
from tests.support.multimodal import (
    QWEN3_5_IMAGE_PREPROCESSING,
    QWEN3_5_PREPROCESSOR_CONFIG,
    hf_pixel_values,
    hf_qwen3_5_rope_index,
)

_SPECIAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
)
_WORDS = ("<unk>", "user", "assistant", "Describe", "this", "and", "that", "<think>", "</think>")
_VOCABULARY = {token: token_id for token_id, token in enumerate((*_WORDS, *_SPECIAL_TOKENS))}
_IMAGE_TOKEN_ID = _VOCABULARY["<|image_pad|>"]
_VIDEO_TOKEN_ID = _VOCABULARY["<|video_pad|>"]
# The Qwen3.5 template's image test, vision markers and thinking switch, in miniature.
_CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|> {{ message['role'] }} "
    "{% if message['content'] is string %}{{ message['content'] }} {% else %}"
    "{% for item in message['content'] %}"
    "{% if 'image' in item or 'image_url' in item or item['type'] == 'image' %}"
    "<|vision_start|><|image_pad|><|vision_end|> "
    "{% elif item['type'] == 'video' %}<|vision_start|><|video_pad|><|vision_end|> "
    "{% else %}{{ item['text'] }} {% endif %}{% endfor %}{% endif %}<|im_end|> {% endfor %}"
    "{% if add_generation_prompt %}<|im_start|> assistant "
    "{% if enable_thinking is defined and enable_thinking is false %}<think> </think> "
    "{% else %}<think> {% endif %}{% endif %}"
)
_PATCH_DIM = 3 * 16 * 16


class _ImageOnlyQwen3VLProcessor(Qwen3VLProcessor):
    """Transformers' processor as the oracle in an environment without torchvision.

    Without torchvision, transformers 5.16.1 swaps BaseVideoProcessor for a dummy class,
    so a real video processor fails the constructor's class check. Images never use it.
    """

    def check_argument_for_proper_class(self, argument_name: str, argument: object) -> object:
        if argument_name == "video_processor":
            return type(argument)
        return super().check_argument_for_proper_class(argument_name, argument)


def _tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(WordLevel(_VOCABULARY, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        additional_special_tokens=list(_SPECIAL_TOKENS),
    )
    tokenizer.chat_template = _CHAT_TEMPLATE
    return tokenizer


def _image_processor(**overrides: object) -> Qwen2VLImageProcessorPil:
    return Qwen2VLImageProcessorPil(**{**QWEN3_5_PREPROCESSOR_CONFIG, **overrides})


def _config() -> Qwen3_5Config:
    return Qwen3_5Config(image_token_id=_IMAGE_TOKEN_ID, video_token_id=_VIDEO_TOKEN_ID)


def _assert_hf_pixels(actual: ProcessedPrompt, expected: torch.Tensor) -> None:
    """The uint8 rows are transformers' FP32 pixels before its normalization and frames."""

    assert actual.images.pixel_values.dtype == torch.uint8
    assert torch.equal(hf_pixel_values(actual.images.pixel_values, 2), expected)


def _processor(*, max_images: int = 4, max_image_tokens: int = 16384) -> Qwen3_5InputProcessor:
    limits = ImageLimits(max_images, max_image_tokens)
    return Qwen3_5InputProcessor(_config(), _image_processor(), limits=limits)


def _encoder(tokenizer: PreTrainedTokenizerFast | None = None, **limits: int) -> PromptEncoder:
    decoder = TokenizerTextDecoder(tokenizer or _tokenizer())
    return PromptEncoder(decoder, _processor(**limits), len(_VOCABULARY))


def _hf_processor() -> Qwen3VLProcessor:
    return _ImageOnlyQwen3VLProcessor(
        image_processor=_image_processor(),
        tokenizer=_tokenizer(),
        video_processor=Qwen3VLVideoProcessor(),
        chat_template=_CHAT_TEMPLATE,
    )


def _image(width: int, height: int, seed: int, channels: int = 3) -> Image.Image:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randint(0, 256, (height, width, channels), generator=generator)
    return Image.fromarray(values.to(torch.uint8).numpy())


def _bytes_url(data: bytes, subtype: str = "png") -> str:
    return f"data:image/{subtype};base64,{base64.b64encode(data).decode()}"


def _data_url(image: Image.Image, image_format: str = "PNG", **save_options: object) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **save_options)
    return _bytes_url(buffer.getvalue(), image_format.lower())


def _rotated_jpeg_url() -> str:
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 degrees clockwise for display
    return _data_url(_image(160, 96, seed=2), "JPEG", exif=exif)


def _png_url_with_split_pixel_data() -> str:
    # An unknown chunk between two IDAT chunks: Pillow raises SyntaxError when it decodes.
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    pixels = zlib.compress(b"".join(b"\0" + b"\x10\x20\x30" * 64 for _ in range(64)))
    png = b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", pixels[:20]),
            chunk(b"\0bad", b""),
            chunk(b"IDAT", pixels[20:]),
            chunk(b"IEND", b""),
        )
    )
    return _bytes_url(png)


def _jpeg_url_with_mistyped_exif() -> str:
    # The ASCII DateTime entry retagged as integer TileLength (0x0132 -> 0x0143): rewriting
    # the EXIF block while undoing the rotation raises struct.error.
    exif = Image.Exif()
    exif[0x0112] = 6
    exif[0x0132] = "2026:10:01 00:00:00"
    buffer = io.BytesIO()
    _image(32, 16, seed=12).save(buffer, "JPEG", exif=exif)
    jpeg = buffer.getvalue()
    assert jpeg.count(b"\x01\x32\x00\x02") == 1
    return _bytes_url(jpeg.replace(b"\x01\x32\x00\x02", b"\x01\x43\x00\x02"), "jpeg")


def _chat(*sources: object, image_key: str = "image") -> list[dict]:
    content: list[dict] = []
    for source in sources:
        if image_key == "image_url":
            content.append({"type": "image_url", "image_url": {"url": source}})
        else:
            content.append({"type": "image", "image": source})
        content.append({"type": "text", "text": "Describe this"})
    return [{"role": "user", "content": content}]


def _assert_same_prompt(actual: ProcessedPrompt, expected: ProcessedPrompt) -> None:
    assert torch.equal(actual.input_ids, expected.input_ids)
    assert torch.equal(actual.images.pixel_values, expected.images.pixel_values)
    assert actual.images.spans == expected.images.spans
    assert actual.rope_layout == expected.rope_layout


def test_chat_matches_transformers_qwen3_vl_processor() -> None:
    # Upscaled PNG, EXIF-rotated JPEG and an in-process RGBA image.
    messages = _chat(
        _data_url(_image(100, 70, seed=1)),
        _rotated_jpeg_url(),
        _image(300, 200, seed=3, channels=4),
    )
    expected = _hf_processor().apply_chat_template(
        messages, tokenize=True, return_dict=True, add_generation_prompt=True, return_tensors="pt"
    )

    actual = _encoder().chat(messages)

    input_ids = expected["input_ids"][0]
    assert actual.input_ids.dtype == torch.int32
    assert torch.equal(actual.input_ids, input_ids.to(torch.int32))
    _assert_hf_pixels(actual, expected["pixel_values"])
    grids = expected["image_grid_thw"].tolist()
    assert [list(span.grid_thw) for span in actual.images.spans] == grids
    assert grids[1][1] > grids[1][2]  # the wide JPEG was rotated upright before preprocessing
    is_image = expected["mm_token_type_ids"][0] == 1
    expanded = torch.zeros_like(is_image)
    for span in actual.images.spans:
        expanded[span.offset : span.end] = True
    assert torch.equal(expanded, is_image)
    positions, delta = hf_qwen3_5_rope_index(
        input_ids, expected["mm_token_type_ids"][0], expected["image_grid_thw"], 2
    )
    assert torch.equal(actual.rope_layout.positions(0, len(input_ids)), positions)
    assert actual.rope_layout.decode_offset == delta


def test_raw_prompt_matches_transformers_qwen3_vl_processor() -> None:
    text = "Describe <|vision_start|><|image_pad|><|vision_end|> this"
    image = _image(64, 300, seed=4)
    expected = _hf_processor()(text=[text], images=[image], return_tensors="pt")

    actual = _encoder().text(text, [_data_url(image)])

    assert torch.equal(actual.input_ids, expected["input_ids"][0].to(torch.int32))
    _assert_hf_pixels(actual, expected["pixel_values"])


def test_openai_image_url_parts_fetched_bytes_and_in_process_images_are_the_same_inputs() -> None:
    image = _image(120, 90, seed=5)
    encoder = _encoder()
    url = _data_url(image)

    expected = encoder.chat(_chat(url))

    _assert_same_prompt(encoder.chat(_chat(url, image_key="image_url")), expected)
    # The bytes the API server fetches for an http(s) or file:// URL.
    fetched = base64.b64decode(url.partition(",")[2])
    _assert_same_prompt(encoder.chat(_chat(fetched, image_key="image_url")), expected)
    _assert_same_prompt(encoder.chat(_chat(image)), expected)


def test_online_chats_carry_their_images_and_rope_layout_like_offline_ones() -> None:
    encoder = _encoder()
    url = _data_url(_image(120, 90, seed=5))
    fetched = base64.b64decode(url.partition(",")[2])
    chat = ChatPrompt(tuple(_chat(url, fetched, image_key="image_url")))

    online, _ = tokenize(
        encoder, TokenizeMsg(3, chat, SamplingParams(max_new_tokens=1024, temperature=0))
    )
    offline = encoder.request(
        3, encoder.chat(_chat(url, url)), SamplingParams(max_new_tokens=1024, temperature=0)
    )

    assert len(online.images.spans) == 2
    _assert_same_prompt(online, offline)


def test_image_digest_covers_size_and_rgb_bytes() -> None:
    encoder = _encoder()
    rgb = bytes(range(18))
    edited = bytearray(rgb)
    edited[0] ^= 1

    def digest(image: Image.Image) -> bytes:
        return encoder.text("<|image_pad|> this", [image]).images.spans[0].digest

    tall = digest(Image.frombytes("RGB", (2, 3), rgb))
    assert tall != digest(Image.frombytes("RGB", (3, 2), rgb))
    assert tall != digest(Image.frombytes("RGB", (2, 3), bytes(edited)))
    assert tall == digest(Image.frombytes("RGB", (2, 3), rgb))


def test_template_kwargs_reach_the_template_and_cannot_set_its_renderer() -> None:
    tokenizer = _tokenizer()
    encoder = _encoder(tokenizer)
    messages = [{"role": "user", "content": "Describe this"}]

    with mock.patch.object(
        tokenizer, "apply_chat_template", wraps=tokenizer.apply_chat_template
    ) as render:
        encoder.chat(messages, template_kwargs={"enable_thinking": False, "custom_flag": 1})
    assert render.call_args.kwargs["custom_flag"] == 1
    assert render.call_args.kwargs["tokenize"] is False

    for reserved in ("tokenize", "chat_template", "add_generation_prompt"):
        with pytest.raises(ValueError, match=f"cannot set {reserved}"):
            encoder.chat(messages, template_kwargs={reserved: True})


def test_thinking_switch_is_passed_only_when_set() -> None:
    tokenizer = _tokenizer()
    encoder = _encoder(tokenizer)
    messages = [{"role": "user", "content": "Describe this"}]

    with mock.patch.object(
        tokenizer, "apply_chat_template", wraps=tokenizer.apply_chat_template
    ) as render:
        default = encoder.chat(messages)
        disabled = encoder.chat(messages, template_kwargs={"enable_thinking": False})

    assert "enable_thinking" not in render.call_args_list[0].kwargs
    assert render.call_args_list[1].kwargs["enable_thinking"] is False
    assert default.images is None and default.rope_layout is None
    assert default.input_ids.tolist() == tokenizer.encode(
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        add_special_tokens=False,
    )
    assert disabled.input_ids[-2:].tolist() == [_VOCABULARY["<think>"], _VOCABULARY["</think>"]]


def test_chat_template_rejections_are_value_errors() -> None:
    tokenizer = _tokenizer()
    tokenizer.chat_template = "{{ raise_exception('System messages take text only') }}"

    with pytest.raises(ValueError, match="cannot render chat: System messages take text only"):
        _encoder(tokenizer).chat([{"role": "system", "content": "Describe this"}])


def test_chat_template_type_errors_are_value_errors() -> None:
    tokenizer = _tokenizer()
    # The tool-call arguments loop of the Qwen3.5 template.
    tokenizer.chat_template = (
        "{% for message in messages %}{% for call in message['tool_calls'] %}"
        "{% for name, value in call['function']['arguments']|items %}{{ name }} "
        "{% endfor %}{% endfor %}{% endfor %}"
    )
    # The OpenAI API's JSON-string arguments, which the template cannot iterate.
    tool_call = {"type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}

    with pytest.raises(ValueError, match="cannot render chat: Can only get item pairs"):
        _encoder(tokenizer).chat(
            [{"role": "assistant", "content": None, "tool_calls": [tool_call]}]
        )


@pytest.mark.parametrize(
    ("messages", "message"),
    (
        ([{"role": "user", "content": ["Describe this"]}], "list of content parts"),
        ([{"role": "user", "content": 123}], "list of content parts"),
        ([{"role": "user", "content": {"type": "text", "text": "this"}}], "list of content parts"),
        ([{"role": "user", "content": {}}], "list of content parts"),
        (["Describe this"], "messages must be mappings"),
        ([None], "messages must be mappings"),
        ({"role": "user", "content": "Describe this"}, "list of messages"),
        (None, "list of messages"),
    ),
)
def test_malformed_chats_are_rejected(messages: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _encoder().chat(messages)


def test_large_images_are_scaled_down_to_the_token_budget() -> None:
    image = _image(1000, 700, seed=6)
    expected = _image_processor()(
        images=[image],
        size={"shortest_edge": 65536, "longest_edge": 64 * 32 * 32},
        return_tensors="pt",
    )

    actual = _encoder(max_image_tokens=64).text("<|image_pad|> this", [image])

    (span,) = actual.images.spans
    assert span.num_tokens <= 64
    assert list(span.grid_thw) == expected["image_grid_thw"][0].tolist()
    _assert_hf_pixels(actual, expected["pixel_values"])


def test_image_upscaled_beyond_the_token_budget_is_rejected() -> None:
    # The minimum size rounds a 100x70 image up to 70 tokens.
    encoder = _encoder(max_image_tokens=64)
    with pytest.raises(ValueError, match="at most 64 tokens"):
        encoder.check(encoder.text("<|image_pad|> this", [_image(100, 70, 7)]))


@pytest.mark.parametrize(
    ("source", "message"),
    (
        ("https://example.com/cat.png", "base64 data:image"),
        ("data:text/plain;base64,aGVsbG8=", "base64 data:image"),
        ("data:image/png,raw", "base64 data:image"),
        ("data:image/png;base64,not base64!", "not valid base64"),
        ("data:image/png;base64,aGVsbG8=", "cannot decode image"),
        (_png_url_with_split_pixel_data(), "cannot decode image"),
        (_jpeg_url_with_mistyped_exif(), "cannot decode image"),
        (b"\x89PNG", "cannot decode image"),
        (None, "data: URL, encoded bytes or a PIL image"),
    ),
)
def test_unsupported_image_sources_are_rejected(source: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _encoder().chat(_chat(source))


@pytest.mark.parametrize("image_format", ("PNG", "JPEG", "WEBP", "GIF"))
def test_web_image_formats_are_decoded(image_format: str) -> None:
    prompt = _encoder().chat(_chat(_data_url(_image(64, 48, 13), image_format)))

    assert len(prompt.images.spans) == 1


def test_other_image_formats_are_never_decoded() -> None:
    # Pillow identifies EPS whatever the URL says, and decoding EPS runs Ghostscript.
    eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 8 8\n%%EndComments\nshowpage\n"

    with mock.patch("PIL.EpsImagePlugin.Ghostscript") as ghostscript:
        with pytest.raises(ValueError, match="cannot decode image"):
            _encoder().chat(_chat(_bytes_url(eps)))

    ghostscript.assert_not_called()


def test_decompression_bombs_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    encoder = _encoder()

    with pytest.warns(Image.DecompressionBombWarning):
        with pytest.raises(ValueError, match="decompression-bomb"):
            encoder.text("<|image_pad|> this", [_data_url(_image(12, 10, 8))])
    with pytest.raises(ValueError, match="cannot decode image"):
        encoder.text("<|image_pad|> this", [_data_url(_image(30, 30, 8))])


@pytest.mark.parametrize(
    ("text", "num_images", "message"),
    (
        ("<|image_pad|> <|image_pad|> this", 1, "2 image placeholders for 1 images"),
        ("<|image_pad|> this", 0, "1 image placeholders for 0 images"),
        ("Describe this", 1, "0 image placeholders for 1 images"),
        ("<|video_pad|> this", 0, "video inputs are not supported"),
        ("Describe <|image_pad|>", 1, "followed by a text token"),
        ("<|image_pad|> and <|image_pad|> this", 2, "at most 1 images"),
    ),
)
def test_prompts_with_inconsistent_placeholders_are_rejected(
    text: str, num_images: int, message: str
) -> None:
    images = [_image(64, 64, seed) for seed in range(num_images)]
    encoder = _encoder(max_images=1)

    with pytest.raises(ValueError, match=message):
        encoder.check(encoder.text(text, images))


def test_video_parts_are_rejected() -> None:
    messages = [{"role": "user", "content": [{"type": "video", "video": "clip.mp4"}]}]

    encoder = _encoder()
    with pytest.raises(ValueError, match="video inputs are not supported"):
        encoder.check(encoder.chat(messages))


def test_token_id_prompts_round_trip_through_validation() -> None:
    encoder = _encoder()
    prompt = encoder.chat(_chat(_image(64, 64, 9), _image(96, 64, 10)))

    assert encoder.check(prompt) == prompt.rope_layout
    assert encoder.check(encoder.tokens([1, 2, 3])) is None


def _token_prompt() -> tuple[torch.Tensor, ImageInputs]:
    span = ImageSpan(1, 4, (1, 4, 4), bytes(32))
    input_ids = torch.tensor([3, *[_IMAGE_TOKEN_ID] * 4, 4], dtype=torch.int32)
    return input_ids, ImageInputs(torch.zeros(16, _PATCH_DIM, dtype=torch.uint8), (span,))


def test_the_encoder_pairs_a_prompts_images_with_their_layout() -> None:
    input_ids, images = _token_prompt()
    layout = _processor().validate_token_ids(input_ids, images)

    with pytest.raises(ValueError, match="rope layout does not match"):
        _encoder().check(ProcessedPrompt(input_ids, images, None))
    with pytest.raises(ValueError, match="rope layout does not match"):
        _encoder().check(ProcessedPrompt(torch.tensor([3, 4], dtype=torch.int32), None, layout))


def test_token_id_prompt_layout_comes_from_its_spans() -> None:
    input_ids, images = _token_prompt()

    layout = _processor().validate_token_ids(input_ids, images)

    assert layout.images == ((1, 2, 2),)
    assert layout.decode_offset == -2


@pytest.mark.parametrize(
    ("edit", "message"),
    (
        (lambda ids, images: (ids.view(2, 3), images), "one-dimensional"),
        (lambda ids, images: (ids, None), "placeholders but no images"),
        (
            lambda ids, images: (torch.cat([ids, ids.new_tensor([_VIDEO_TOKEN_ID])]), images),
            "video",
        ),
        (
            lambda ids, images: (torch.cat([ids, ids.new_tensor([_IMAGE_TOKEN_ID, 4])]), images),
            "do not match",
        ),
        (lambda ids, images: (ids.roll(1), images), "do not match"),
        (lambda ids, images: (ids[:-1], images), "followed by a text token"),
        (
            lambda ids, images: (
                ids,
                ImageInputs(images.pixel_values[:, :-1].contiguous(), images.spans),
            ),
            "768 values",
        ),
        (
            lambda ids, images: (
                ids,
                ImageInputs(images.pixel_values, (replace(images.spans[0], grid_thw=(2, 2, 4)),)),
            ),
            "one temporal patch",
        ),
    ),
)
def test_token_id_prompts_with_stray_or_misplaced_placeholders_are_rejected(
    edit: object, message: str
) -> None:
    input_ids, images = edit(*_token_prompt())

    with pytest.raises(ValueError, match=message):
        _processor().validate_token_ids(input_ids, images)


def test_token_id_prompts_obey_the_request_image_limits() -> None:
    spans = (ImageSpan(1, 100, (1, 20, 20), bytes(32)), ImageSpan(102, 4, (1, 4, 4), bytes(32)))
    input_ids = torch.tensor([3, *[_IMAGE_TOKEN_ID] * 100, 4, *[_IMAGE_TOKEN_ID] * 4, 4])
    images = ImageInputs(torch.zeros(416, _PATCH_DIM, dtype=torch.uint8), spans)

    with pytest.raises(ValueError, match="at most 64 tokens"):
        _processor(max_image_tokens=64).validate_token_ids(input_ids, images)
    with pytest.raises(ValueError, match="at most 1 images"):
        _processor(max_images=1).validate_token_ids(input_ids, images)


@pytest.mark.parametrize(
    ("image_processor", "message"),
    (
        (_image_processor(patch_size=14), "patch_size"),
        (_image_processor(merge_size=1), "merge_size"),
        (
            _image_processor(image_mean=[0.5] * 4, image_std=[0.5] * 4),
            "normalizes 4 channels, the model has 3",
        ),
    ),
)
def test_processor_rejects_inconsistent_configuration(
    image_processor: Qwen2VLImageProcessorPil, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Qwen3_5InputProcessor(_config(), image_processor, limits=ImageLimits(4, 16))


@pytest.mark.parametrize("max_image_tokens", (0, 63))
def test_a_budget_below_the_smallest_image_takes_no_images_but_checks_token_ids(
    max_image_tokens: int,
) -> None:
    encoder = _encoder(max_image_tokens=max_image_tokens)
    input_ids, _ = _token_prompt()

    assert encoder.text("Describe this").images is None
    with pytest.raises(ValueError, match="the smallest image needs 64 tokens"):
        encoder.text("<|image_pad|> this", [_image(64, 64, 11)])
    with pytest.raises(ValueError, match="placeholders but no images"):
        encoder.check(ProcessedPrompt(input_ids, None, None))


def test_a_vision_checkpoint_without_image_preprocessing_takes_the_processor_defaults(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        processor = create_input_processor(
            "Qwen3_5ForConditionalGeneration",
            str(tmp_path),
            hf_config=_config(),
            limits=ImageLimits(2, 1024),
        )

    assert f"{tmp_path} has no preprocessor_config.json" in caplog.text
    prompt = PromptEncoder(TokenizerTextDecoder(_tokenizer()), processor, len(_VOCABULARY)).text(
        "<|image_pad|> this", [_image(64, 64, 11)]
    )
    assert prompt.images.spans[0].num_tokens == 4


@pytest.mark.parametrize(
    "architecture", ("Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration")
)
def test_vision_language_architectures_load_their_processor(
    architecture: str, tmp_path: Path
) -> None:
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(dict(QWEN3_5_PREPROCESSOR_CONFIG))
    )

    processor = create_input_processor(
        architecture,
        str(tmp_path),
        hf_config=_config(),
        limits=ImageLimits(2, 1024),
    )

    assert isinstance(processor, Qwen3_5InputProcessor)
    prompt = PromptEncoder(TokenizerTextDecoder(_tokenizer()), processor, len(_VOCABULARY)).text(
        "<|image_pad|> this", [_image(64, 64, 11)]
    )
    assert prompt.images.spans[0].num_tokens == 64


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        # 16777216 pixels over 32 x 32 pixels per token.
        ({}, QWEN3_5_IMAGE_PREPROCESSING),
        # A step the image processor skips is an identity.
        (
            {"do_rescale": False, "do_normalize": False},
            ImagePreprocessing(64, 16384, 1.0, (0.0,) * 3, (1.0,) * 3),
        ),
        # Without one, the image processor's defaults: 3136 and 1003520 pixels and CLIP's
        # statistics.
        (
            None,
            ImagePreprocessing(
                4,
                980,
                1 / 255,
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ),
    ),
    ids=["preprocessing", "skipped-steps", "defaults"],
)
def test_reading_the_preprocessing_returns_it_and_leaves_the_config_alone(
    overrides: dict[str, object] | None, expected: ImagePreprocessing, tmp_path: Path
) -> None:
    if overrides is not None:
        (tmp_path / "preprocessor_config.json").write_text(
            json.dumps({**QWEN3_5_PREPROCESSOR_CONFIG, **overrides})
        )
    config = _config()
    before = config.to_dict()

    read = read_image_preprocessing(
        "Qwen3_5ForConditionalGeneration", str(tmp_path), hf_config=config
    )

    assert read == expected
    assert config.to_dict() == before
    assert read_image_preprocessing("Qwen3ForCausalLM", str(tmp_path), hf_config=config) is None
