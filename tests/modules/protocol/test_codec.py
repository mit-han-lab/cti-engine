from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import msgpack
import pytest
import torch

from cti.multimodal import ImageInputs, ImageSpan, RopePositionLayout
from cti.multimodal.qwen3_5 import Qwen3_5RopeLayout
from cti.protocol import (
    AdmissionPriority,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    ChatPrompt,
    DetokenizeMsg,
    PromptReply,
    RejectionKind,
    RequestEnd,
    RequestRejectedMsg,
    SamplingParams,
    TextPrompt,
    TokenizeMsg,
    TokenPrompt,
    TokenUsage,
    UserMsg,
    UserReply,
)
from cti.sampling import TokenLogprobs
from tests.support.imports import assert_not_loaded


def codec(hint: object):
    """The request link's codec of values of the declared type ``hint``."""

    return BaseBackendMsg.family().codec(hint)


def _wire(value: object) -> object:
    """``value`` across the msgpack wire format the ZMQ queues use."""

    return msgpack.unpackb(msgpack.packb(value, use_bin_type=True), raw=False, strict_map_key=False)


def test_a_message_crosses_as_its_index_and_fields_with_enums_as_values() -> None:
    message = UserMsg(
        uid=0,
        input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        sampling=SamplingParams(max_new_tokens=1024, temperature=0).snapshot(),
        priority=AdmissionPriority.HIGH,
    )

    encoded = BaseBackendMsg.family().encode(message)
    result = BaseBackendMsg.family().decode(_wire(encoded))

    assert encoded[0] == 0 and encoded[1] == 0 and encoded[4] == "high"
    assert isinstance(result, UserMsg) and result.priority is AdmissionPriority.HIGH


@pytest.mark.parametrize("priority", [True, 1, 1.5, object()])
def test_request_priority_rejects_non_strings(priority: object) -> None:
    with pytest.raises(TypeError, match="priority must be high, normal, or low"):
        UserMsg(
            uid=0,
            input_ids=torch.tensor([1], dtype=torch.int32),
            sampling=SamplingParams(max_new_tokens=1024, temperature=0).snapshot(),
            priority=priority,
        )
    with pytest.raises(TypeError, match="priority must be high, normal, or low"):
        TokenizeMsg(
            uid=0,
            prompt=TextPrompt("hello"),
            sampling=SamplingParams(max_new_tokens=1024, temperature=0).snapshot(),
            priority=priority,
        )


def test_request_priority_rejects_unknown_tier() -> None:
    with pytest.raises(ValueError, match="priority must be high, normal, or low"):
        UserMsg(
            uid=0,
            input_ids=torch.tensor([1], dtype=torch.int32),
            sampling=SamplingParams(max_new_tokens=1024, temperature=0).snapshot(),
            priority="urgent",
        )


@pytest.mark.parametrize(
    ("message", "protocol"),
    ((RequestRejectedMsg(uid=7, reason="capacity exhausted"), BaseFrontendMsg),),
)
def test_failure_protocol_round_trip(message: object, protocol: object) -> None:
    assert protocol.family().decode(protocol.family().encode(message)) == message


def test_a_requests_stops_cross_the_wire() -> None:
    policy = SamplingParams(
        max_new_tokens=1024,
        temperature=0,
        stop=("\n\n", "END"),
        stop_token_ids=(7,),
        skip_special_tokens=False,
        logprobs=True,
    ).snapshot()
    request = UserMsg(uid=3, input_ids=torch.tensor([1, 2], dtype=torch.int32), sampling=policy)
    stopped = DetokenizeMsg(
        uid=3,
        token_ids=(4, 5),
        skip_special_tokens=False,
        logprobs=(TokenLogprobs(4, -0.1, ((4, -0.1),)), TokenLogprobs(5, -0.2, ((5, -0.2),))),
        end=RequestEnd(TokenUsage(2, 2), 1024, stop_str="END"),
    )

    received = BaseBackendMsg.family().decode(_wire(BaseBackendMsg.family().encode(request)))
    assert received.sampling == policy
    decoded = BaseTokenizerMsg.family().decode(_wire(BaseTokenizerMsg.family().encode(stopped)))
    assert decoded == stopped and decoded.end.finish_reason == "stop"


@pytest.mark.parametrize(
    ("message", "protocol"),
    (
        (
            TokenizeMsg(
                1,
                ChatPrompt(({"role": "user", "content": "hi"},), (), {"enable_thinking": False}),
                SamplingParams(max_new_tokens=1024, temperature=0, stop=("x",)).snapshot(),
            ),
            BaseTokenizerMsg,
        ),
        (
            TokenizeMsg(
                2,
                TokenPrompt((5, 6)),
                SamplingParams(max_new_tokens=1024, temperature=0).snapshot(),
                return_prompt=True,
            ),
            BaseTokenizerMsg,
        ),
        (
            TokenizeMsg(
                3, TextPrompt("hi"), SamplingParams(max_new_tokens=1024, temperature=0).snapshot()
            ),
            BaseTokenizerMsg,
        ),
        (PromptReply(2, (5, 6), "ab", {5: "a"}), BaseFrontendMsg),
        (
            UserReply(
                2,
                "c",
                True,
                TokenUsage(2, 1),
                "stop",
                token_ids=(7,),
                stop_str="c",
                logprobs=(TokenLogprobs(7, -0.5, ((7, -0.5), (8, -1.0))),),
                prompt_logprobs=(None, TokenLogprobs(6, -2.0)),
                token_texts={7: "c", 8: "d", 6: "b"},
            ),
            BaseFrontendMsg,
        ),
    ),
)
def test_prompts_and_replies_cross_the_wire(message: object, protocol: object) -> None:
    wire = msgpack.packb(protocol.family().encode(message))
    assert protocol.family().decode(msgpack.unpackb(wire, strict_map_key=False)) == message


@pytest.mark.parametrize(
    ("message_type", "protocol"),
    ((RequestRejectedMsg, BaseTokenizerMsg), (RequestRejectedMsg, BaseFrontendMsg)),
)
def test_rejection_kind_crosses_the_wire_as_its_enum(message_type: type, protocol: object) -> None:
    message = message_type(uid=7, reason="too long", kind=RejectionKind.CONTEXT_LENGTH_EXCEEDED)
    wire = msgpack.unpackb(
        msgpack.packb(protocol.family().encode(message), use_bin_type=True), raw=False
    )

    assert protocol.family().decode(wire).kind is RejectionKind.CONTEXT_LENGTH_EXCEEDED


def _wire_round_trip(value: object, hint: object) -> object:
    """``value`` encoded as the declared type ``hint``, across the wire, and decoded."""

    half = codec(hint)
    return half.decode(_wire(half.encode(value)))


@dataclass(frozen=True, slots=True, eq=False)
class _Tensors:
    values: tuple[torch.Tensor, ...]


@pytest.mark.parametrize(
    "tensor",
    (
        torch.tensor([[1.5, -0.0], [float("nan"), float("-inf")]]).to(torch.bfloat16),
        torch.randn(2, 3, 4).to(torch.bfloat16)[:, 1:],
        torch.randn(5, 3).to(torch.float16).t(),
        torch.arange(-6, 6, dtype=torch.int64).reshape(3, 2, 2),
        torch.tensor([True, False, True]),
        torch.tensor(7, dtype=torch.int32),
        torch.empty(0, 1536, dtype=torch.bfloat16),
        torch.randint(256, (3, 768), dtype=torch.uint8),
    ),
)
def test_tensors_of_any_shape_and_dtype_round_trip_bit_exactly(tensor: torch.Tensor) -> None:
    (result,) = _wire_round_trip(_Tensors((tensor,)), _Tensors).values

    assert result.dtype == tensor.dtype
    assert result.shape == tensor.shape
    assert result.is_contiguous()
    assert torch.equal(
        result.reshape(-1).view(torch.uint8), tensor.contiguous().reshape(-1).view(torch.uint8)
    )


def test_multimodal_inputs_round_trip_bit_exactly() -> None:
    spans = (
        ImageSpan(2, 4, (1, 4, 4), bytes(range(32))),
        ImageSpan(9, 2, (1, 2, 4), bytes(range(32, 64))),
    )
    pixels = torch.randint(256, (24, 768), dtype=torch.uint8)
    images = ImageInputs(pixels, spans)
    layout = Qwen3_5RopeLayout.from_spans(spans, 2)

    decoded_images = _wire_round_trip(images, ImageInputs)
    decoded_layout = _wire_round_trip(layout, RopePositionLayout)

    assert decoded_images.spans == spans
    assert type(decoded_images.spans[0].grid_thw) is tuple
    assert torch.equal(decoded_images.pixel_values, pixels)
    assert decoded_layout == layout
    # An architecture's layout travels by name and its own fields.
    assert codec(RopePositionLayout).encode(layout)[0] == "Qwen3_5RopeLayout"


def test_codec_rejects_unknown_values() -> None:
    with pytest.raises(TypeError, match="no codec for"):
        codec(object)
    with pytest.raises(TypeError, match="SamplingParams is not a message of this link"):
        BaseTokenizerMsg.family().encode(SamplingParams(max_new_tokens=1))
    with pytest.raises(ValueError, match="tensor dtype 'Size'"):
        codec(torch.Tensor).decode(["Size", [], b""])


def test_backend_messages_cross_the_wire_format() -> None:
    message = UserMsg(
        uid=3,
        input_ids=torch.tensor([5, 6], dtype=torch.int32),
        sampling=SamplingParams(max_new_tokens=1024, top_k=4).snapshot(),
    )
    result = BaseBackendMsg.family().decode(_wire(BaseBackendMsg.family().encode(message)))

    assert result.uid == 3
    assert result.input_ids.dtype == torch.int32
    assert result.input_ids.tolist() == [5, 6]
    assert result.sampling == SamplingParams(max_new_tokens=1024, top_k=4).snapshot()


def test_a_user_message_carries_its_images_and_layout_across_the_wire(tmp_path: Path) -> None:
    spans = (
        ImageSpan(2, 4, (1, 4, 4), bytes(range(32))),
        ImageSpan(9, 2, (1, 2, 4), bytes(range(32, 64))),
    )
    pixels = torch.randint(256, (24, 768), dtype=torch.uint8)
    layout = Qwen3_5RopeLayout.from_spans(spans, 2)
    messages = [
        UserMsg(
            uid=4,
            input_ids=torch.arange(12, dtype=torch.int32),
            sampling=SamplingParams(max_new_tokens=1024, temperature=0).snapshot(),
            images=ImageInputs(pixels, spans),
            rope_layout=layout,
        ),
        UserMsg(
            uid=5,
            input_ids=torch.tensor([7], dtype=torch.int32),
            sampling=SamplingParams(max_new_tokens=1024, temperature=0).snapshot(),
        ),
    ]
    encoded = msgpack.packb(
        [BaseBackendMsg.family().encode(m) for m in messages], use_bin_type=True
    )
    wire = msgpack.unpackb(encoded, raw=False)

    with_images, text = (BaseBackendMsg.family().decode(message) for message in wire)

    assert with_images.images.spans == spans
    assert torch.equal(with_images.images.pixel_values, pixels)
    assert with_images.rope_layout == layout
    assert with_images.rope_layout.decode_offset == layout.decode_offset
    assert (text.images, text.rope_layout) == (None, None)

    # A TP peer has never constructed a processor or imported a model layout. Decode
    # in a fresh interpreter to avoid hiding missing registrations through test imports.
    payload = tmp_path / "backend.msgpack"
    payload.write_bytes(encoded)
    assert_not_loaded(
        "import msgpack\n"
        "from pathlib import Path\n"
        "from cti.protocol import BaseBackendMsg\n"
        f"wire = msgpack.unpackb(Path({str(payload)!r}).read_bytes(), raw=False)\n"
        "images, text = (BaseBackendMsg.family().decode(message) for message in wire)\n"
        f"assert images.rope_layout.images == {layout.images!r}\n"
        f"assert images.rope_layout.decode_offset == {layout.decode_offset!r}\n"
        "assert text.rope_layout is None",
        "transformers",
        "PIL",
        "cti.multimodal.qwen3_5.processor",
    )


@pytest.mark.parametrize("name", ["UnknownLayout", "os.system", "torch", "dataclass"])
def test_a_layout_decodes_only_as_an_architectures(name: str) -> None:
    layout = Qwen3_5RopeLayout.from_spans((ImageSpan(0, 1, (1, 2, 2), bytes(32)),), 2)
    wire = _wire(codec(RopePositionLayout).encode(layout))
    wire[0] = name

    with pytest.raises(KeyError, match=name):
        codec(RopePositionLayout).decode(wire)


def test_a_link_decodes_only_its_own_messages() -> None:
    with pytest.raises(IndexError):
        BaseBackendMsg.family().decode([7, 0])
