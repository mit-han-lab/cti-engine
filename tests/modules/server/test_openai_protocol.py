"""Choice rendering: the client's text ends where the engine-named stop string begins, and
a token's id and logprob leave with the last of its text. Chat content parts: text, and
image URLs from the user."""

from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from cti.protocol import ChatPrompt, TokenUsage, UserReply
from cti.sampling import TokenLogprobs
from cti.server.openai.rendering import ChoiceOutput, ChoiceRenderer
from cti.server.openai.requests import ChatCompletionRequest, OpenAIError

_PIECES = ("a", "b", "\n", "\n\n", "ab", "baa", "END", "EN", "D", "x")
_STOPS = ("\n\n", "END", "aba", "b\nb")


def _engine(rng: random.Random, stop: tuple[str, ...]) -> tuple[list[str], str | None]:
    """Detokenized pieces up to the one completing the first stop string to end, as the
    scheduler decides it, and that string."""

    pieces: list[str] = []
    text = ""
    for _ in range(rng.randint(1, 12)):
        pieces.append(rng.choice(_PIECES))
        text += pieces[-1]
        found = [(text.find(s) + len(s), -len(s), s) for s in stop if s in text]
        if found:
            return pieces, min(found)[2]
    return pieces, None


@pytest.mark.parametrize("seed", range(300))
def test_rendered_text_is_the_output_cut_at_its_stop_string(seed: int) -> None:
    rng = random.Random(seed)
    stop = tuple(rng.sample(_STOPS, rng.randint(1, 3)))
    include = rng.random() < 0.5
    pieces, stop_str = _engine(rng, stop)
    renderer = ChoiceRenderer(stop, include_stop_str=include, wants_prompt=False, prompt_only=False)
    choice = ChoiceOutput()

    for index, piece in enumerate(pieces):
        finished = index == len(pieces) - 1
        usage = TokenUsage(1, len(pieces)) if finished else None
        reason = ("stop" if stop_str else "length") if finished else None
        stop_at = stop_str if finished else None
        logprobs = TokenLogprobs(index, -1.0)
        reply = UserReply(0, piece, finished, usage, reason, (index,), stop_at, (logprobs,))
        reply.token_texts[index] = piece
        for delta in renderer.add(reply):
            choice.add(delta)
            assert [entry.token_id for entry in delta.logprobs] == list(delta.token_ids)
            released = "".join(pieces[token_id] for token_id in choice.token_ids)
            assert finished or len(released) <= len(choice.text)

    text = "".join(pieces)
    if stop_str is not None:
        begin = text.find(stop_str)
        text = text[: begin + len(stop_str)] if include else text[:begin]
    assert choice.text == text
    assert choice.token_ids == list(range(len(pieces)))
    assert [choice.token_texts[entry.token_id] for entry in choice.logprobs] == pieces


def _chat_request(content: object, role: str = "user") -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": role, "content": content}]}
    )


@pytest.mark.parametrize("detail", ("auto", "low", "high"))
def test_chat_content_parts_reach_the_prompt(detail: str) -> None:
    image = {"url": "https://example.com/cat.png", "detail": detail}
    content = [{"type": "text", "text": "Hi"}, {"type": "image_url", "image_url": image}]

    [prompt] = _chat_request(content).prompts()

    assert prompt == ChatPrompt(({"role": "user", "content": content},))


_IMAGE_PART = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}


@pytest.mark.parametrize(
    ("role", "content", "param", "message"),
    (
        ("system", [_IMAGE_PART], "messages[0].content[0]", "only user messages"),
        ("assistant", [_IMAGE_PART], "messages[0].content[0]", "only user messages"),
        ("user", [{"type": "input_audio"}], "messages[0].content[0].type", "not supported"),
        ("user", [{"text": "Hi"}], "messages[0].content[0].type", "not supported"),
        ("user", [], "messages[0].content", "non-empty list"),
        (
            "user",
            [{"type": "image_url", "image_url": {"url": "x", "detail": "ultra"}}],
            "messages[0].content[0].image_url.detail",
            "'auto', 'low' or 'high'",
        ),
        (
            "user",
            [{"type": "image_url", "image_url": {}}],
            "messages[0].content[0].image_url.url",
            "required",
        ),
        (
            "user",
            [{"type": "text", "text": "Hi", "x": 1}],
            "messages[0].content[0].x",
            "Unrecognized",
        ),
    ),
)
def test_invalid_content_parts_name_their_path(
    role: str, content: object, param: str, message: str
) -> None:
    with pytest.raises(ValidationError) as raised:
        _chat_request(content, role)

    error = OpenAIError.from_validation(raised.value.errors())
    assert (error.param, error.status_code) == (param, 400)
    assert message in error.message
