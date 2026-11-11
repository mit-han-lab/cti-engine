"""Tool-call parsing: each format's calls and content, whatever deltas the text arrives in."""

from __future__ import annotations

import json
import random
from typing import get_args, get_type_hints

import pytest

from cti.config import ServerOptions
from cti.server.tool_calls import (
    PARSERS,
    ParseDelta,
    ToolCallFormat,
    opens_think,
    parser_for,
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "metric": {"type": "boolean"},
                    "scale": {"type": "number"},
                    "hours": {"type": "array"},
                    "note": {"type": ["integer", "string"]},
                },
            },
        },
    },
    {"type": "function", "function": {"name": "now"}},
]


def _merge(deltas: list[ParseDelta]) -> tuple[str, list[tuple[str, str]]]:
    """The content, and each call's name and whole arguments, of a run of deltas."""

    content = "".join(delta.content for delta in deltas)
    calls: dict[int, list[str]] = {}
    for delta in deltas:
        for call in delta.calls:
            if call.name is not None:
                assert call.index not in calls, "a call is named once"
                calls[call.index] = [call.name, ""]
            calls[call.index][1] += call.arguments
    return content, [(name, arguments) for name, arguments in calls.values()]


def _parse(fmt: str, text: str, chunks: list[str] | None = None, *, in_think: bool = False):
    parser = ToolCallFormat(fmt, in_think).parser(_TOOLS)
    deltas = [parser.feed(chunk) for chunk in (chunks if chunks is not None else [text])]
    return _merge([*deltas, parser.finish()])


def _hermes(name: str, arguments: dict) -> str:
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"


def _coder(name: str, **parameters: str) -> str:
    body = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in parameters.items()
    )
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


_WEATHER = {"city": "Paris {x} </b>", "days": 3}
# Format, text, the content and the calls it parses to.
_GOLDEN = {
    "hermes-text": ("hermes", "Just text, <b>no</b> call.", "Just text, <b>no</b> call.", []),
    "hermes-one": (
        "hermes",
        "Sure.\n" + _hermes("get_weather", _WEATHER) + "\nDone.",
        "Sure.\n\nDone.",
        [("get_weather", json.dumps(_WEATHER))],
    ),
    "hermes-two": (
        "hermes",
        _hermes("now", {}) + "\n" + _hermes("get_weather", {"hours": [1, {"a": "}"}]}),
        "\n",
        [("now", "{}"), ("get_weather", json.dumps({"hours": [1, {"a": "}"}]}))],
    ),
    "hermes-unknown": ("hermes", _hermes("delete_all", {}), _hermes("delete_all", {}), []),
    "hermes-truncated": ("hermes", '<tool_call>\n{"name": "ge', '<tool_call>\n{"name": "ge', []),
    "coder-one": (
        "qwen3_coder",
        "Let me check.\n" + _coder("get_weather", city="Paris\nnorth", days="3") + "\n",
        "Let me check.\n\n",
        [("get_weather", '{"city": "Paris\\nnorth", "days": 3}')],
    ),
    "coder-typed": (
        "qwen3_coder",
        _coder("get_weather", metric="True", scale="2.5", hours="[1, 2]", note="soon", city="7"),
        "",
        [
            (
                "get_weather",
                '{"metric": true, "scale": 2.5, "hours": [1, 2], "note": "soon", "city": "7"}',
            )
        ],
    ),
    "coder-unknown": ("qwen3_coder", _coder("delete_all"), _coder("delete_all"), []),
    # Thinking: a mention inside thought stays content; a call at a line start ends it.
    "think-mention": (
        "hermes",
        "<think>I could emit <tool_call> here.</think>Answer.",
        "<think>I could emit <tool_call> here.</think>Answer.",
        [],
    ),
    "think-line-start": (
        "hermes",
        "<think>Weather needed.\n" + _hermes("now", {}),
        "<think>Weather needed.\n",
        [("now", "{}")],
    ),
    "think-after": (
        "hermes",
        "<think>Weather needed.</think>\n" + _hermes("now", {}),
        "<think>Weather needed.</think>\n",
        [("now", "{}")],
    ),
    "coder-think-mention": (
        "qwen3_coder",
        "<think>Maybe <tool_call> later.</think>No call.",
        "<think>Maybe <tool_call> later.</think>No call.",
        [],
    ),
    "coder-think-line-start": (
        "qwen3_coder",
        "<think>Check.\n" + _coder("now"),
        "<think>Check.\n",
        [("now", "{}")],
    ),
    "coder-think-after": (
        "qwen3_coder",
        "<think>Check.</think>\n\n" + _coder("now"),
        "<think>Check.</think>\n\n",
        [("now", "{}")],
    ),
}


@pytest.mark.parametrize("case", list(_GOLDEN))
def test_each_format_parses_its_calls_and_keeps_the_rest_as_content(case: str) -> None:
    fmt, text, content, calls = _GOLDEN[case]

    assert _parse(fmt, text) == (content, calls)


def test_a_prompt_that_opens_thinking_starts_the_text_in_thought() -> None:
    # Qwen3.5's generation prompt ends in "<think>\n": the text holds only the close.
    text = "I could write <tool_call> now.\n</think>\n\n" + _coder("now")

    assert _parse("qwen3_coder", text, in_think=True) == (
        "I could write <tool_call> now.\n</think>\n\n",
        [("now", "{}")],
    )


@pytest.mark.parametrize("case", list(_GOLDEN))
def test_any_split_of_the_text_parses_as_the_whole(case: str) -> None:
    fmt, text, _, _ = _GOLDEN[case]
    whole = _parse(fmt, text)
    generator = random.Random(case)
    splits = [list(text)] + [
        [text[a:b] for a, b in zip(cuts, cuts[1:], strict=False)]
        for cuts in (
            [0, *sorted(generator.sample(range(1, len(text)), k=min(len(text) - 1, k))), len(text)]
            for k in (1, 3, 9)
        )
    ]

    for chunks in splits:
        assert _parse(fmt, text, chunks) == whole, chunks


def test_streamed_hermes_arguments_go_out_before_the_call_ends() -> None:
    parser = ToolCallFormat("hermes").parser(_TOOLS)
    head = parser.feed('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Pa')

    assert [(c.name, c.arguments) for c in head.calls] == [
        ("get_weather", ""),
        (None, '{"city": "Pa'),
    ]


@pytest.mark.parametrize(
    ("template", "fmt", "thinks"),
    (
        (
            "{{ '<tool_call>' }}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
            "hermes",
            False,
        ),
        (
            "<function=x><parameter=y>{% if add_generation_prompt %}<think>\n{% endif %}",
            "qwen3_coder",
            True,
        ),
        ("{{ messages[0].content }}", None, False),
        ("{{ raise_exception('no') }}", None, False),
    ),
)
def test_the_chat_template_names_the_format_and_whether_it_opens_thinking(
    template: str, fmt: str | None, thinks: bool
) -> None:
    assert parser_for(template) == fmt
    assert opens_think(template) is thinks


def test_the_server_option_names_a_parser_over_the_template_s() -> None:
    hint = get_type_hints(ServerOptions)["tool_call_parser"]
    coder = "<function=x><parameter=y>{% if add_generation_prompt %}<think>\n{% endif %}"

    assert get_args(hint) == ("auto", *PARSERS)
    assert all(ToolCallFormat(name).parser([]) for name in PARSERS)  # each one parses
    assert ToolCallFormat.of(coder) == ToolCallFormat("qwen3_coder", in_think=True)
    assert ToolCallFormat.of(coder, "hermes") == ToolCallFormat("hermes", in_think=True)
    assert ToolCallFormat.of("{{ messages }}") is None
    assert ToolCallFormat.of(None) is None
    assert ToolCallFormat.of(None, "hermes") == ToolCallFormat("hermes")
