from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from cookbook.chat import _resolve_model, run_chat


class _ChatClient:
    def __init__(
        self,
        models: tuple[str, ...],
        responses: tuple[tuple[str, ...], ...] = (),
    ) -> None:
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=tuple(SimpleNamespace(id=model) for model in models),
            )
        )
        self.requests: list[dict[str, object]] = []
        self._responses = iter(responses)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_completion),
        )

    def _create_completion(self, **request: object) -> tuple[SimpleNamespace, ...]:
        self.requests.append(request)
        return tuple(
            SimpleNamespace(
                choices=(SimpleNamespace(delta=SimpleNamespace(content=fragment)),),
            )
            for fragment in next(self._responses)
        )


def test_chat_resolves_the_only_server_model() -> None:
    client = _ChatClient(("Qwen/Qwen3-0.6B",))

    assert _resolve_model(client, None) == "Qwen/Qwen3-0.6B"  # type: ignore[arg-type]
    assert _resolve_model(client, "explicit") == "explicit"  # type: ignore[arg-type]


@pytest.mark.parametrize("models", ((), ("first", "second")))
def test_chat_requires_an_explicit_model_for_an_ambiguous_server(
    models: tuple[str, ...],
) -> None:
    client = _ChatClient(models)

    with pytest.raises(RuntimeError, match="--model is required"):
        _resolve_model(client, None)  # type: ignore[arg-type]


def test_chat_streams_public_api_responses_and_resets_local_history() -> None:
    client = _ChatClient(
        ("Qwen/Qwen3-0.6B",),
        (("first", " response"), ("second",)),
    )
    commands = iter(("hello", "/reset", "again", "/exit"))
    output = io.StringIO()

    history = run_chat(  # type: ignore[arg-type]
        client,
        "Qwen/Qwen3-0.6B",
        max_tokens=32,
        input_fn=lambda _prompt: next(commands),
        output=output,
    )

    assert history == (
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "second"},
    )
    assert output.getvalue() == "first response\nsecond\n"
    assert [request["messages"] for request in client.requests] == [
        [{"role": "user", "content": "hello"}],
        [{"role": "user", "content": "again"}],
    ]
    assert all(request["stream"] is True for request in client.requests)
    assert all(request["max_tokens"] == 32 for request in client.requests)
