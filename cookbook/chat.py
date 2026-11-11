"""Chat with an already-running cti OpenAI-compatible server."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from typing import TextIO, TypeAlias

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)

Message: TypeAlias = ChatCompletionMessageParam


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _resolve_model(client: OpenAI, requested_model: str | None) -> str:
    if requested_model is not None:
        return requested_model

    models = tuple(model.id for model in client.models.list().data)
    if len(models) != 1:
        raise RuntimeError(
            f"--model is required unless the server exposes exactly one model; found {len(models)}"
        )
    return models[0]


def _stream_completion(
    client: OpenAI,
    model: str,
    messages: list[Message],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    output: TextIO,
) -> str:
    fragments: list[str] = []
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", file=output, flush=True)
            fragments.append(content)
    print(file=output, flush=True)
    return "".join(fragments)


def run_chat(
    client: OpenAI,
    model: str,
    *,
    max_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 1.0,
    input_fn: Callable[[str], str] = input,
    output: TextIO | None = None,
) -> tuple[Message, ...]:
    """Run one terminal conversation and return its final local history."""

    output = sys.stdout if output is None else output
    history: list[Message] = []
    while True:
        try:
            command = input_fn("$ ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            print(file=output)
            break

        if not command:
            continue
        if command == "/exit":
            break
        if command == "/reset":
            history.clear()
            continue
        if command.startswith("/"):
            print(f"Unknown command: {command}", file=output, flush=True)
            continue

        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": command,
        }
        request_messages = [*history, user_message]
        try:
            response = _stream_completion(
                client,
                model,
                request_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                output=output,
            )
        except KeyboardInterrupt:
            print(file=output)
            break
        assistant_message: ChatCompletionAssistantMessageParam = {
            "role": "assistant",
            "content": response,
        }
        history.extend((user_message, assistant_message))
    return tuple(history)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "unused"))
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=_positive_int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    with OpenAI(base_url=args.base_url, api_key=args.api_key) as client:
        model = _resolve_model(client, args.model)
        run_chat(
            client,
            model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )


if __name__ == "__main__":
    main()
