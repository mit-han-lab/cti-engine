from __future__ import annotations

import json
import time
from typing import Any

from .._profile import RequestTiming


async def measure_stream_request(
    client: Any,
    model: str,
    prompt: str,
    output_tokens: int,
) -> RequestTiming:
    """Measure one cti OpenAI stream and validate its timing observations."""
    started = time.perf_counter()
    token_times: list[float] = []
    usage = None
    terminal_chunks = 0
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "top_k": 1,
    }
    async with client.stream("POST", "/chat/completions", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            encoded = line.removeprefix("data:").strip()
            if not encoded or encoded == "[DONE]":
                continue
            chunk = json.loads(encoded)
            chunk_usage = chunk.get("usage")
            if chunk_usage is not None:
                if usage is not None:
                    raise RuntimeError("stream emitted token usage more than once")
                usage = chunk_usage
            for choice in chunk.get("choices", ()):
                # cti emits one nonterminal choice chunk per generated token.
                # This validates cti receipts, not the generic OpenAI contract.
                if choice.get("finish_reason") is None:
                    token_times.append(time.perf_counter())
                else:
                    terminal_chunks += 1
    if usage is None:
        raise RuntimeError("stream completed without terminal token usage")
    if terminal_chunks != 1:
        raise RuntimeError(f"stream emitted {terminal_chunks} terminal chunks")
    return RequestTiming.from_observation(
        started=started,
        token_times=token_times,
        prompt_tokens=int(usage["prompt_tokens"]),
        completion_tokens=int(usage["completion_tokens"]),
        expected_output_tokens=output_tokens,
    )
