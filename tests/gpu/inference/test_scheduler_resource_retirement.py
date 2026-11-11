"""With one request slot and overlap scheduling, a request that ends early (a stop token, or
a client that drops its stream) hands its slot to the queued request: that one gives the golden
output, and a full golden batch still runs afterwards."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.gpu import isolated_runtime_env, run_isolated_process
from tests.support.graph_captures import graph_captures
from tests.support.inference_correctness import GOLDEN_CASES
from tests.support.paths import REPO_ROOT

pytestmark = pytest.mark.checkpoints(QWEN3_0_6B)


async def _requests(model: str, port: int) -> None:
    import asyncio

    from openai import AsyncOpenAI

    first, replacement = GOLDEN_CASES
    async with AsyncOpenAI(
        base_url=f"http://127.0.0.1:{port}/v1", api_key="test", max_retries=0, timeout=60
    ) as client:

        def complete(prompt, max_tokens: int, **options):
            return client.completions.create(
                model=model, prompt=prompt, temperature=0, max_tokens=max_tokens, **options
            )

        def golden(case):
            return complete(
                list(case.prompt_ids), len(case.output_ids), extra_body={"ignore_eos": True}
            )

        # The first request stops at its first token while the replacement waits for its slot.
        stopped, replaced = await asyncio.gather(
            complete(
                list(first.prompt_ids), 4, extra_body={"stop_token_ids": [first.output_ids[0]]}
            ),
            golden(replacement),
        )
        assert (stopped.choices[0].finish_reason, stopped.choices[0].text) == ("stop", "")
        assert replaced.choices[0].text == replacement.output_text

        # A client drops a long stream after its first text while the replacement waits.
        stream = await complete(
            list(first.prompt_ids), 400, stream=True, extra_body={"ignore_eos": True}
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].text:
                assert first.output_text.startswith(chunk.choices[0].text)
                break
        queued = asyncio.create_task(golden(replacement))
        await stream.close()
        assert (await queued).choices[0].text == replacement.output_text

        # Afterwards the engine still serves a full golden batch.
        responses = await asyncio.gather(*(golden(case) for case in GOLDEN_CASES))
        assert [r.choices[0].text for r in responses] == [c.output_text for c in GOLDEN_CASES]


def _run_child() -> None:
    import asyncio
    import socket

    from cti.config import (
        CacheOptions,
        EngineOptions,
        EngineTuning,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
        ServerOptions,
    )
    from cti.server import OnlineServer

    model = str(QWEN3_0_6B.snapshot())
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with OnlineServer(
        EngineOptions(
            model=ModelOptions(path=model, max_seq_len=512),
            memory=MemoryOptions(inference_ratio=0.5),
            scheduling=SchedulingOptions(max_running_requests=1, max_batch_tokens=128),
            cache=CacheOptions(prefix="radix"),
        ),
        ServerOptions(port=port),
        tuning=EngineTuning(attention_backend="fi", cuda_graph_bs=(1,)),
    ):
        asyncio.run(_requests(model, port))


def test_early_finished_requests_hand_their_slot_to_the_queued_one(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "resource-retirement")
    env.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.inference.test_scheduler_resource_retirement"),
        log_root / "resource-retirement.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=300,
    )
    assert output.count("Selected attention backend fi: explicit attention backend") == 1
    assert graph_captures(output, "decode", [1]) == 1


if __name__ == "__main__":
    _run_child()
