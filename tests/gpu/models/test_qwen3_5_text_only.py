"""Qwen3.5 loaded text only (``model.text_only``) serves the same greedy text as the default
load, without its vision tower, and refuses a prompt with images."""

import json
import sys

import pytest

from tests.support.checkpoints import QWEN3_5_9B
from tests.support.gpu import offline_child_env, run_isolated_process
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5_gpu import selected_checkpoint

pytestmark = pytest.mark.checkpoints(QWEN3_5_9B)

_TOKENS = 32
_RECEIPT = "TEXT_ONLY="
_PROMPTS = ["The capital of France is", "def fibonacci(n):"]


def test_a_text_only_load_serves_the_default_text(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    env, logs = offline_child_env(tmp_path, "text-only")
    output = run_isolated_process(
        (sys.executable, "-m", __name__, str(selected_checkpoint().snapshot())),
        logs / "child.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=1200,
    )
    (line,) = [line for line in output.splitlines() if line.startswith(_RECEIPT)]
    receipt = json.loads(line.removeprefix(_RECEIPT))
    assert receipt["text_only"] == receipt["default"]
    assert all(len(token_ids) == _TOKENS for token_ids in receipt["default"])
    assert receipt["refusal"] == "the model takes no images"


def _run(snapshot):
    from cti.config import EngineOptions, ModelOptions
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    params = SamplingParams(max_new_tokens=_TOKENS, ignore_eos=True, temperature=0.0)
    receipt = {}
    for text_only in (False, True):
        llm = LLM(
            EngineOptions(model=ModelOptions(path=snapshot, max_seq_len=4096, text_only=text_only))
        )
        try:
            outputs = llm.generate(_PROMPTS, params)
            receipt["text_only" if text_only else "default"] = [
                output["token_ids"] for output in outputs
            ]
            if text_only:
                image = {"type": "image_url", "image_url": {"url": _PIXEL}}
                chat = [{"role": "user", "content": [image, {"type": "text", "text": "Hi"}]}]
                try:
                    llm.chat([chat], params)
                except ValueError as error:
                    receipt["refusal"] = str(error)
        finally:
            llm.shutdown()
    print(_RECEIPT + json.dumps(receipt), flush=True)


# A 1x1 PNG.
_PIXEL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQ"
    "DwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


if __name__ == "__main__":
    from tests.support.qwen3_5_prefix import child_main

    child_main(_run, *sys.argv[1:])
