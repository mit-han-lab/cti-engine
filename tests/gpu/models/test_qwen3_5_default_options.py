"""Qwen3.5 starts with every option at its default, max_running_requests sized to the memory
budget (at most 256) and logged once, and serves a greedy request."""

import re
import sys

import pytest

from tests.support.checkpoints import QWEN3_5_9B
from tests.support.gpu import offline_child_env, run_isolated_process
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5_gpu import selected_checkpoint

pytestmark = pytest.mark.checkpoints(QWEN3_5_9B)

_TOKENS = 8
_RECEIPT = "DEFAULT_OPTIONS_TOKENS="
_AUTO = re.compile(r"max_running_requests=(\d+) \(auto: .*; cap 256\)")


def test_default_options_start_and_serve(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    env, logs = offline_child_env(tmp_path, "default-options")
    output = run_isolated_process(
        (sys.executable, "-m", __name__, str(selected_checkpoint().snapshot())),
        logs / "child.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=900,
    )
    (chosen,) = _AUTO.findall((logs / "child.log").read_text())
    assert 1 <= int(chosen) <= 256
    (line,) = [line for line in output.splitlines() if line.startswith(_RECEIPT)]
    assert int(line.removeprefix(_RECEIPT)) == _TOKENS


def _run(snapshot):
    from cti.config import EngineOptions, ModelOptions
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    llm = LLM(EngineOptions(model=ModelOptions(path=snapshot)))
    try:
        params = SamplingParams(max_new_tokens=_TOKENS, ignore_eos=True, temperature=0.0)
        (output,) = llm.generate(["The capital of France is"], params)
    finally:
        llm.shutdown()
    print(_RECEIPT + str(len(output["token_ids"])), flush=True)


if __name__ == "__main__":
    from tests.support.qwen3_5_prefix import child_main

    child_main(_run, *sys.argv[1:])
