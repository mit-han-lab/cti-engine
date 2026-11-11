"""One tensor-parallel GPU case: its test module run as a child process in one mode."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tests.support.gpu import isolated_runtime_env, run_isolated_process
from tests.support.paths import REPO_ROOT, source_pythonpath

# Short chats the real-model cases serve.
CHAT_PROMPTS = (
    "Explain why the sky is blue in two sentences.",
    "Write a haiku about a lighthouse.",
    "What is the derivative of x^3 + 2x?",
    "List three uses of a paperclip.",
    "Summarize the plot of Hamlet briefly.",
    "How does a refrigerator keep food cold?",
    "Give a short tip for learning a new language.",
    "What causes the seasons on Earth?",
    "Translate 'good morning' into French and Spanish.",
    "Why do cats purr?",
    "Describe the water cycle.",
    "What is 17 times 23? Show the steps.",
    "Name a famous bridge and where it is.",
    "How do vaccines train the immune system?",
    "Suggest a name for a friendly robot.",
    "What is the difference between weather and climate?",
)


def run_case(
    script: Path,
    tmp_path: Path,
    component: str,
    model: Path,
    tp: int,
    mode: str,
    *,
    observe: str | None = None,
    arguments: tuple[str, ...] = (),
    expected: tuple[str, ...] = (),
) -> dict:
    """Run ``script --model MODEL --tp TP --mode MODE`` and return its receipt. With
    ``observe``, the variable its engine processes' observers read, they write their
    evidence beside the receipt; ``expected`` are the forbidden log patterns a case that
    fails on purpose emits."""

    env, logs = isolated_runtime_env(tmp_path, component)
    env.update(
        # Every rank compares each batch's digest with its peers' before running it.
        CTI_CHECKS="1",
        PYTHONPATH=source_pythonpath(REPO_ROOT),
        PYTHONNOUSERSITE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    receipt = logs / f"tp{tp}-{mode}.json"
    if observe is not None:
        observe_dir = receipt.with_suffix("")
        observe_dir.mkdir()
        env[observe] = str(observe_dir)
    run_isolated_process(
        (
            sys.executable,
            "-B",
            str(script.resolve()),
            *("--model", str(model), "--tp", str(tp), "--mode", mode),
            *("--receipt", str(receipt)),
            *arguments,
        ),
        logs / f"tp{tp}-{mode}.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=1200,
        expected=expected,
    )
    return json.loads(receipt.read_text())


__all__ = ["CHAT_PROMPTS", "run_case"]
