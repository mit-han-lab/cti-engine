"""Qwen3-MoE against Transformers on a pinned checkpoint: teacher-forced top-1 tokens and
logprobs, cold and through prefix hits, against an FP32 truth within bounds calibrated by BF16
Transformers (``tests/support/hf_parity.py``); and each published variant's chat template
rendered by the engine as Transformers renders it. Run by hand only: ``README.md``."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.hf_parity import ParityRun, ParitySettings, check_parity
from tests.support.qwen3_moe import PUBLISHED

# The CTI_QWEN3_MOE_* settings (README.md).
SETTINGS = ParitySettings("CTI_QWEN3_MOE_")
# The checkpoints whose token ids the parity prompts hold.
PARITY_CHECKPOINTS = (PUBLISHED["30b-a3b"].pin,)
# The checkpoints whose chat template takes enable_thinking: the hybrid one's.
_THINKING_SWITCH = frozenset({"30b-a3b", "30b-a3b-base"})
# Each chat template: one turn, a system prompt, several turns.
_CONVERSATIONS = (
    [{"role": "user", "content": "What is 13 times 17?"}],
    [
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": "Name a noble gas."},
    ],
    [
        {"role": "user", "content": "Add 2 and 3."},
        {"role": "assistant", "content": "5"},
        {"role": "user", "content": "Now double it."},
    ],
)


def test_qwen3_moe_matches_transformers(tmp_path: Path) -> None:
    """Every prompt passes every gate, cold and through prefix hits, each path allowing one
    near-tie prompt."""

    checkpoint = SETTINGS.checkpoint(PARITY_CHECKPOINTS)
    run = ParityRun.from_settings(SETTINGS, checkpoint, moe=True)
    check_parity(run, tmp_path, "qwen3_moe_hf_parity")


@pytest.mark.parametrize(
    "variant",
    [
        "30b-a3b",
        "30b-a3b-instruct-2507",
        "30b-a3b-thinking-2507",
        "30b-a3b-base",
        "coder-30b-a3b-instruct",
    ],
)
def test_the_engine_renders_each_chat_template_as_transformers(variant: str) -> None:
    """The engine's chat prompts are Transformers' token ids, with the template's default and
    with thinking turned off: only the hybrid checkpoint's template, which the base checkpoint
    ships too, reacts to that."""

    from transformers import AutoTokenizer

    from cti.tokenizer import PromptEncoder
    from cti.tokenizer.text import TokenizerTextDecoder

    pin = PUBLISHED[variant].pin
    # Its tokenizer files alone, from the local Hugging Face cache.
    tokenizer = AutoTokenizer.from_pretrained(
        pin.repo_id, revision=pin.revision, local_files_only=True
    )
    engine = PromptEncoder(TokenizerTextDecoder(tokenizer), None, len(tokenizer))
    reacting = set()
    for index, messages in enumerate(_CONVERSATIONS):
        rendered = {}
        for enable_thinking in (None, False):
            options = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
            expected = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=False, **options
            )
            ids = engine.chat(messages, template_kwargs=options).input_ids
            assert ids.tolist() == expected, (variant, index, enable_thinking)
            rendered[enable_thinking] = expected
        if rendered[False] != rendered[None]:
            reacting.add(index)
    assert bool(reacting) == (variant in _THINKING_SWITCH), (variant, reacting)
