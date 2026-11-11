"""The parity check's judge on synthetic top-1 rows: settings select the engine ranks and paths,
an engine within the calibration passes, a decided flip or a large error fails its gate, and a
MoE checkpoint excuses one prompt per path that fails only on logprob error at a near tie; a
checkpoint borrows another's prompt ids only if it renders them too."""

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tests.support.checkpoints import QWEN3_0_6B
from tests.support.hf_parity import (
    ParityRun,
    ParitySettings,
    check_borrowed_prompt_ids,
    path_failures,
    teacher_forced,
    top1_gates,
    unstable_rows,
)
from tests.support.hf_truth import tokenize_prompt

# Every row's top-two margin; row 3's is a near tie.
_MARGINS = torch.tensor([3.0, 2.0, 1.5, 0.1, 4.0, 2.5])


def _truth() -> dict[str, torch.Tensor]:
    """Six positions whose top-1 is token 0 at logit 0 and top-2 token 1, under a logsumexp of
    1, so each top-1 logprob is -1."""

    rows = len(_MARGINS)
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).repeat(rows, 1)
    logits = torch.zeros(rows, 8)
    logits[:, 1] = -_MARGINS
    logits[:, 2:] = -10.0
    return {"topk_ids": ids, "topk_logits": logits, "logsumexp": torch.ones(rows)}


def _candidate(errors: dict[int, float] = {}, flips: tuple[int, ...] = ()):  # noqa: B006
    ids = torch.zeros(len(_MARGINS), dtype=torch.int64)
    ids[list(flips)] = 1
    logprobs = torch.full((len(_MARGINS),), -1.0)
    for row, error in errors.items():
        logprobs[row] += error
    return ids, logprobs


def _comparison(name: str, bf16, engine) -> dict:
    return {"name": name, "paths": {"cold": {"engine": engine, "gates": top1_gates(bf16, engine)}}}


def test_settings_select_the_engine_ranks_and_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = ParitySettings("CTI_TEST_PARITY_")
    checkpoint = QWEN3_0_6B

    run = ParityRun.from_settings(settings, checkpoint, moe=True)
    assert (run.tensor_parallel_size, run.paths) == (1, ("cold", "prefix_hit"))
    assert ParityRun.from_settings(settings, checkpoint, moe=False).paths == ("cold",)
    monkeypatch.setenv("CTI_TEST_PARITY_TP_SIZE", "2")
    assert ParityRun.from_settings(settings, checkpoint, moe=True).tensor_parallel_size == 2
    monkeypatch.setenv("CTI_TEST_PARITY_TP_SIZE", "0")
    with pytest.raises(ValueError, match="CTI_TEST_PARITY_TP_SIZE must be a positive integer"):
        ParityRun.from_settings(settings, checkpoint, moe=True)


_WORDS = ("<unk>", "<|im_start|>", "<|im_end|>", "user", "assistant", "hi", "<think>", "x")
_TURNS = "{% for m in messages %}<|im_start|> {{ m.role }} {{ m.content }} <|im_end|> {% endfor %}"
_THINK_OFF = "{% if enable_thinking is defined and enable_thinking is false %}"
_PROMPTS = (
    {"name": "chat", "kind": "chat", "messages": [{"role": "user", "content": "hi"}]},
    {"name": "text", "kind": "text", "text": "hi x"},
)
_CHAT_KWARGS = {"add_generation_prompt": True, "enable_thinking": False}


def _save_tokenizer(folder: Path, chat_template: str) -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        models.WordLevel({word: i for i, word in enumerate(_WORDS)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.chat_template = chat_template
    tokenizer.save_pretrained(folder)
    return tokenizer


def test_a_checkpoint_borrows_prompt_ids_only_if_it_renders_them(tmp_path: Path) -> None:
    """Template files may differ (here the default of thinking); the rendered ids may not."""

    donor = _save_tokenizer(
        tmp_path / "donor",
        _TURNS + _THINK_OFF + "<|im_start|> assistant {% else %}<think> {% endif %}",
    )
    key = "org/donor@0"
    frozen = {
        "chat_template_kwargs": _CHAT_KWARGS,
        "prompts": [
            {**prompt, "token_ids": {key: tokenize_prompt(donor, prompt, _CHAT_KWARGS)}}
            for prompt in _PROMPTS
        ],
    }
    _save_tokenizer(
        tmp_path / "ours",
        _TURNS + "{% if enable_thinking %}<think> {% else %}<|im_start|> assistant {% endif %}",
    )
    check_borrowed_prompt_ids(tmp_path / "ours", frozen, key)

    _save_tokenizer(tmp_path / "other", _TURNS + "<think> ")
    with pytest.raises(ValueError, match="prompt 'chat'.*tokenizes it differently"):
        check_borrowed_prompt_ids(tmp_path / "other", frozen, key)
    (tmp_path / "bare").mkdir()
    with pytest.raises(ValueError, match="ships no tokenizer"):
        check_borrowed_prompt_ids(tmp_path / "bare", frozen, key)


def test_an_engine_within_the_calibration_passes() -> None:
    truth = _truth()
    bf16 = teacher_forced(truth, *_candidate({0: 0.01, 4: -0.02}))
    engine = teacher_forced(truth, *_candidate({1: 0.02, 5: 0.01}))

    gates = top1_gates(bf16, engine)

    assert all(gate["passed"] for gate in gates.values()), gates
    assert engine["worst_position"] == 1


def test_decided_flips_and_large_errors_fail_their_gates() -> None:
    truth = _truth()
    bf16 = teacher_forced(truth, *_candidate({0: 0.01}))

    flipped = top1_gates(bf16, teacher_forced(truth, *_candidate(flips=(0, 1, 2, 4))))
    erring = top1_gates(bf16, teacher_forced(truth, *_candidate({2: 3.0})))

    assert not flipped["decided_disagreements"]["passed"]
    failed = {name for name, gate in erring.items() if not gate["passed"]}
    assert failed >= {"top1_logprob_error_max"} and "decided_disagreements" not in failed


def test_a_declared_floor_share_relaxes_only_the_floor_ceiling() -> None:
    truth = _truth()
    bf16 = teacher_forced(truth, *_candidate(flips=(0,)))
    engine = teacher_forced(truth, *_candidate(flips=(0,)))

    shared = top1_gates(bf16, engine)
    declared = top1_gates(bf16, engine, bf16_max_disagreement_share=1.0)

    assert not shared["bf16_decided_disagreements"]["passed"]
    assert declared["bf16_decided_disagreements"]["passed"]
    assert {
        name: gate for name, gate in shared.items() if name != "bf16_decided_disagreements"
    } == {name: gate for name, gate in declared.items() if name != "bf16_decided_disagreements"}


def test_a_moe_checkpoint_excuses_one_near_tie_prompt_per_path() -> None:
    truth = _truth()
    bf16 = teacher_forced(truth, *_candidate({0: 0.01}))
    at_tie = teacher_forced(truth, *_candidate({3: 3.0}))
    decided = teacher_forced(truth, *_candidate({2: 3.0}))
    clean = teacher_forced(truth, *_candidate())

    (gates,) = path_failures([_comparison("a", bf16, at_tie)], "cold", moe=False).values()
    assert "top1_logprob_error_max" in gates and "decided_disagreements" not in gates
    excused = [_comparison("a", bf16, at_tie), _comparison("b", bf16, clean)]
    assert path_failures(excused, "cold", moe=True) == {}
    assert excused[0]["paths"]["cold"]["excused"]
    assert set(path_failures([_comparison("a", bf16, decided)], "cold", moe=True)) == {"a"}
    two = [_comparison("a", bf16, at_tie), _comparison("b", bf16, at_tie)]
    assert set(path_failures(two, "cold", moe=True)) == {"a", "b"}


def test_the_error_is_reported_apart_before_and_after_the_truths_first_eos() -> None:
    truth = _truth()
    engine = teacher_forced(truth, *_candidate({1: 0.05, 4: 0.5}, flips=(5,)), eos_row=2)

    assert engine["first_eos_row"] == 2
    before, after = engine["before_eos"], engine["after_eos"]
    assert (before["rows"], after["rows"]) == (3, 3)
    assert before["top1_logprob_error_max"] == pytest.approx(0.05)
    assert after["top1_logprob_error_max"] == pytest.approx(0.5)
    assert (before["decided_disagreements"], after["decided_disagreements"]) == (0, 1)


def test_rows_the_schedulings_disagree_on_are_reported_and_not_counted() -> None:
    truth = _truth()
    cold, _ = _candidate(flips=(0, 1))
    chunked, _ = _candidate(flips=(1,))
    stable = ~unstable_rows([cold, chunked])
    bf16 = teacher_forced(truth, *_candidate({0: 0.01}), stable=stable)
    engine = teacher_forced(truth, *_candidate(flips=(0, 1)), stable=stable)

    gates = top1_gates(bf16, engine)

    assert stable.tolist() == [False, True, True, True, True, True]
    # Row 0 flips between schedulings: reported, not one of the engine's decided rows.
    assert (engine["unstable"], engine["decided"], engine["decided_disagreements"]) == (1, 4, 1)
    assert gates["unstable_share"]["value"] == pytest.approx(1 / 6)


def test_an_unstable_rows_error_is_not_counted() -> None:
    truth = _truth()
    stable = torch.tensor([False, True, True, True, True, True])

    engine = teacher_forced(truth, *_candidate({0: 3.0, 1: 0.05}), eos_row=2, stable=stable)

    # Row 0 agrees with a large error, but the schedulings disagree on it.
    assert engine["top1_logprob_error"]["max"] == pytest.approx(0.05)
    assert engine["before_eos"]["top1_logprob_error_max"] == pytest.approx(0.05)
    assert engine["worst_position"] == 1
