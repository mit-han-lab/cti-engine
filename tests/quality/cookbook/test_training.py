from __future__ import annotations

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import BatchEncoding, PreTrainedTokenizerFast

from cookbook.grpo import _collect_rewarded_sampling, _has_reward_variation, _merge_sampling
from cookbook.sft import _render_data, _rotated
from cti.protocol import SampledSequence, SamplingParams, SamplingResult
from cti.training import SFTDatum


def _sampling_result(*, revision: int, seed: int, text: str) -> SamplingResult:
    sampling = SamplingParams(
        temperature=0.7, max_new_tokens=2, seed=seed, logprobs=True, logprobs_processed=True
    ).snapshot()
    sample = SampledSequence(
        request_id=0,
        prompt_token_ids=(1, 2),
        sampled_token_ids=(3, 4),
        sampling_logprobs=(-0.1, -0.2),
        cached_prompt_tokens=0,
        sampling=sampling,
        finish_reason="length",
    )
    return SamplingResult((sample,), (text,), revision)


class _SamplingStub:
    def __init__(self) -> None:
        self.seeds: list[tuple[int | None, ...]] = []

    def sample(
        self,
        prompts: list[list[int]],
        sampling_params: list[SamplingParams],
    ) -> SamplingResult:
        self.seeds.append(tuple(params.seed for params in sampling_params))
        outputs = ((97,), (97,)) if len(self.seeds) == 1 else ((1,), (2,))
        samples = tuple(
            SampledSequence(
                request_id=request_id,
                prompt_token_ids=tuple(prompt),
                sampled_token_ids=outputs[request_id],
                sampling_logprobs=(-0.1,),
                cached_prompt_tokens=0,
                sampling=params.snapshot(),
                finish_reason="length",
            )
            for request_id, (prompt, params) in enumerate(
                zip(prompts, sampling_params, strict=True)
            )
        )
        return SamplingResult(samples, ("", ""), 3)


def test_merge_sampling_preserves_provenance_and_reindexes_requests() -> None:
    merged = _merge_sampling(
        (
            _sampling_result(revision=3, seed=11, text="first"),
            _sampling_result(revision=3, seed=12, text="second"),
        )
    )

    assert tuple(sample.request_id for sample in merged.samples) == (0, 1)
    assert tuple(sample.sampling.seed for sample in merged.samples) == (11, 12)
    assert merged.texts == ("first", "second")
    assert merged.model_revision == 3


def test_merge_sampling_rejects_mixed_revisions() -> None:
    with pytest.raises(ValueError, match="does not match policy revision"):
        _merge_sampling(
            (
                _sampling_result(revision=3, seed=11, text="first"),
                _sampling_result(revision=4, seed=12, text="second"),
            )
        )


def test_reward_filter_uses_bounded_near_zero_policy() -> None:
    assert not _has_reward_variation((1.0, 1.0 + 1e-7))
    assert _has_reward_variation((1.0, 1.1))


def test_reward_collection_retries_with_deterministic_request_seeds() -> None:
    llm = _SamplingStub()

    sampling, rewards, rejected_groups = _collect_rewarded_sampling(
        llm,  # type: ignore[arg-type]
        ((1, 2),),
        revision=3,
        accumulation_index=0,
        accumulation_steps=2,
        groups_per_batch=1,
        group_size=2,
        max_group_attempts=2,
        max_new_tokens=1,
        temperature=0.7,
    )

    assert rejected_groups == 1
    assert rewards == (1.0, 2.0)
    assert sampling.model_revision == 3
    assert llm.seeds == [(25, 26), (27, 28)]


def test_sft_rotation_is_deterministic_and_wraps() -> None:
    data = tuple(SFTDatum((0, token), (False, True)) for token in range(1, 4))

    assert _rotated(data, 0) == data
    assert _rotated(data, 1) == data[1:] + data[:1]
    assert _rotated(data, 4) == data[1:] + data[:1]


def test_sft_rendering_extracts_token_ids_from_transformers_five_chat_results() -> None:
    vocabulary = {
        "<unk>": 0,
        "user": 1,
        "assistant": 2,
        "Four.": 3,
        "Blue.": 4,
        "Jupiter.": 5,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }} {{ message['content'] }} "
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant{% endif %}"
    )
    default_result = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Question"}],
        tokenize=True,
        add_generation_prompt=True,
    )
    assert isinstance(default_result, BatchEncoding)

    data = _render_data(tokenizer)

    assert len(data) == 4
    for datum in data:
        assert all(type(token_id) is int for token_id in datum.token_ids)
        assert datum.target_mask[-1] is True
        assert sum(datum.target_mask) == 1
