"""A batch's sampling policies: one params for every prompt or one each, checked before anything
is sent; offline, chat and online sampling all take them this way."""

from __future__ import annotations

import pytest

from cti.protocol import SamplingParams

_GREEDY = SamplingParams(temperature=0, max_new_tokens=4)
_HOT = SamplingParams(temperature=0.7, max_new_tokens=8)


def test_one_params_samples_every_prompt_and_a_list_one_each() -> None:
    assert SamplingParams.specs(["a", "b"], _GREEDY) == [_GREEDY.snapshot()] * 2
    assert SamplingParams.specs(["a", "b"], [_GREEDY, _HOT]) == [
        _GREEDY.snapshot(),
        _HOT.snapshot(),
    ]


@pytest.mark.parametrize(
    ("prompts", "params", "error", "match"),
    [
        (["a", "b"], [_GREEDY], ValueError, "same length"),
        ([], _GREEDY, ValueError, "must not be empty"),
        (["a"], [{"temperature": 0}], TypeError, "SamplingParams values"),
    ],
    ids=["length", "empty", "type"],
)
def test_a_batch_that_does_not_match_its_prompts_is_refused(
    prompts: list[str], params: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        SamplingParams.specs(prompts, params)
