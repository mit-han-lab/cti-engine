"""A sample's evidence is assembled by one rule, whichever replies carry it: the offline
sampler's ``DetokenizeMsg`` and the online server's ``UserReply`` give the same sample."""

from __future__ import annotations

import pytest

from cti.protocol import (
    DetokenizeMsg,
    RequestEnd,
    SampledSequence,
    SamplingParams,
    TokenUsage,
    UserReply,
)
from cti.sampling import TokenLogprobs

_PROMPT = (11, 12, 13)
_USAGE = TokenUsage(prompt_tokens=3, completion_tokens=3, cached_prompt_tokens=2)
_TOP = ((7, -0.1), (8, -2.0))
_LOGPROBS = tuple(TokenLogprobs(t, -0.5 * i, _TOP) for i, t in enumerate((7, 8, 9)))
_PROMPT_LOGPROBS = (None, TokenLogprobs(12, -1.0), TokenLogprobs(13, -1.5))


def _spec(**fields: object):
    return SamplingParams(max_new_tokens=None, **fields).snapshot()


def _offline(spec, *, stop_str: str | None = None, logprobs: bool = True) -> SampledSequence:
    """The sample from the engine's replies: a step of two tokens, then the last."""

    asked = spec.law.logprobs is not None and logprobs
    msgs = [
        DetokenizeMsg(
            uid=4,
            token_ids=(7, 8),
            logprobs=_LOGPROBS[:2] if asked else (),
            prompt_logprobs=_PROMPT_LOGPROBS if spec.law.prompt_logprobs else None,
        ),
        DetokenizeMsg(
            uid=4,
            token_ids=(9,),
            logprobs=_LOGPROBS[2:] if asked else (),
            end=RequestEnd(_USAGE, 16, stop_str=stop_str),
        ),
    ]
    return SampledSequence.assemble(
        0,
        _PROMPT,
        spec,
        token_ids=[t for msg in msgs for t in msg.token_ids],
        logprobs=[entry for msg in msgs for entry in msg.logprobs],
        prompt_logprobs=msgs[0].prompt_logprobs,
        ending=msgs[-1].end,
    )


def _online(spec, *, stop_str: str | None = None) -> SampledSequence:
    """The sample from the frontend's replies: one per engine step, as the engine's."""

    asked = spec.law.logprobs is not None
    replies = [
        UserReply(
            uid=4,
            incremental_output="x",
            finished=False,
            token_ids=(7, 8),
            logprobs=_LOGPROBS[:2] if asked else (),
            prompt_logprobs=_PROMPT_LOGPROBS if spec.law.prompt_logprobs else None,
        ),
        UserReply(
            uid=4,
            incremental_output="x",
            finished=True,
            usage=_USAGE,
            finish_reason="stop" if stop_str else "length",
            token_ids=(9,),
            stop_str=stop_str,
            logprobs=_LOGPROBS[2:] if asked else (),
            max_new_tokens=16,
        ),
    ]
    return SampledSequence.assemble(
        0,
        _PROMPT,
        spec,
        token_ids=[t for reply in replies for t in reply.token_ids],
        logprobs=[entry for reply in replies for entry in reply.logprobs],
        prompt_logprobs=replies[0].prompt_logprobs,
        ending=replies[-1],
    )


@pytest.mark.parametrize(
    ("fields", "stop_str"),
    (
        ({}, None),
        ({"logprobs": True, "top_logprobs": 2, "prompt_logprobs": 1}, None),
        ({"stop": ["ab"]}, "ab"),
    ),
    ids=("plain", "logprobs", "stop-string"),
)
def test_offline_and_online_replies_assemble_the_same_sample(fields: dict, stop_str) -> None:
    spec = _spec(**fields)

    offline, online = _offline(spec, stop_str=stop_str), _online(spec, stop_str=stop_str)

    assert offline == online
    # Up to the end of the context resolves to the limit admission applied.
    assert offline.sampling.max_new_tokens == 16
    assert offline.cached_prompt_tokens == 2
    if spec.law.logprobs is not None:
        assert offline.sampling_logprobs == (0.0, -0.5, -1.0)
        assert offline.sampling_top_logprobs == (_TOP,) * 3


def test_a_sample_whose_logprobs_were_asked_must_carry_each_tokens() -> None:
    with pytest.raises(RuntimeError, match="request 0 has no selected-token log-probability"):
        _offline(_spec(logprobs=True), logprobs=False)


def test_the_visible_text_ends_where_the_stop_string_begins() -> None:
    stopped = _offline(_spec(stop=["ab"]), stop_str="ab")

    assert stopped.visible_text("xyzabq") == "xyz"
    assert stopped.visible_text("xyz") == "xyz"
    assert _offline(_spec()).visible_text("xyzabq") == "xyzabq"
