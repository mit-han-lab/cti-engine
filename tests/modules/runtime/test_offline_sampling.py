from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cti.protocol import (
    AbortBackendMsg,
    AbortMsg,
    DetokenizeMsg,
    RequestAbortedMsg,
    RequestEnd,
    RequestRejectedMsg,
    SamplingParams,
    TokenUsage,
    UserMsg,
)
from cti.runtime._sampler import _Command
from cti.tokenizer.text import TextStream

_POLICY = SamplingParams(temperature=0, max_new_tokens=1).snapshot()


class _Decoder:
    """Decodes token ``t`` as the letter ``t`` letters past ``a``."""

    prefix_stable = True

    def stream(self, *, skip_special_tokens: bool) -> TextStream:
        return TextStream(self, skip_special_tokens=skip_special_tokens)

    def renders(self, token_id: int, *, skip_special_tokens: bool) -> bool:
        return True

    def batch_decode(self, batches: list[list[int]], *, skip_special_tokens: bool) -> list[str]:
        return ["".join(chr(ord("a") + t) for t in batch) for batch in batches]


def _ended(*revisions: int) -> _Command:
    """A two-request command whose requests ended at ``revisions``."""

    command = _Command(
        0, [UserMsg(uid, torch.tensor([1], dtype=torch.int32), _POLICY) for uid in (0, 1)]
    )
    for uid, revision in enumerate(revisions):
        end = RequestEnd(TokenUsage(1, 1), 1, model_revision=revision)
        command.receive(DetokenizeMsg(uid=uid, token_ids=(uid,), end=end))
    return command


def test_a_sample_names_the_output_limit_admission_applied() -> None:
    """Up to the end of the context resolves at admission; the sample's policy is exact
    evidence, so it carries the limit the request ran under, not None."""
    policy = SamplingParams(temperature=0, max_new_tokens=None).snapshot()
    command = _Command(5, [UserMsg(5, torch.tensor([1, 2, 3], dtype=torch.int32), policy)])

    command.receive(
        DetokenizeMsg(
            uid=5,
            token_ids=(7,),
            end=RequestEnd(TokenUsage(prompt_tokens=3, completion_tokens=1), 29),
        )
    )

    assert command.records[5].sample(0).sampling.max_new_tokens == 29


def test_a_command_reports_the_revision_its_requests_ran_at() -> None:
    result = _ended(3, 3).result(SimpleNamespace(decoder=_Decoder()))

    assert result.model_revision == 3 and result.texts == ("a", "b")


def test_an_optimizer_step_between_a_commands_requests_fails_it() -> None:
    with pytest.raises(RuntimeError, match="the model changed while sampling; resample"):
        _ended(3, 4).result(SimpleNamespace(decoder=_Decoder()))


def test_a_rejection_aborts_the_open_requests_and_fails_the_command_once_released() -> None:
    command = _Command(
        4, [UserMsg(uid, torch.tensor([1], dtype=torch.int32), _POLICY) for uid in (4, 5, 6)]
    )

    command.receive(RequestRejectedMsg(uid=5, reason="ingress hold is full"))
    assert command.aborts == [AbortBackendMsg(uid=4), AbortBackendMsg(uid=6)]
    assert command.open == {4, 6}
    for uid in (4, 6):
        command.receive(RequestAbortedMsg(uid=uid))

    assert command.open == set()
    assert command.rejection is not None and command.rejection.request_id == 1


def test_a_message_that_is_no_reply_is_refused() -> None:
    with pytest.raises(TypeError, match="AbortMsg is not a reply to a request"):
        _ended().receive(AbortMsg(uid=0))
