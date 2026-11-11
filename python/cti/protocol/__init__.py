# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/message/__init__.py`); see THIRD_PARTY_NOTICES.

from typing import TYPE_CHECKING

from ._endpoints import ServingEndpoints
from ._frontend import BaseFrontendMsg, PromptReply, UserReply
from ._generation import GenerationOutput, SampledSequence, SamplingPrompt, SamplingResult
from ._request import (
    AdmissionPriority,
    RejectionKind,
    RequestRejectedError,
    SamplingParams,
    SamplingSpec,
)
from ._tokenizer import (
    AbortMsg,
    BaseTokenizerMsg,
    ChatImagePart,
    ChatPrompt,
    ChatTextPart,
    DetokenizeMsg,
    InferenceReply,
    Prompt,
    RequestAbortedMsg,
    RequestEnd,
    RequestRejectedMsg,
    TextPrompt,
    TokenizeMsg,
    TokenPrompt,
    context_length_exceeded,
)
from ._usage import TokenUsage

if TYPE_CHECKING:
    from ._backend import AbortBackendMsg as AbortBackendMsg
    from ._backend import BaseBackendMsg as BaseBackendMsg
    from ._backend import UserMsg as UserMsg

# The rank-bound messages carry tensors: importing the protocol does not load torch.
_BACKEND_MESSAGES = ("AbortBackendMsg", "BaseBackendMsg", "UserMsg")


def __getattr__(name: str) -> object:
    if name in _BACKEND_MESSAGES:
        from . import _backend

        return getattr(_backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GenerationOutput",
    "SampledSequence",
    "SamplingPrompt",
    "SamplingResult",
    "AdmissionPriority",
    "RequestRejectedError",
    "SamplingParams",
    "SamplingSpec",
    "AbortMsg",
    "BaseTokenizerMsg",
    "ChatImagePart",
    "ChatPrompt",
    "ChatTextPart",
    "DetokenizeMsg",
    "InferenceReply",
    "Prompt",
    "RejectionKind",
    "RequestAbortedMsg",
    "RequestEnd",
    "RequestRejectedMsg",
    "TextPrompt",
    "TokenizeMsg",
    "TokenPrompt",
    "context_length_exceeded",
    "BaseFrontendMsg",
    "PromptReply",
    "UserReply",
    "ServingEndpoints",
    "TokenUsage",
] + list(_BACKEND_MESSAGES)
