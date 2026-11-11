# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/llm/llm.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, List

from cti.config import EngineOptions, EngineTuning, LaunchFile
from cti.protocol import GenerationOutput, SamplingParams, SamplingPrompt, SamplingResult

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from cti.runtime import TrainingSession
    from cti.tokenizer import PromptEncoder


class LLM:
    """The offline facade: bounded sampling, a local training session and the model revision."""

    # Chats preprocessed at once: image decoding and processing release the GIL.
    CHAT_WORKERS = 4

    def __init__(
        self,
        options: EngineOptions,
        *,
        tuning: EngineTuning | None = None,
        resume_from_checkpoint: str | os.PathLike[str] | None = None,
    ) -> None:
        """``tuning`` is internal, for tests and benchmarks; None keeps its defaults."""

        checkpoint_path = None
        if resume_from_checkpoint is not None:
            if options.training is None:
                raise ValueError("resume_from_checkpoint requires a training section")
            raw_path = os.fspath(resume_from_checkpoint)
            if type(raw_path) is not str:
                raise TypeError("resume_from_checkpoint must resolve to a string")
            if not raw_path:
                raise ValueError("resume_from_checkpoint must not be empty")
            checkpoint_path = os.path.realpath(os.path.abspath(os.path.expanduser(raw_path)))

        # Imported here so that `import cti.llm` stays light.
        from cti.runtime import EngineClient

        self._client = EngineClient(options, tuning, checkpoint=checkpoint_path)
        self._finalizer = self._client.close_at_exit(self)
        self._chats = ThreadPoolExecutor(self.CHAT_WORKERS, thread_name_prefix="cti-chat")

    @classmethod
    def from_yaml(
        cls,
        path: str | os.PathLike[str],
        *,
        resume_from_checkpoint: str | os.PathLike[str] | None = None,
    ) -> LLM:
        """Construct from a launch file; its ``server`` section is checked and unused."""

        return cls(
            LaunchFile.load(path).engine,
            resume_from_checkpoint=resume_from_checkpoint,
        )

    def shutdown(self) -> None:
        self._chats.shutdown(cancel_futures=True)
        self._client.close()
        self._finalizer.cancel()

    @property
    def model_revision(self) -> int:
        return self._client.model_revision()

    def prompt_encoder(self) -> PromptEncoder:
        """The engine's prompt encoder, loaded here once; its ``chat`` builds image prompts."""

        return self._client.prompt_encoder()

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The tokenizer the engine encodes prompts with, its chat template included; loaded
        with the prompt encoder."""

        return self._client.prompt_encoder().tokenizer

    def training_session(self) -> TrainingSession:
        return self._client.training_session(self)

    def generate(
        self,
        prompts: List[SamplingPrompt],
        sampling_params: List[SamplingParams] | SamplingParams,
    ) -> List[GenerationOutput]:
        return self.sample(prompts, sampling_params).generation_outputs()

    def sample(
        self,
        prompts: List[SamplingPrompt],
        sampling_params: List[SamplingParams] | SamplingParams,
    ) -> SamplingResult:
        """Return exact bounded samples for serving or downstream training.

        A prompt is text, its token ids, or a ``ProcessedPrompt`` from
        ``prompt_encoder()``, which is how a prompt carries images.
        """

        return self._client.sample(prompts, SamplingParams.specs(prompts, sampling_params))

    def chat(
        self,
        conversations: Sequence[Sequence[Mapping[str, object]]],
        sampling_params: List[SamplingParams] | SamplingParams,
        *,
        enable_thinking: bool | None = None,
    ) -> SamplingResult:
        """Sample each chat in ``conversations``, its images included, as ``sample`` does
        with the prompts ``prompt_encoder().chat`` builds; the chats are preprocessed
        concurrently. ``enable_thinking`` None keeps the chat template's default."""

        # Validated before any chat is preprocessed.
        policies = SamplingParams.specs(conversations, sampling_params)
        template_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        render = partial(self._client.prompt_encoder().chat, template_kwargs=template_kwargs)
        return self._client.sample(list(self._chats.map(render, conversations)), policies)
