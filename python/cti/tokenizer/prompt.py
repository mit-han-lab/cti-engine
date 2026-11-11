"""The one owner of prompt encoding: text, chats and token ids into engine prompts."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from jinja2 import TemplateError

from cti.multimodal import ProcessedPrompt, create_input_processor
from cti.protocol import (
    AdmissionPriority,
    ChatPrompt,
    TextPrompt,
    TokenPrompt,
    UserMsg,
)

from .text import load_text_decoder

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizerBase

    from cti.multimodal import ImageLimits, InputProcessor, RopePositionLayout
    from cti.protocol import Prompt, SamplingSpec

    from .text import TokenizerTextDecoder


class PromptEncoder:
    """Encode prompts the same way online and offline; an invalid prompt raises
    ``ValueError``. ``processor``, None for a text-only model, expands images, and
    ``check`` is the one validator of a prompt's images, through it; token ids must lie in
    the model's ``vocab_size``, which may exceed the tokenizer's. ``decoder`` is the
    process's one text decoder, over ``tokenizer``."""

    def __init__(
        self, decoder: TokenizerTextDecoder, processor: InputProcessor | None, vocab_size: int
    ) -> None:
        self.decoder = decoder
        self._processor = processor
        self._vocab_size = vocab_size

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str,
        *,
        hf_config: PretrainedConfig,
        limits: ImageLimits,
    ) -> PromptEncoder:
        architecture = hf_config.architectures[0]
        processor = create_input_processor(
            architecture, model_path, hf_config=hf_config, limits=limits
        )
        return cls(load_text_decoder(model_path), processor, hf_config.get_text_config().vocab_size)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The checkpoint's tokenizer, with its chat template."""

        return self.decoder.tokenizer

    def request(
        self,
        uid: int,
        prompt: Prompt | ProcessedPrompt | str | Sequence[int],
        sampling: SamplingSpec,
        priority: AdmissionPriority = AdmissionPriority.NORMAL,
    ) -> UserMsg:
        """The engine request of one prompt, online or offline (text, token ids or a
        processed prompt) under its validated ``sampling``, its images checked."""

        match prompt:
            case TextPrompt(text=text) | (str() as text):
                prompt = self.text(text)
            case ChatPrompt(messages=messages, tools=tools, template_kwargs=template_kwargs):
                prompt = self.chat(messages, tools=tools, template_kwargs=template_kwargs)
            case TokenPrompt(token_ids=token_ids) | ((list() | tuple()) as token_ids):
                if not token_ids:
                    raise ValueError("token prompts must not be empty")
                prompt = self.tokens(token_ids)
        rope_layout = self.check(prompt)
        return UserMsg(uid, prompt.input_ids, sampling, priority, prompt.images, rope_layout)

    def text(self, text: str, images: Sequence[object] = ()) -> ProcessedPrompt:
        return self.tokens(self.tokenizer.encode(text), images)

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] = (),
        template_kwargs: Mapping[str, Any] | None = None,
    ) -> ProcessedPrompt:
        """``template_kwargs`` are the chat template's own variables, such as
        ``enable_thinking``; the template decides what an unknown one means."""

        render = self.tokenizer.apply_chat_template
        template_kwargs = dict(template_kwargs or {})
        if reserved := sorted(template_kwargs.keys() & _parameters(render)):
            raise ValueError(f"chat template kwargs cannot set {', '.join(reserved)}")
        if self._processor is None and any(map(_has_media, messages)):
            raise ValueError("the model takes no images")
        images = [] if self._processor is None else self._processor.chat_images(messages)
        if tools:
            template_kwargs["tools"] = list(tools)
        try:
            text = render(
                list(messages), tokenize=False, add_generation_prompt=True, **template_kwargs
            )
        except (TemplateError, TypeError) as error:
            # Templates reject malformed chats through raise_exception, and fail with
            # TypeError on fields of the wrong type, such as the OpenAI API's JSON-string
            # tool-call arguments where Qwen3.5's template iterates a mapping.
            raise ValueError(f"cannot render chat: {error}") from error
        # The rendered template already holds the model's control tokens.
        return self.tokens(self.tokenizer.encode(text, add_special_tokens=False), images)

    def tokens(self, token_ids: Sequence[int], images: Sequence[object] = ()) -> ProcessedPrompt:
        vocab_size = self._vocab_size
        if stray := [token_id for token_id in token_ids if not 0 <= token_id < vocab_size]:
            raise ValueError(f"token id {stray[0]} is outside the vocabulary [0, {vocab_size})")
        if self._processor is not None:
            return self._processor.expand(token_ids, images)
        if images:
            raise ValueError("the model takes no images")
        return ProcessedPrompt(torch.tensor(token_ids, dtype=torch.int32), None, None)

    def check(self, prompt: ProcessedPrompt) -> RopePositionLayout | None:
        """The layout of ``prompt``'s images, once its placeholders are checked against them."""

        if prompt.input_ids.device.type != "cpu" or prompt.input_ids.dtype != torch.int32:
            raise ValueError("prompt token ids must be a CPU int32 tensor")
        if self._processor is None and prompt.images is not None:
            raise ValueError("the model takes no images")
        processor = self._processor
        rope_layout = processor and processor.validate_token_ids(prompt.input_ids, prompt.images)
        if rope_layout != prompt.rope_layout:
            raise ValueError("the prompt's rope layout does not match its images")
        return rope_layout

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool) -> str:
        return self.decoder.batch_decode(
            [list(token_ids)], skip_special_tokens=skip_special_tokens
        )[0]


def _has_media(message: Mapping[str, Any]) -> bool:
    """Whether a chat message holds a content part other than text."""

    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, str) or not isinstance(content, Sequence):
        return False
    return any(isinstance(part, Mapping) and part.get("type") != "text" for part in content)


def _parameters(render: Callable[..., object]) -> set[str]:
    """The chat renderer's own parameters, which are not template variables."""

    parameters = inspect.signature(render).parameters.values()
    return {p.name for p in parameters if p.kind is not inspect.Parameter.VAR_KEYWORD}
