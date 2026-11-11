"""The OpenAI request schema of the API server: request models, their sampling, and errors.

Field names and defaults follow OpenAI, and null means a field's default. Extensions use
vLLM's names (``top_k``, ``min_p``, ``repetition_penalty``, ``ignore_eos``,
``stop_token_ids``, ``include_stop_str_in_output``, ``skip_special_tokens``,
``return_token_ids``, ``min_tokens``). Unknown fields are rejected; a known OpenAI or
vLLM field the engine cannot honor stays out of the schema and is rejected by name with
code ``unsupported_parameter``, unless its value asks for nothing.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self

from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from cti.protocol import (
    AdmissionPriority,
    ChatImagePart,
    ChatPrompt,
    ChatTextPart,
    Prompt,
    SamplingParams,
    TextPrompt,
    TokenPrompt,
)
from cti.sampling import SamplingError

if TYPE_CHECKING:
    from cti.protocol import SamplingSpec

TokenId = Annotated[StrictInt, Field(ge=0)]


class OpenAIError(Exception):
    """An error answered with an OpenAI error object."""

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        code: str | None = None,
        status_code: int = 400,
    ) -> None:
        """A 4xx is an ``invalid_request_error``, anything else a ``server_error``."""

        super().__init__(message)
        self.message = message
        self.param = param
        self.code = code
        self.status_code = status_code

    @classmethod
    def from_validation(cls, errors: Sequence[Any]) -> Self:
        """The first of a request body's validation errors; ``param`` is the full path of
        the field at fault, as in ``messages[0].role``."""

        error = errors[0]
        if error["type"] == "json_invalid":  # no field to name
            return cls(f"Malformed JSON body: {error['ctx']['error']}")
        param = _param_path(error["loc"], error.get("ctx", {}).get("param"))
        if error["type"] == "extra_forbidden":
            return cls(f"Unrecognized request argument supplied: {param}", param=param)
        if error["type"] == "missing":
            return cls(error["msg"], param=param)
        code = "unsupported_parameter" if error["type"] == "unsupported_parameter" else None
        return cls(error["msg"], param=param, code=code or "invalid_value")

    def response(self) -> JSONResponse:
        error = {
            "message": self.message,
            "type": "invalid_request_error" if self.status_code < 500 else "server_error",
            "param": self.param,
            "code": self.code,
        }
        return JSONResponse({"error": error}, status_code=self.status_code)


def _param_path(location: Sequence[str | int], tail: str | None = None) -> str | None:
    """``location`` as a dotted path, then ``tail``, a path below it like ``[1].type``."""

    path = ""
    for part in location[1:] if location and location[0] == "body" else location:
        path += f"[{part}]" if isinstance(part, int) else f".{part}" if path else str(part)
    if tail:
        path += tail if tail.startswith("[") or not path else f".{tail}"
    return path or None


def _invalid(message: str) -> PydanticCustomError:
    return PydanticCustomError("invalid_value", message)


class _OpenAIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _null_is_default(cls, body: object) -> object:
        """OpenAI lets a client send null for a field to mean its default."""

        if not isinstance(body, dict):
            return body
        return {
            name: value for name, value in body.items() if value is not None or cls._nullable(name)
        }

    @classmethod
    def _nullable(cls, name: str) -> bool:
        field = cls.model_fields.get(name)
        return field is None or field.default is None


class CalledFunction(_OpenAIModel):
    name: str
    arguments: str  # a JSON object

    @field_validator("arguments")
    @classmethod
    def _object(cls, arguments: str) -> str:
        try:
            value = json.loads(arguments)
        except ValueError:
            value = None
        if not isinstance(value, dict):
            raise _invalid("arguments must be a JSON object")
        return arguments


class ToolCall(_OpenAIModel):
    """A call an assistant message made, as a chat completion returned it."""

    id: str
    type: Literal["function"] = "function"
    function: CalledFunction


class ChatMessage(_OpenAIModel):
    """A message of text, or of text and, from the user, image parts; ``name`` reaches the
    chat template with it. An assistant's may hold only ``tool_calls``; a tool's answers
    one by ``tool_call_id``."""

    _PARTS: ClassVar[Mapping[str, TypeAdapter]] = MappingProxyType(
        {"text": TypeAdapter(ChatTextPart), "image_url": TypeAdapter(ChatImagePart)}
    )

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ChatTextPart | ChatImagePart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    @field_validator(
        "content",
        mode="plain",
        json_schema_input_type=str | list[ChatTextPart | ChatImagePart] | None,
    )
    @classmethod
    def _content(cls, content: object, info: ValidationInfo) -> object:
        """Each part is validated as its ``type`` names, so errors name its path."""

        if content is None or isinstance(content, str):
            return content
        if not isinstance(content, list) or not content:
            raise _invalid("content must be a string or a non-empty list of content parts")
        for index, part in enumerate(content):
            kind = part.get("type") if isinstance(part, dict) else None
            if kind not in cls._PARTS:
                message = f"content part type {kind!r} is not supported; use text or image_url"
                raise PydanticCustomError("invalid_value", message, {"param": f"[{index}].type"})
            if kind == "image_url" and info.data.get("role", "user") != "user":
                message = "only user messages may carry images"
                raise PydanticCustomError("invalid_value", message, {"param": f"[{index}]"})
            try:
                cls._PARTS[kind].validate_python(part)
            except ValidationError as invalid:
                error = invalid.errors()[0]
                param = f"[{index}].{_param_path(error['loc'])}"
                raise PydanticCustomError(error["type"], error["msg"], {"param": param}) from None
        return content

    @model_validator(mode="after")
    def _fields_of_role(self) -> Self:
        if self.tool_calls is not None and self.role != "assistant":
            raise PydanticCustomError(
                "invalid_value",
                "only assistant messages may carry tool_calls",
                {"param": "tool_calls"},
            )
        if (self.tool_call_id is None) != (self.role != "tool"):
            raise PydanticCustomError(
                "invalid_value",
                "tool messages, and only they, carry tool_call_id",
                {"param": "tool_call_id"},
            )
        if self.content is None and not self.tool_calls:
            raise PydanticCustomError(
                "missing",
                "content is required unless the message carries tool_calls",
                {"param": "content"},
            )
        return self

    def turn(self) -> dict[str, Any]:
        """The message as the chat template takes it: absent content is empty, and each
        call's arguments a mapping, as templates iterate them."""

        turn = self.model_dump(exclude_none=True)
        turn.setdefault("content", "")
        for call in turn.get("tool_calls", ()):
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        return turn


class FunctionDefinition(_OpenAIModel):
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict[str, Any] | None = None  # a JSON schema
    strict: bool | None = None  # accepted; calls are parsed, not constrained


class Tool(_OpenAIModel):
    type: Literal["function"]
    function: FunctionDefinition


class _FunctionName(_OpenAIModel):
    name: str


class NamedToolChoice(_OpenAIModel):
    type: Literal["function"]
    function: _FunctionName


class StreamOptions(_OpenAIModel):
    include_usage: bool = False


class GenerationRequest(_OpenAIModel):
    """The fields both generation endpoints share."""

    # Known fields the engine does not honor: the values that ask for nothing, which are
    # accepted, and the reason any other value is rejected.
    _UNSUPPORTED: ClassVar[dict[str, tuple[tuple[object, ...], str]]] = {}

    # Only the protocol's own ranges are checked here; the sampling policy checks its fields
    # (``sampling``). JSON object keys are strings: a logit bias token ID parses from one.
    model: str
    n: StrictInt = Field(1, ge=1)
    temperature: float = Field(1.0, le=2.0)
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    logit_bias: dict[int, float] | None = None
    min_tokens: StrictInt = 0
    seed: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    stop_token_ids: list[StrictInt] | None = None
    include_stop_str_in_output: bool = False
    skip_special_tokens: bool = True
    ignore_eos: bool = False
    return_token_ids: bool = False
    user: str | None = None
    priority: AdmissionPriority = AdmissionPriority.NORMAL

    @model_validator(mode="before")
    @classmethod
    def _reject_unsupported(cls, body: object) -> object:
        if not isinstance(body, dict):
            return body
        for name, value in body.items():
            if value is None or name not in cls._UNSUPPORTED:
                continue
            noops, reason = cls._UNSUPPORTED[name]
            if not any(type(value) is type(noop) and value == noop for noop in noops):
                raise PydanticCustomError("unsupported_parameter", reason, {"param": name})
        return {name: value for name, value in body.items() if name not in cls._UNSUPPORTED}

    @field_validator("top_p")
    @classmethod
    def _top_p(cls, top_p: float) -> float:
        if top_p == 0.0:
            raise PydanticCustomError("unsupported_parameter", "top_p=0 is not supported")
        return top_p

    @field_validator("top_k")
    @classmethod
    def _top_k(cls, top_k: int) -> int:
        # vLLM's -1 keeps the whole vocabulary, the policy's None.
        if top_k != -1 and top_k < 1:
            raise _invalid("top_k must be -1 or a positive integer")
        return top_k

    @field_validator("stop", mode="plain", json_schema_input_type=str | list[str] | None)
    @classmethod
    def _stops(cls, stop: object) -> str | list[str] | None:
        # Unbounded in count and length, like vLLM and SGLang; the policy refuses empty ones.
        if stop is None or isinstance(stop, str):
            return stop
        if isinstance(stop, list) and all(isinstance(sequence, str) for sequence in stop):
            return stop
        raise _invalid("stop must be a string or a list of strings")

    @model_validator(mode="after")
    def _seeds_fit(self) -> Self:
        if self.seed is not None and self.seed + self.n > 2**63:
            raise PydanticCustomError(
                "invalid_value",
                "seed + n - 1 must be below 2**63: choice j samples with seed + j",
                {"param": "seed"},
            )
        return self

    @model_validator(mode="after")
    def _stream_options_need_stream(self) -> Self:
        if self.stream_options is not None and not self.stream:
            raise PydanticCustomError(
                "invalid_value", "stream_options requires stream", {"param": "stream_options"}
            )
        return self

    @property
    def include_usage(self) -> bool:
        return self.stream_options is not None and self.stream_options.include_usage

    @property
    def stop_strings(self) -> tuple[str, ...]:
        return (self.stop,) if isinstance(self.stop, str) else tuple(self.stop or ())

    @property
    def return_prompt(self) -> bool:
        """Whether the response repeats the prompt, which the tokenizer then sends."""

        return self.return_token_ids

    @property
    def prompt_only(self) -> bool:
        """Whether the client asks for no output, only the prompt."""

        return False

    @property
    @abstractmethod
    def output_budget(self) -> int | None:
        """The most output tokens the engine samples; None runs to the end of the context."""

    @property
    @abstractmethod
    def top_logprobs_count(self) -> int | None:
        """How many likeliest alternatives each output token's logprobs list, or None when
        the request asks for no logprobs."""

    @property
    def prompt_logprobs_count(self) -> int | None:
        """How many likeliest alternatives each prompt token's logprobs list, or None when
        the request asks for no prompt logprobs."""

        return None

    @property
    def _top_logprobs(self) -> int:
        """The alternatives the client asked of each output token, logprobs or not."""

        return self.top_logprobs_count or 0

    def _param(self, field: str) -> str:
        """The request field the policy's ``field`` comes from."""

        return "max_tokens" if field == "max_new_tokens" else field

    def sampling(self) -> list[SamplingSpec]:
        """Each of a prompt's ``n`` choices' sampling policy, its logprobs raw; with
        a seed, choice ``j`` samples with ``seed + j``, as vLLM's parallel sampling does. Only
        choice 0 scores the prompt, so its siblings keep reading the prefix cache;
        ``CallRenderer`` gives them its prompt logprobs. A field the policy refuses is a 400
        naming the request field."""

        top_logprobs = self.top_logprobs_count
        # Greedy takes no filters: a temperature-0 request's are dropped, not refused.
        greedy = self.temperature == 0.0
        params = SamplingParams(
            temperature=self.temperature,
            top_k=None if greedy or self.top_k == -1 else self.top_k,
            top_p=1.0 if greedy else self.top_p,
            min_p=self.min_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            ignore_eos=self.ignore_eos,
            max_new_tokens=self.output_budget,
            seed=self.seed,
            logprobs=top_logprobs is not None,
            top_logprobs=self._top_logprobs,
            prompt_logprobs=self.prompt_logprobs_count,
            stop=self.stop_strings,
            stop_token_ids=self.stop_token_ids,
            skip_special_tokens=self.skip_special_tokens,
            logit_bias=self.logit_bias,
            min_tokens=self.min_tokens,
        )
        try:
            return [
                replace(
                    params,
                    seed=None if self.seed is None else self.seed + choice,
                    prompt_logprobs=None if choice else params.prompt_logprobs,
                ).snapshot()
                for choice in range(self.n)
            ]
        except SamplingError as error:
            param = self._param(error.field)
            raise OpenAIError(str(error), param=param, code="invalid_value") from error

    @property
    def offered_tools(self) -> tuple[dict[str, Any], ...]:
        """The tools the model may call: none but in a chat."""

        return ()


class ChatCompletionRequest(GenerationRequest):
    """A chat completion. ``max_completion_tokens`` and the deprecated ``max_tokens`` both
    bound the output; with neither, it runs to the end of the context, as OpenAI and vLLM
    chat do."""

    _UNSUPPORTED: ClassVar[dict[str, tuple[tuple[object, ...], str]]] = {
        **GenerationRequest._UNSUPPORTED,
        "response_format": (({"type": "text"},), "response_format is not supported"),
    }

    messages: list[ChatMessage] = Field(min_length=1)
    max_completion_tokens: int | None = None
    max_tokens: int | None = None
    logprobs: bool = False
    top_logprobs: StrictInt = 0
    # The model is offered the tools tool_choice leaves it, and its calls are parsed out of
    # its text; required and a named tool are not enforced, nor parallel_tool_calls.
    tools: list[Tool] | None = Field(None, min_length=1)
    tool_choice: Literal["none", "auto", "required"] | NamedToolChoice | None = None
    parallel_tool_calls: bool = True
    # The chat template's own variables, such as enable_thinking, as vLLM and SGLang take.
    chat_template_kwargs: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _tool_choice_names_a_tool(self) -> Self:
        names = [tool.function.name for tool in self.tools or ()]
        if len(set(names)) < len(names):
            raise PydanticCustomError(
                "invalid_value", "tools must have distinct names", {"param": "tools"}
            )
        choice = self.tool_choice
        if not names and choice not in (None, "none", "auto"):
            raise PydanticCustomError(
                "invalid_value", "tool_choice needs tools", {"param": "tool_choice"}
            )
        if isinstance(choice, NamedToolChoice) and choice.function.name not in names:
            raise PydanticCustomError(
                "invalid_value",
                f"tool_choice names {choice.function.name!r}, which is not among tools",
                {"param": "tool_choice.function.name"},
            )
        return self

    @model_validator(mode="after")
    def _one_output_budget(self) -> Self:
        if self.max_completion_tokens is not None and self.max_tokens is not None:
            raise PydanticCustomError(
                "invalid_value",
                "set max_completion_tokens or the deprecated max_tokens, not both",
                {"param": "max_tokens"},
            )
        return self

    @property
    def output_budget(self) -> int | None:
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        return self.max_tokens

    @property
    def top_logprobs_count(self) -> int | None:
        return self.top_logprobs if self.logprobs else None

    @property
    def _top_logprobs(self) -> int:
        return self.top_logprobs

    def _param(self, field: str) -> str:
        if field == "max_new_tokens" and self.max_completion_tokens is not None:
            return "max_completion_tokens"
        return super()._param(field)

    @property
    def offered_tools(self) -> tuple[dict[str, Any], ...]:
        """The tools the chat template offers: none with ``tool_choice`` none, the named
        one when it names one, else all."""

        choice = self.tool_choice
        if choice == "none":
            return ()
        return tuple(
            tool.model_dump(exclude_none=True)
            for tool in self.tools or ()
            if not isinstance(choice, NamedToolChoice) or tool.function.name == choice.function.name
        )

    def prompts(self) -> list[Prompt]:
        turns = tuple(message.turn() for message in self.messages)
        return [ChatPrompt(turns, self.offered_tools, self.chat_template_kwargs or {})]


def _is_token_ids(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(type(token_id) is int and token_id >= 0 for token_id in value)
    )


_PROMPT_SCHEMA = str | list[str] | list[TokenId] | list[list[TokenId]]


class CompletionRequest(GenerationRequest):
    _UNSUPPORTED: ClassVar[dict[str, tuple[tuple[object, ...], str]]] = {
        **GenerationRequest._UNSUPPORTED,
        "suffix": (("",), "suffix is not supported"),
        "best_of": ((1,), "best_of is not supported"),
    }

    prompt: str | list[str] | list[int] | list[list[int]]
    max_tokens: int = Field(16, ge=0)
    echo: bool = False
    # Each output token's logprob and its likeliest ``logprobs`` alternatives; with
    # ``echo``, each prompt token's too.
    logprobs: StrictInt | None = None

    @field_validator("prompt", mode="plain", json_schema_input_type=_PROMPT_SCHEMA)
    @classmethod
    def _prompt_shape(cls, prompt: object) -> object:
        """OpenAI's and vLLM's shapes: a batch is all strings or all token-id lists."""

        if (
            isinstance(prompt, str)
            or _is_token_ids(prompt)
            or (
                isinstance(prompt, list)
                and len(prompt) > 0
                and (
                    all(isinstance(entry, str) for entry in prompt)
                    or all(_is_token_ids(entry) for entry in prompt)
                )
            )
        ):
            return prompt
        raise PydanticCustomError(
            "invalid_value",
            "prompt must be a string, a non-empty list of strings, a non-empty list of token "
            "ids, or a non-empty list of token-id lists, never a mix",
        )

    @property
    def output_budget(self) -> int:
        # A prompt-only request (max_tokens 0, OpenAI's way to echo or score a prompt) still
        # samples one token, which gives it the prompt's usage, and drops it.
        return self.max_tokens or 1

    @property
    def return_prompt(self) -> bool:
        return self.echo or self.return_token_ids

    @property
    def prompt_only(self) -> bool:
        return self.max_tokens == 0

    @property
    def top_logprobs_count(self) -> int | None:
        return self.logprobs

    @property
    def prompt_logprobs_count(self) -> int | None:
        return self.logprobs if self.echo else None

    def _param(self, field: str) -> str:
        return "logprobs" if field in ("top_logprobs", "prompt_logprobs") else super()._param(field)

    def prompts(self) -> list[Prompt]:
        """One engine prompt per batch entry; a list of token ids is one prompt."""

        prompt = self.prompt
        if isinstance(prompt, str):
            return [TextPrompt(prompt)]
        if _is_token_ids(prompt):
            return [TokenPrompt(tuple(prompt))]
        return [
            TextPrompt(entry) if isinstance(entry, str) else TokenPrompt(tuple(entry))
            for entry in prompt
        ]
