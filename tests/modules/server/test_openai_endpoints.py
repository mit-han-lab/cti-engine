"""OpenAI contracts of /v1/completions and /v1/chat/completions over HTTP."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace

import pytest
from fastapi.testclient import TestClient

from cti.config import ServerOptions
from cti.protocol import (
    AdmissionPriority,
    BaseTokenizerMsg,
    ChatPrompt,
    PromptReply,
    RejectionKind,
    RequestRejectedMsg,
    SamplingParams,
    TextPrompt,
    TokenizeMsg,
    TokenPrompt,
    TokenUsage,
    UserReply,
)
from cti.sampling import LogprobMode, TokenLogprobs
from cti.server import frontend
from cti.server.openai.requests import ChatCompletionRequest, CompletionRequest
from tests.support.frontend_link import TokenizerLink, serving_api

_USAGE = TokenUsage(prompt_tokens=3, completion_tokens=2, cached_prompt_tokens=1)
# The prompt, which the tokenizer sends when the response repeats it, then two tokens.
_OUTPUT = (
    PromptReply(0, (1, 2, 3), "Hi"),
    UserReply(0, "He", False, token_ids=(10,)),
    UserReply(0, "llo", True, _USAGE, "stop", token_ids=(11,)),
)
_CHAT = {"model": "m", "messages": [{"role": "user", "content": "Hi"}]}
_COMPLETION = {"model": "m", "prompt": "Hi"}


@dataclass(frozen=True)
class _Late:
    """A reply that arrives on its own, after the replies before it were read."""

    reply: object


# The servers a test opened, their HTTP clients closed after it.
_OPEN: list[_Server] = []


@pytest.fixture(autouse=True)
def _close_servers() -> Iterator[None]:
    yield
    while _OPEN:
        _OPEN.pop().close()


class _Server:
    """An HTTP client of a frontend whose engine requests get scripted replies, sent back
    over its tokenizer link; ``respond``, when given, runs after each call's replies."""

    def __init__(
        self,
        replies: Sequence[object] | Callable[[TokenizeMsg], Sequence[object]] = _OUTPUT,
        *,
        tool_calls: str = "auto",
        respond: Callable[[TokenizerLink], None] | None = None,
    ) -> None:
        def script(link: TokenizerLink, requests: Sequence[TokenizeMsg]) -> None:
            loop = asyncio.get_running_loop()
            for message in requests:
                for scripted in replies(message) if callable(replies) else replies:
                    late = isinstance(scripted, _Late)
                    reply = scripted.reply if late else scripted
                    if not message.return_prompt and isinstance(reply, PromptReply):
                        continue
                    reply = replace(reply, uid=message.uid)
                    if late:
                        loop.call_later(0.05, link.deliver, reply)
                    else:
                        link.deliver(reply)
            if respond is not None:
                respond(link)

        self.link = TokenizerLink(script)
        self.manager = self.link.manager
        self.api = serving_api(self.link, ServerOptions(tool_call_parser=tool_calls))
        self.app = self.api.app
        self._client: TestClient | None = None

    @property
    def client(self) -> TestClient:
        """An HTTP client, serving the app on a loop of its own until the test ends."""

        if self._client is None:
            self._client = TestClient(self.app).__enter__()
            _OPEN.append(self)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.__exit__(None, None, None)

    @property
    def sent(self) -> list[TokenizeMsg]:
        return self.link.requests

    @property
    def messages(self) -> list[list[BaseTokenizerMsg]]:
        """Each message of requests to the tokenizer: one per call."""

        return [msgs for _, msgs in self.link.messages if isinstance(msgs[0], TokenizeMsg)]

    @property
    def aborted(self) -> list[int]:
        """The requests the frontend aborted, once it holds none."""

        self.settle()
        return self.link.aborted

    def settle(self) -> None:
        """Wait until the frontend holds no request."""

        if self._client is not None:
            self._client.portal.call(self.link.settled)

    async def __aenter__(self) -> _Server:
        """Serve on the test's own loop instead of the HTTP client's."""

        self._serving = self.app.router.lifespan_context(self.app)
        await self._serving.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._serving.__aexit__(*exc)

    def post(self, path: str, body: dict[str, object]):
        return self.client.post(path, json=body)

    def stream(self, path: str, body: dict[str, object]) -> list[dict[str, object]]:
        response = self.post(path, {**body, "stream": True})
        assert response.status_code == 200, response.text
        events = [line.removeprefix("data: ") for line in response.text.splitlines() if line]
        assert events[-1] == "[DONE]"
        return [json.loads(event) for event in events[:-1]]


def _error(response) -> dict[str, object]:
    assert response.status_code == 400, response.text
    return response.json()["error"]


def test_completion_fields_reach_the_engine_request() -> None:
    server = _Server()
    body = {
        "model": "m",
        "prompt": [[1, 2], [3]],
        "max_tokens": 5,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 4,
        "min_p": 0.05,
        "presence_penalty": 0.5,
        "frequency_penalty": -0.5,
        "repetition_penalty": 1.1,
        "seed": 7,
        "ignore_eos": True,
        "stop": "\n\n",
        "stop_token_ids": [9],
        "logit_bias": {"50256": -100, "7": 2.5},
        "min_tokens": 3,
        "include_stop_str_in_output": True,
        "skip_special_tokens": False,
        "return_token_ids": True,
        "echo": True,
        "priority": "high",
        "user": "someone",
    }

    assert server.post("/v1/completions", body).status_code == 200

    assert [message.prompt for message in server.sent] == [TokenPrompt((1, 2)), TokenPrompt((3,))]
    message = server.sent[0]
    assert (
        message.sampling
        == SamplingParams(
            temperature=0.5,
            top_k=4,
            top_p=0.9,
            min_p=0.05,
            presence_penalty=0.5,
            frequency_penalty=-0.5,
            repetition_penalty=1.1,
            ignore_eos=True,
            max_new_tokens=5,
            seed=7,
            stop=("\n\n",),
            stop_token_ids=(9,),
            skip_special_tokens=False,
            logit_bias={50256: -100.0, 7: 2.5},
            min_tokens=3,
        ).snapshot()
    )
    assert message.return_prompt
    assert message.priority is AdmissionPriority.HIGH


def test_chat_fields_reach_the_engine_request() -> None:
    server = _Server()
    body = {**_CHAT, "stop": ["a", "b"], "seed": 3, "max_tokens": 4}

    assert server.post("/v1/chat/completions", body).status_code == 200

    [message] = server.sent
    assert message.prompt == ChatPrompt(({"role": "user", "content": "Hi"},))
    assert (message.sampling.seed, message.sampling.stop) == (3, ("a", "b"))
    assert not message.return_prompt


def test_chat_template_kwargs_reach_the_prompt() -> None:
    server = _Server()
    kwargs = {"enable_thinking": False, "custom_flag": {"depth": 2}}

    body = {**_CHAT, "chat_template_kwargs": kwargs}
    assert server.post("/v1/chat/completions", body).status_code == 200
    [message] = server.sent
    assert message.prompt == ChatPrompt(({"role": "user", "content": "Hi"},), (), kwargs)

    body = {**_CHAT, "chat_template_kwargs": ["enable_thinking"]}
    assert server.post("/v1/chat/completions", body).status_code == 400


def test_stop_strings_are_unbounded_in_count_and_length() -> None:
    """Like vLLM and SGLang, beyond OpenAI's 4 sequences."""
    server = _Server()
    stops = ["x" * 1000, *"abcdefg"]

    assert server.post("/v1/chat/completions", {**_CHAT, "stop": stops}).status_code == 200
    assert server.sent[0].sampling.stop == tuple(stops)


@pytest.mark.parametrize(
    ("path", "fields", "top_logprobs", "prompt_logprobs"),
    (
        ("/v1/chat/completions", {}, None, None),
        ("/v1/chat/completions", {"logprobs": True}, 0, None),
        ("/v1/chat/completions", {"logprobs": True, "top_logprobs": 20}, 20, None),
        ("/v1/completions", {"logprobs": 0}, 0, None),
        ("/v1/completions", {"logprobs": 5}, 5, None),
        ("/v1/completions", {"logprobs": 5, "echo": True}, 5, 5),
    ),
)
def test_logprobs_reach_the_engine_raw(
    path: str,
    fields: dict,
    top_logprobs: int | None,
    prompt_logprobs: int | None,
) -> None:
    server = _Server()
    body = _COMPLETION if path == "/v1/completions" else _CHAT

    assert server.post(path, {**body, **fields}).status_code == 200

    sampling = server.sent[0].sampling.law
    assert (sampling.logprobs, sampling.top_logprobs) == (
        None if top_logprobs is None else LogprobMode.RAW,
        top_logprobs or 0,
    )
    assert sampling.prompt_logprobs == prompt_logprobs


@pytest.mark.parametrize("seed", (None, 5))
@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize(
    ("path", "body", "prompts"),
    (
        ("/v1/chat/completions", _CHAT, [ChatPrompt(({"role": "user", "content": "Hi"},))]),
        ("/v1/completions", _COMPLETION, [TextPrompt("Hi")]),
        (
            "/v1/completions",
            {**_COMPLETION, "prompt": ["a", "b"]},
            [TextPrompt("a"), TextPrompt("b")],
        ),
    ),
)
def test_n_samples_each_prompt_n_times_with_vllm_indices_seeds_and_usage(
    path: str,
    body: dict,
    prompts: list[object],
    stream: bool,
    seed: int | None,
) -> None:
    server = _Server()
    body = {**body, "n": 3, "seed": seed}
    if stream:
        body["stream_options"] = {"include_usage": True}

    if stream:
        chunks = server.stream(path, body)
        usage = chunks[-1]["usage"]
        indices = {chunk["choices"][0]["index"] for chunk in chunks[:-1]}
    else:
        response = server.post(path, body).json()
        usage = response["usage"]
        indices = [choice["index"] for choice in response["choices"]]

    # Choice i * n + j is the j-th of prompt i, seeded seed + j.
    choices = 3 * len(prompts)
    assert sorted(indices) == list(range(choices))
    assert [message.prompt for message in server.sent] == [p for p in prompts for _ in range(3)]
    seeds = [None] * choices if seed is None else [5, 6, 7] * len(prompts)
    assert [message.sampling.seed for message in server.sent] == seeds
    # As vLLM counts it: each prompt once, every choice's output.
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (3 * len(prompts), 2 * choices)
    assert usage["prompt_tokens_details"] == {"cached_tokens": len(prompts)}


@pytest.mark.parametrize("path", ("/v1/completions", "/v1/chat/completions"))
def test_a_seed_whose_last_choice_overflows_is_a_400(path: str) -> None:
    server = _Server()
    body = _COMPLETION if path == "/v1/completions" else _CHAT

    error = _error(server.post(path, {**body, "n": 2, "seed": 2**63 - 1}))

    assert "seed + n - 1 must be below 2**63" in error["message"]
    assert (error["param"], error["code"]) == ("seed", "invalid_value")
    assert server.sent == []


@pytest.mark.parametrize(
    ("fields", "filters"),
    (
        ({"temperature": 0, "top_k": 5, "top_p": 0.5}, (None, 1.0)),
        ({"temperature": 0.7, "top_k": -1, "top_p": 0.5}, (None, 0.5)),
        ({"temperature": 0.7, "top_k": 5}, (5, 1.0)),
    ),
)
def test_greedy_drops_filters_and_top_k_minus_one_keeps_all(fields: dict, filters: tuple) -> None:
    server = _Server()

    assert server.post("/v1/chat/completions", {**_CHAT, **fields}).status_code == 200

    law = server.sent[0].sampling.law
    assert (law.top_k, law.top_p) == filters


@pytest.mark.parametrize(
    ("path", "fields", "param"),
    (
        ("/v1/chat/completions", {"top_logprobs": 2}, "top_logprobs"),
        ("/v1/chat/completions", {"logprobs": True, "top_logprobs": 21}, "top_logprobs"),
        ("/v1/completions", {"logprobs": 21}, "logprobs"),
        ("/v1/completions", {"logprobs": True}, "logprobs"),
        ("/v1/completions", {"logprobs": -1, "echo": True}, "logprobs"),
        ("/v1/completions", {"max_tokens": -1}, "max_tokens"),
        ("/v1/chat/completions", {"max_tokens": 0}, "max_tokens"),
        ("/v1/chat/completions", {"max_completion_tokens": -2}, "max_completion_tokens"),
        ("/v1/completions", {"max_tokens": 4, "min_tokens": 5}, "min_tokens"),
    ),
)
def test_a_refused_sampling_field_is_a_400_naming_it(path: str, fields: dict, param: str) -> None:
    server = _Server()
    body = _COMPLETION if path == "/v1/completions" else _CHAT

    error = _error(server.post(path, {**body, **fields}))

    assert (error["param"], error["code"]) == (param, "invalid_value")
    assert server.sent == []


# Two tokens and a stop token, whose text never shows but whose logprob does.
_LOGPROB_OUTPUT = (
    _OUTPUT[0],
    UserReply(
        0,
        "He",
        False,
        token_ids=(10,),
        logprobs=(TokenLogprobs(10, -0.5, ((10, -0.5), (12, -1.0))),),
        token_texts={10: "He", 12: "é"},
    ),
    UserReply(
        0,
        "llo",
        False,
        token_ids=(11,),
        logprobs=(TokenLogprobs(11, -0.25, ((13, -0.125), (14, -9999.0))),),
        token_texts={11: "llo", 13: "lo", 14: "x"},
    ),
    UserReply(
        0,
        "",
        True,
        TokenUsage(3, 3),
        "stop",
        token_ids=(2,),
        logprobs=(TokenLogprobs(2, -1.0, ((2, -1.0), (15, -2.0))),),
        token_texts={2: "<end>", 15: "!"},
    ),
)


def _chat_token(text: str, logprob: float) -> dict[str, object]:
    return {"token": text, "logprob": logprob, "bytes": list(text.encode())}


_CHAT_LOGPROBS = [
    {
        **_chat_token("He", -0.5),
        "top_logprobs": [_chat_token("He", -0.5), _chat_token("é", -1.0)],
    },
    {
        **_chat_token("llo", -0.25),
        "top_logprobs": [_chat_token("lo", -0.125), _chat_token("x", -9999.0)],
    },
    {
        **_chat_token("<end>", -1.0),
        "top_logprobs": [_chat_token("<end>", -1.0), _chat_token("!", -2.0)],
    },
]
_COMPLETION_LOGPROBS = {
    "tokens": ["He", "llo", "<end>"],
    "token_logprobs": [-0.5, -0.25, -1.0],
    "top_logprobs": [
        {"He": -0.5, "é": -1.0},
        {"llo": -0.25, "lo": -0.125, "x": -9999.0},
        {"<end>": -1.0, "!": -2.0},
    ],
    "text_offset": [0, 2, 5],
}


def test_chat_logprobs_cover_every_output_token_streamed_or_not() -> None:
    server = _Server(_LOGPROB_OUTPUT)
    body = {**_CHAT, "logprobs": True, "top_logprobs": 2, "return_token_ids": True}

    whole = server.post("/v1/chat/completions", body).json()["choices"][0]
    chunks = [chunk["choices"][0] for chunk in server.stream("/v1/chat/completions", body)]

    assert whole["logprobs"] == {"content": _CHAT_LOGPROBS}
    assert whole["token_ids"] == [10, 11, 2] and whole["message"]["content"] == "Hello"
    # Each chunk carries its own tokens' logprobs; the end chunk has none.
    assert [len(chunk["logprobs"]["content"]) for chunk in chunks[:-1]] == [
        len(chunk["token_ids"]) for chunk in chunks[:-1]
    ]
    assert [e for chunk in chunks[:-1] for e in chunk["logprobs"]["content"]] == _CHAT_LOGPROBS
    assert chunks[-1]["logprobs"] is None


def test_completion_logprobs_follow_the_legacy_format_streamed_or_not() -> None:
    server = _Server(_LOGPROB_OUTPUT)
    body = {**_COMPLETION, "logprobs": 2}

    whole = server.post("/v1/completions", body).json()["choices"][0]
    chunks = [chunk["choices"][0] for chunk in server.stream("/v1/completions", body)]

    assert whole["logprobs"] == _COMPLETION_LOGPROBS
    streamed = {
        key: [value for chunk in chunks[:-1] for value in chunk["logprobs"][key]]
        for key in _COMPLETION_LOGPROBS
    }
    assert streamed == _COMPLETION_LOGPROBS
    assert chunks[-1]["logprobs"] is None


# The prompt naming its first token, and the first reply's prompt logprobs naming the rest.
_PROMPT_LOGPROB_OUTPUT = (
    PromptReply(0, (1, 4), "Hi", {1: "H"}),
    replace(
        _LOGPROB_OUTPUT[1],
        prompt_logprobs=(None, TokenLogprobs(4, -2.0, ((3, -0.5),))),
        token_texts={**_LOGPROB_OUTPUT[1].token_texts, 4: "i", 3: "o"},
    ),
    *_LOGPROB_OUTPUT[2:],
)


def test_echoed_completion_logprobs_start_with_the_prompts_streamed_or_not() -> None:
    server = _Server(_PROMPT_LOGPROB_OUTPUT)
    body = {**_COMPLETION, "echo": True, "logprobs": 2}

    whole = server.post("/v1/completions", body).json()["choices"][0]
    chunks = [chunk["choices"][0] for chunk in server.stream("/v1/completions", body)]

    expected = {
        "tokens": ["H", "i", *_COMPLETION_LOGPROBS["tokens"]],
        "token_logprobs": [None, -2.0, *_COMPLETION_LOGPROBS["token_logprobs"]],
        "top_logprobs": [None, {"i": -2.0, "o": -0.5}, *_COMPLETION_LOGPROBS["top_logprobs"]],
        "text_offset": [0, 1, 2, 4, 7],
    }
    assert whole["text"] == "HiHello" and whole["logprobs"] == expected
    # The prompt's logprobs open the stream with the first output token's.
    assert chunks[0]["text"] == "HiHe" and chunks[0]["logprobs"]["tokens"] == ["H", "i", "He"]
    streamed = {
        key: [value for chunk in chunks[:-1] for value in chunk["logprobs"][key]]
        for key in expected
    }
    assert streamed == expected


def test_only_choice_0_scores_the_prompt_and_its_siblings_share_its_logprobs() -> None:
    def replies(message: TokenizeMsg) -> Sequence[object]:
        if message.sampling.law.prompt_logprobs is None:  # a sibling: no token texts
            return (replace(_PROMPT_LOGPROB_OUTPUT[0], token_texts={}), *_LOGPROB_OUTPUT[1:])
        return tuple(map(_Late, _PROMPT_LOGPROB_OUTPUT))  # choice 0 opens after its siblings

    server = _Server(replies)
    body = {**_COMPLETION, "prompt": ["a", "b"], "n": 2, "echo": True, "logprobs": 2}

    whole = server.post("/v1/completions", body).json()["choices"]
    chunks = server.stream("/v1/completions", body)

    assert [m.sampling.law.prompt_logprobs for m in server.sent] == [2, None] * 4
    assert len({json.dumps(choice["logprobs"]) for choice in whole}) == 1
    assert whole[1]["logprobs"]["tokens"][:2] == ["H", "i"]
    openings = {}
    for chunk in chunks:
        openings.setdefault(chunk["choices"][0]["index"], chunk["choices"][0]["logprobs"])
    assert sorted(openings) == [0, 1, 2, 3]
    assert len({json.dumps(logprobs) for logprobs in openings.values()}) == 1


@pytest.mark.parametrize(
    ("path", "body"), (("/v1/completions", _COMPLETION), ("/v1/chat/completions", _CHAT))
)
def test_a_request_without_logprobs_gets_null_logprobs(path: str, body: dict) -> None:
    server = _Server(_LOGPROB_OUTPUT)

    assert server.post(path, body).json()["choices"][0]["logprobs"] is None


@pytest.mark.parametrize(
    ("url", "status"), (("data:image/png;base64,AA==", 200), ("ftp://example.com/cat.png", 400))
)
def test_chat_images_are_fetched_before_they_reach_the_tokenizer(url: str, status: int) -> None:
    server = _Server()
    image = {"type": "image_url", "image_url": {"url": url}}
    body = {**_CHAT, "messages": [{"role": "user", "content": [image]}]}

    response = server.post("/v1/chat/completions", body)

    assert response.status_code == status, response.text
    if status == 200:
        [message] = server.sent
        assert message.prompt.messages[0]["content"] == [image]  # a data: URL passes as is
    else:
        assert response.json()["error"]["param"] == "image_url"
        assert server.sent == []


@pytest.mark.parametrize(
    ("prompt", "prompts"),
    (
        ("hi", [TextPrompt("hi")]),
        ([1, 2], [TokenPrompt((1, 2))]),
        (["a", "b"], [TextPrompt("a"), TextPrompt("b")]),
        ([[1], [2, 3]], [TokenPrompt((1,)), TokenPrompt((2, 3))]),
    ),
)
def test_each_completion_prompt_is_one_request_and_one_choice(
    prompt: object, prompts: list[object]
) -> None:
    server = _Server()

    response = server.post("/v1/completions", {"model": "m", "prompt": prompt}).json()

    assert [message.prompt for message in server.sent] == prompts
    assert [choice["index"] for choice in response["choices"]] == list(range(len(prompts)))
    assert response["usage"]["prompt_tokens"] == 3 * len(prompts)
    assert response["usage"]["prompt_tokens_details"] == {"cached_tokens": len(prompts)}


@pytest.mark.parametrize(
    "prompt", ([], [[]], [-1], [1, "a"], [[1, 2], "x"], ["x", [1]], [1.5], [True], True, None)
)
def test_malformed_completion_prompts_are_400s(prompt: object) -> None:
    server = _Server()

    error = _error(server.post("/v1/completions", {"model": "m", "prompt": prompt}))

    assert error["param"] == "prompt"
    assert server.sent == []


_UNSUPPORTED = (
    ("/v1/completions", {"best_of": 2}, "best_of is not supported"),
    ("/v1/completions", {"suffix": "x"}, "suffix is not supported"),
    ("/v1/chat/completions", {"response_format": {"type": "json_object"}}, "not supported"),
)


@pytest.mark.parametrize(("path", "fields", "message"), _UNSUPPORTED)
def test_unsupported_parameters_are_typed_400s(path: str, fields: dict, message: str) -> None:
    server = _Server()
    body = _COMPLETION if path == "/v1/completions" else _CHAT

    error = _error(server.post(path, {**body, **fields}))

    assert error == {
        "message": error["message"],
        "type": "invalid_request_error",
        "param": list(fields)[-1],
        "code": "unsupported_parameter",
    }
    assert message in error["message"]
    assert server.sent == []


_NOOPS = {
    "/v1/completions": {"best_of": 1, "suffix": ""},
    "/v1/chat/completions": {"response_format": {"type": "text"}},
}


@pytest.mark.parametrize("path", ("/v1/completions", "/v1/chat/completions"))
def test_unsupported_fields_that_ask_for_nothing_are_accepted_and_kept_out_of_the_schema(
    path: str,
) -> None:
    server = _Server()
    body = _COMPLETION if path == "/v1/completions" else _CHAT

    assert server.post(path, {**body, **_NOOPS[path]}).status_code == 200
    model = CompletionRequest if path == "/v1/completions" else ChatCompletionRequest
    fields = set(model.model_json_schema()["properties"])
    assert not fields & {"best_of", "suffix", "response_format"}


@pytest.mark.parametrize(
    ("field", "value", "message", "param", "code"),
    (
        ("stop", "", "non-empty strings", "stop", "invalid_value"),
        ("stop", ["a", ""], "non-empty strings", "stop", "invalid_value"),
        ("stop_token_ids", [-1], "must be non-negative", "stop_token_ids", "invalid_value"),
        (
            "stream_options",
            {"include_usage": True},
            "stream_options requires stream",
            "stream_options",
            "invalid_value",
        ),
        ("priority", "urgent", "'high', 'normal' or 'low'", "priority", "invalid_value"),
        ("temperature", 2.5, "less than or equal to 2", "temperature", "invalid_value"),
        ("top_p", 1.5, "top_p must be in (0, 1]", "top_p", "invalid_value"),
        ("top_p", 0, "top_p=0 is not supported", "top_p", "unsupported_parameter"),
        ("top_k", 0, "top_k must be -1 or a positive integer", "top_k", "invalid_value"),
        ("seed", -1, "seed must be an integer in [0, 2**63)", "seed", "invalid_value"),
        ("n", 0, "greater than or equal to 1", "n", "invalid_value"),
        ("min_p", 1.5, "min_p must be in [0, 1]", "min_p", "invalid_value"),
        ("presence_penalty", 2.5, "must be in [-2, 2]", "presence_penalty", "invalid_value"),
        ("frequency_penalty", -3, "must be in [-2, 2]", "frequency_penalty", "invalid_value"),
        ("repetition_penalty", 0, "must be positive", "repetition_penalty", "invalid_value"),
        ("logit_bias", {"7": 101}, "values must be in [-100, 100]", "logit_bias", "invalid_value"),
        (
            "logit_bias",
            {"-1": 1},
            "tokens must be distinct non-negative",
            "logit_bias",
            "invalid_value",
        ),
        ("logit_bias", {"a": 1}, "valid integer", "logit_bias.a.[key]", "invalid_value"),
        ("min_tokens", -1, "must be a non-negative integer", "min_tokens", "invalid_value"),
        ("stop", ["a", 7], "a string or a list of strings", "stop", "invalid_value"),
        ("unknown", 1, "Unrecognized request argument supplied: unknown", "unknown", None),
    ),
)
@pytest.mark.parametrize("path", ("/v1/completions", "/v1/chat/completions"))
def test_invalid_requests_are_400s_before_reaching_the_engine(
    path: str,
    field: str,
    value: object,
    message: str,
    param: str | None,
    code: str | None,
) -> None:
    server = _Server()
    body = _COMPLETION if path == "/v1/completions" else _CHAT

    error = _error(server.post(path, {**body, field: value}))

    assert message in error["message"]
    assert (error["param"], error["code"]) == (param, code)
    assert server.sent == []


def test_a_malformed_body_names_no_field() -> None:
    server = _Server()

    response = server.client.post(
        "/v1/completions",
        content=b'{"model": "m", "prompt": ',
        headers={"Content-Type": "application/json"},
    )

    error = _error(response)
    assert error["message"].startswith("Malformed JSON body")
    assert (error["param"], error["code"]) == (None, None)
    assert server.sent == []


def test_a_nested_error_names_its_full_path() -> None:
    server = _Server()

    wrong_role = _error(
        server.post(
            "/v1/chat/completions", {**_CHAT, "messages": [{"role": "function", "content": "x"}]}
        )
    )
    unknown = _error(
        server.post(
            "/v1/chat/completions",
            {**_CHAT, "stream": True, "stream_options": {"include_usage": True, "x": 1}},
        )
    )

    assert (wrong_role["param"], wrong_role["code"]) == ("messages[0].role", "invalid_value")
    assert unknown["param"] == "stream_options.x"
    assert server.sent == []


def test_chat_takes_max_completion_tokens_and_a_named_message() -> None:
    server = _Server()
    messages = [{"role": "user", "content": "Hi", "name": "ann"}]

    body = {**_CHAT, "messages": messages, "max_completion_tokens": 7}
    assert server.post("/v1/chat/completions", body).status_code == 200
    both = _error(server.post("/v1/chat/completions", {**body, "max_tokens": 7}))
    negative = _error(server.post("/v1/chat/completions", {**_CHAT, "max_completion_tokens": 0}))
    assert (negative["param"], negative["code"]) == ("max_completion_tokens", "invalid_value")

    (message,) = server.sent
    assert message.sampling.max_new_tokens == 7
    assert message.prompt == ChatPrompt(({"role": "user", "content": "Hi", "name": "ann"},))
    assert (both["param"], both["code"]) == ("max_tokens", "invalid_value")


def test_chat_needs_messages() -> None:
    server = _Server()

    for body in ({"model": "m", "messages": []}, {"model": "m", "prompt": "Hi"}):
        assert _error(server.post("/v1/chat/completions", body))["param"] in ("messages", "prompt")
    assert server.sent == []


_DEFAULTED = dict.fromkeys(("max_tokens", "temperature", "top_p", "n", "presence_penalty"))
_DEFAULTED_CHAT = {**_DEFAULTED, "max_completion_tokens": None}


@pytest.mark.parametrize(
    ("path", "body"), (("/v1/completions", _COMPLETION), ("/v1/chat/completions", _CHAT))
)
def test_a_nested_null_takes_its_default(path: str, body: dict) -> None:
    server = _Server()

    chunks = server.stream(path, {**body, "stream_options": {"include_usage": None}})

    assert all("usage" not in chunk for chunk in chunks)


_NULLABLE = dict.fromkeys(("seed", "stop", "stop_token_ids", "logit_bias", "user"))


@pytest.mark.parametrize(
    ("path", "body"), (("/v1/completions", _COMPLETION), ("/v1/chat/completions", _CHAT))
)
def test_null_fields_take_their_defaults(path: str, body: dict) -> None:
    server = _Server()

    defaulted = _DEFAULTED if path == "/v1/completions" else _DEFAULTED_CHAT
    assert server.post(path, {**body, **defaulted}).status_code == 200

    sampling = server.sent[0].sampling
    # Chat runs to the end of the context, as OpenAI and vLLM do; completions takes 16.
    max_tokens = 16 if path == "/v1/completions" else None
    law = sampling.law
    assert (sampling.max_new_tokens, law.temperature, law.top_p) == (max_tokens, 1.0, 1.0)


@pytest.mark.parametrize(
    ("path", "body"), (("/v1/completions", _COMPLETION), ("/v1/chat/completions", _CHAT))
)
def test_nullable_fields_keep_null(path: str, body: dict) -> None:
    server = _Server()

    assert server.post(path, {**body, **_NULLABLE}).status_code == 200

    (message,) = server.sent
    sampling = message.sampling
    assert (sampling.seed, sampling.stop, sampling.law.stop_token_ids) == (None, (), ())


@pytest.mark.parametrize(
    ("path", "body"), (("/v1/completions", _COMPLETION), ("/v1/chat/completions", _CHAT))
)
def test_an_unknown_field_is_rejected_even_when_null(path: str, body: dict) -> None:
    server = _Server()

    error = _error(server.post(path, {**body, "guided_json": None}))

    assert (error["param"], error["message"]) == (
        "guided_json",
        "Unrecognized request argument supplied: guided_json",
    )
    assert server.sent == []


def test_completion_response_echoes_the_prompt_and_returns_token_ids() -> None:
    server = _Server()
    body = {**_COMPLETION, "echo": True, "return_token_ids": True}

    response = server.post("/v1/completions", body).json()

    assert response["object"] == "text_completion"
    assert response["id"].startswith("cmpl-")
    assert response["choices"] == [
        {
            "index": 0,
            "text": "HiHello",
            "logprobs": None,
            "finish_reason": "stop",
            "prompt_token_ids": [1, 2, 3],
            "token_ids": [10, 11],
        }
    ]
    assert response["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 1},
    }


def test_completion_stream_matches_the_whole_response() -> None:
    server = _Server()
    body = {**_COMPLETION, "echo": True, "return_token_ids": True}

    chunks = server.stream("/v1/completions", {**body, "stream_options": {"include_usage": True}})
    whole = server.post("/v1/completions", body).json()

    choices = [choice for chunk in chunks for choice in chunk["choices"]]
    assert "".join(choice["text"] for choice in choices) == whole["choices"][0]["text"]
    assert [t for choice in choices for t in choice["token_ids"]] == [10, 11]
    assert choices[0]["prompt_token_ids"] == [1, 2, 3]
    assert [choice["finish_reason"] for choice in choices] == [None, None, "stop"]
    assert all(chunk["object"] == "text_completion" for chunk in chunks)
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"] == whole["usage"]
    assert all(chunk["usage"] is None for chunk in chunks[:-1])


def test_batch_completion_streams_each_choice_with_its_index() -> None:
    server = _Server()
    body = {"model": "m", "prompt": ["a", "b"], "stream_options": {"include_usage": True}}

    chunks = server.stream("/v1/completions", body)

    texts: dict[int, str] = {}
    for chunk in chunks[:-1]:
        (choice,) = chunk["choices"]
        texts[choice["index"]] = texts.get(choice["index"], "") + choice["text"]
    assert texts == {0: "Hello", 1: "Hello"}
    assert [c["choices"][0]["finish_reason"] for c in chunks[:-1]].count("stop") == 2
    assert chunks[-1]["usage"]["completion_tokens"] == 4
    assert len({(c["id"], c["created"], c["model"]) for c in chunks}) == 1


@pytest.mark.parametrize("stream", (False, True))
def test_a_rejected_prompt_fails_the_batch_and_aborts_the_others(
    stream: bool,
) -> None:
    rejection = RequestRejectedMsg(0, "too long", RejectionKind.CONTEXT_LENGTH_EXCEEDED)
    still_running = (UserReply(0, "x", False, token_ids=(1,)),)
    server = _Server(lambda message: (rejection,) if message.uid % 2 else still_running)

    response = server.post(
        "/v1/completions", {"model": "m", "prompt": ["a", "b"], "stream": stream}
    )

    assert _error(response) == {
        "message": "too long",
        "type": "invalid_request_error",
        "param": "prompt",
        "code": "context_length_exceeded",
    }
    assert {message.uid for message in server.sent} <= set(server.aborted)


def test_chat_returns_token_ids_like_vllm() -> None:
    server = _Server()
    body = {**_CHAT, "return_token_ids": True}

    response = server.post("/v1/chat/completions", body).json()
    chunks = server.stream("/v1/chat/completions", body)

    assert response["prompt_token_ids"] == [1, 2, 3]
    assert response["choices"][0]["token_ids"] == [10, 11]
    assert response["choices"][0]["message"] == {"role": "assistant", "content": "Hello"}
    assert chunks[0]["prompt_token_ids"] == [1, 2, 3]
    assert [c["choices"][0]["token_ids"] for c in chunks] == [[10], [11], []]


def test_a_stream_sends_one_chunk_per_engine_step_with_its_tokens_and_logprobs() -> None:
    # A speculative round commits "He" and "llo" in one step, then the stop token ends it.
    step = UserReply(
        0,
        "Hello",
        False,
        token_ids=(10, 11),
        logprobs=(TokenLogprobs(10, -0.5), TokenLogprobs(11, -0.25)),
        token_texts={10: "He", 11: "llo"},
    )
    end = UserReply(
        0,
        "",
        True,
        TokenUsage(3, 3),
        "stop",
        token_ids=(2,),
        logprobs=(TokenLogprobs(2, -1.0),),
        token_texts={2: "<end>"},
    )
    server = _Server((_OUTPUT[0], step, end))
    body = {**_CHAT, "return_token_ids": True, "logprobs": True}

    chunks = server.stream("/v1/chat/completions", body)

    deltas = [chunk["choices"][0] for chunk in chunks]
    assert [d["token_ids"] for d in deltas] == [[10, 11], [2], []]
    assert [len((d["logprobs"] or {}).get("content", [])) for d in deltas] == [2, 1, 0]
    assert deltas[0]["delta"] == {"role": "assistant", "content": "Hello"}


def test_chat_stream_starts_with_the_first_reply() -> None:
    usage = TokenUsage(prompt_tokens=2, completion_tokens=2)
    replies = (
        UserReply(0, "O", False, token_ids=(1,)),
        UserReply(0, "K", True, usage, "stop", (2,)),
    )
    server = _Server(replies)

    chunks = server.stream("/v1/chat/completions", _CHAT)

    assert [chunk["choices"][0]["delta"] for chunk in chunks] == [
        {"role": "assistant", "content": "O"},
        {"content": "K"},
        {},
    ]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_a_chat_stream_ends_with_its_finish_reason_and_the_usage_it_opts_into(
    finish_reason: str,
) -> None:
    usage = TokenUsage(prompt_tokens=9, completion_tokens=1, cached_prompt_tokens=5)
    server = _Server((UserReply(0, "OK", True, usage, finish_reason, (5,)),))

    chunks = server.stream(
        "/v1/chat/completions", {**_CHAT, "stream_options": {"include_usage": True}}
    )

    assert [chunk["choices"][0]["finish_reason"] for chunk in chunks[:-1]] == [None, finish_reason]
    assert all(chunk["usage"] is None for chunk in chunks[:-1])
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {
        "prompt_tokens": 9,
        "completion_tokens": 1,
        "total_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 5},
    }
    server.settle()


def test_an_inference_failure_mid_stream_ends_it_with_an_error_event() -> None:
    def first_token_then_failure(_message: TokenizeMsg) -> tuple[UserReply]:
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, server.manager.end, "inference execution failed")
        return (UserReply(0, "O", False, token_ids=(1,)),)

    server = _Server(first_token_then_failure)

    chunks = server.stream("/v1/chat/completions", _CHAT)

    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": "O"}
    assert chunks[1:] == [
        {"error": {"message": "inference execution failed", "type": "server_error"}}
    ]


def test_a_request_that_ended_before_an_inference_failure_succeeds() -> None:
    def output_then_failure(_message: TokenizeMsg) -> tuple[UserReply, ...]:
        # After the frontend's listener has routed the replies.
        asyncio.get_running_loop().call_later(0.05, server.manager.end, "later inference failure")
        return _OUTPUT[1:]

    server = _Server(output_then_failure)

    assert server.post("/v1/completions", _COMPLETION).json()["choices"][0]["text"] == "Hello"


_CONTEXT_LENGTH_ERROR = (
    "This model's maximum context length is 4096 tokens. However, you requested 6906 tokens "
    "(6890 in the prompt, 16 in the completion). Please reduce the length of the prompt or "
    "completion."
)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("path", "body", "param"),
    (("/v1/chat/completions", _CHAT, "messages"), ("/v1/completions", _COMPLETION, "prompt")),
    ids=("chat", "completion"),
)
def test_context_length_rejection_is_an_openai_400(
    stream: bool, path: str, body: dict, param: str
) -> None:
    rejection = RequestRejectedMsg(0, _CONTEXT_LENGTH_ERROR, RejectionKind.CONTEXT_LENGTH_EXCEEDED)
    server = _Server((rejection,))

    response = server.post(path, {**body, "stream": stream})

    assert _error(response) == {
        "message": _CONTEXT_LENGTH_ERROR,
        "type": "invalid_request_error",
        "param": param,
        "code": "context_length_exceeded",
    }


@pytest.mark.parametrize("stream", [False, True])
def test_tokenizer_rejection_is_a_request_error_without_poisoning_frontend(
    stream: bool,
) -> None:
    server = _Server((RequestRejectedMsg(0, "bad prompt", RejectionKind.INVALID_REQUEST),))

    response = server.post("/v1/completions", {**_COMPLETION, "stream": stream})

    assert _error(response) == {
        "message": "bad prompt",
        "type": "invalid_request_error",
        "param": "prompt",
        "code": None,
    }
    assert asyncio.run(server.api.v1_root()) == {"status": "ok"}


@pytest.mark.parametrize("stream", [False, True])
def test_unavailable_rejection_answers_503_before_any_stream(stream: bool) -> None:
    server = _Server((RequestRejectedMsg(0, "ingress hold is full"),))

    response = server.post("/v1/chat/completions", {**_CHAT, "stream": stream})

    assert response.status_code == 503
    assert response.json()["error"] == {
        "message": "request 0 was rejected: ingress hold is full",
        "type": "server_error",
        "param": None,
        "code": None,
    }


def test_an_unavailable_engine_answers_503() -> None:
    server = _Server()
    server.manager.end("inference execution failed")

    response = server.post("/v1/completions", _COMPLETION)

    assert response.status_code == 503
    assert response.json()["error"]["message"] == "inference execution failed"


def test_a_prompt_only_completion_echoes_with_no_output() -> None:
    server = _Server()
    body = {**_COMPLETION, "echo": True, "max_tokens": 0, "return_token_ids": True}

    whole = server.post("/v1/completions", body).json()
    chunks = server.stream("/v1/completions", body)

    # The engine samples one token, for the prompt's usage, and the reply drops it.
    assert server.sent[0].sampling.max_new_tokens == 1
    assert whole["choices"][0]["text"] == "Hi"
    assert whole["choices"][0]["token_ids"] == []
    assert whole["choices"][0]["finish_reason"] == "length"
    assert whole["usage"]["completion_tokens"] == 0
    assert "".join(c["choices"][0]["text"] for c in chunks) == "Hi"
    assert len(chunks) == 2  # the prompt, then the end


def test_a_calls_requests_reach_the_tokenizer_as_one_message() -> None:
    """Every choice of every prompt of a call goes out as one message, so the engine admits
    them together and a repeated seeded call is batched alike."""

    server = _Server()
    body = {"model": "m", "prompt": ["a", "b"], "n": 3, "seed": 7, "temperature": 0.9}
    assert server.post("/v1/completions", body).status_code == 200
    (call,) = server.messages
    assert [message.sampling.seed for message in call] == [7, 8, 9] * 2


def test_cancelled_dispatch_aborts_every_request_of_the_call() -> None:
    def fail(_link: TokenizerLink) -> None:
        raise RuntimeError("tokenizer link broke")

    server = _Server((), respond=fail)

    with pytest.raises(RuntimeError, match="tokenizer link broke"):
        server.post("/v1/completions", {"model": "m", "prompt": ["a", "b"]})

    assert set(server.aborted) == {0, 1}
    server.settle()


def test_backend_failure_before_first_replies_drops_every_request() -> None:
    def fail_after_last(link: TokenizerLink) -> None:
        if len(link.requests) == 3:
            link.manager.end("engine died")

    server = _Server((), respond=fail_after_last)

    response = server.post(
        "/v1/completions", {"model": "m", "prompt": ["a", "b", "c"], "stream": True}
    )

    assert response.status_code == 503
    server.settle()


def test_a_client_that_leaves_a_whole_response_aborts_its_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server((UserReply(0, "x", False, token_ids=(1,)),))

    async def left_immediately(_request) -> None:
        return None

    monkeypatch.setattr(frontend, "_disconnected", left_immediately)

    response = server.post("/v1/completions", {"model": "m", "prompt": ["a", "b"]})

    assert response.status_code == 499
    assert set(server.aborted) == {0, 1}
    server.settle()


class _StreamingEngine:
    """A frontend whose requests each stream a token every 5 ms until they are aborted;
    ``silent`` ones never reply, and the send that carries request number ``block_send``
    never returns. It serves while open, on the test's loop."""

    def __init__(
        self,
        *,
        silent: bool = False,
        block_send: int | None = None,
    ) -> None:
        self.blocked = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

        async def respond(link: TokenizerLink, requests: Sequence[TokenizeMsg]) -> None:
            if block_send is not None and len(link.requests) >= block_send:
                self.blocked.set()
                await asyncio.Event().wait()
            if not silent:
                for message in requests:
                    self._tasks.append(asyncio.create_task(self._generate(message.uid)))

        self.link = TokenizerLink(respond)
        self.app = serving_api(self.link).app
        self._serving = self.app.router.lifespan_context(self.app)

    async def __aenter__(self) -> _StreamingEngine:
        await self._serving.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._serving.__aexit__(*exc)

    @property
    def aborted(self) -> list[int]:
        return self.link.aborted

    async def _generate(self, uid: int) -> None:
        while uid in self.link.manager.pending:
            self.link.deliver(UserReply(uid, "x", False, token_ids=(1,)))
            await asyncio.sleep(0.005)

    async def settled(self) -> set[asyncio.Task]:
        """The tasks still running once the frontend holds no request and the engine's own
        tasks have seen the aborts, but for the frontend's listener."""

        await self.link.settled()
        await asyncio.gather(*self._tasks)
        tasks = asyncio.all_tasks() - {asyncio.current_task()}
        return {task for task in tasks if task.get_name() != "frontend-listener"}


async def _call_app(app, body: dict, send=None, receive_disconnect: asyncio.Event | None = None):
    """POST ``body`` to /v1/completions through the ASGI app; returns the messages sent."""

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/completions",
        "raw_path": b"/v1/completions",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("client", 1),
        "server": ("server", 80),
    }
    pending = [{"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}]
    disconnect = receive_disconnect or asyncio.Event()

    async def receive() -> dict:
        if pending:
            return pending.pop()
        await disconnect.wait()
        return {"type": "http.disconnect"}

    sent: list[dict] = []

    async def record(message: dict) -> None:
        sent.append(message)
        if send is not None:
            await send(message, sent)

    await app(scope, receive, record)
    return sent


_BATCH_STREAM = {"model": "m", "prompt": ["a", "b"], "max_tokens": 1000, "stream": True}


def test_a_failed_send_ends_the_stream_and_aborts_its_requests() -> None:
    async def fail_third_chunk(message: dict, sent: list[dict]) -> None:
        if len(sent) == 4:  # the start, two chunks, then this one
            raise OSError("connection reset")

    async def run_test() -> None:
        async with _StreamingEngine() as engine:
            with pytest.raises(OSError, match="connection reset"):
                await _call_app(engine.app, _BATCH_STREAM, fail_third_chunk)

            assert await engine.settled() == set()
            assert sorted(engine.aborted) == [0, 1]

    asyncio.run(run_test())


def test_a_failed_last_send_leaves_no_request_state() -> None:
    async def fail_finish_chunk(message: dict, sent: list[dict]) -> None:
        if b'"finish_reason": "stop"' in message.get("body", b""):
            raise OSError("connection reset")

    async def run_test() -> None:
        async with _Server() as server:
            with pytest.raises(OSError, match="connection reset"):
                await _call_app(server.app, {**_COMPLETION, "stream": True}, fail_finish_chunk)

            await server.link.settled()
            assert server.link.aborted == []  # the request had already ended

    asyncio.run(run_test())


def test_a_client_leaving_mid_stream_aborts_its_requests() -> None:
    async def run_test() -> None:
        disconnect = asyncio.Event()

        async def leave_after_two_chunks(message: dict, sent: list[dict]) -> None:
            if len(sent) == 3:
                disconnect.set()

        async with _StreamingEngine() as engine:
            sent = await _call_app(engine.app, _BATCH_STREAM, leave_after_two_chunks, disconnect)

            assert sent[0]["status"] == 200
            assert await engine.settled() == set()
            assert sorted(engine.aborted) == [0, 1]

    asyncio.run(run_test())


def test_a_client_leaving_before_the_first_reply_aborts_its_requests() -> None:
    async def run_test() -> None:
        disconnect = asyncio.Event()
        disconnect.set()

        async with _StreamingEngine(silent=True) as engine:
            sent = await _call_app(engine.app, _BATCH_STREAM, receive_disconnect=disconnect)

            assert sent[0]["status"] == 499
            assert await engine.settled() == set()
            assert sorted(engine.aborted) == [0, 1]

    asyncio.run(run_test())


def test_a_client_leaving_during_dispatch_aborts_its_requests() -> None:
    async def run_test() -> None:
        disconnect = asyncio.Event()

        async with _StreamingEngine(block_send=2) as engine:

            async def leave_once_blocked() -> None:
                await engine.blocked.wait()
                disconnect.set()

            leaving = asyncio.create_task(leave_once_blocked())
            sent = await _call_app(engine.app, _BATCH_STREAM, receive_disconnect=disconnect)
            await leaving

            assert sent[0]["status"] == 499
            assert await engine.settled() == set()
            # The second request was being sent when the client left.
            assert sorted(engine.aborted) == [0, 1]

    asyncio.run(run_test())


def test_a_rejection_answers_a_stream_without_waiting_for_the_other_prompts() -> None:
    rejection = RequestRejectedMsg(1, "too long", RejectionKind.CONTEXT_LENGTH_EXCEEDED)
    # The first request is queued without output; the second is rejected at once.
    server = _Server(lambda message: (rejection,) if message.uid else ())

    response = server.post("/v1/completions", {"model": "m", "prompt": ["a", "b"], "stream": True})

    assert _error(response)["code"] == "context_length_exceeded"
    assert {message.uid for message in server.sent} <= set(server.aborted)


def test_a_cancelled_dispatch_aborts_every_request_of_the_call() -> None:
    async def run_test() -> None:
        async with _StreamingEngine(block_send=2) as engine:
            call = asyncio.create_task(_call_app(engine.app, {**_BATCH_STREAM, "stream": False}))
            await engine.blocked.wait()

            call.cancel()

            with pytest.raises(asyncio.CancelledError):
                await call
            assert await engine.settled() == set()
            assert sorted(engine.aborted) == [0, 1]

    asyncio.run(run_test())


def _stopped(*texts: str, stop_str: str | None) -> tuple[PromptReply | UserReply, ...]:
    """The prompt, then one reply per text, token ids 1.., the last ending the output at
    ``stop_str``."""

    usage = TokenUsage(prompt_tokens=1, completion_tokens=len(texts))
    *body, last = texts
    return (
        PromptReply(0, (7,), "Q"),
        *(UserReply(0, text, False, token_ids=(i + 1,)) for i, text in enumerate(body)),
        UserReply(0, last, True, usage, "stop", token_ids=(len(texts),), stop_str=stop_str),
    )


@pytest.mark.parametrize(
    ("texts", "include", "streamed"),
    (
        # The held "\n" may begin the stop; the last reply completes it inside its text.
        (("Hi\n", "\nthere"), False, ["Hi", ""]),
        # Included, the stop string is released as it streams; the text after it is cut.
        (("Hi\n", "\nthere"), True, ["Hi\n", "\n"]),
        # A held tail that turns out not to begin the stop is released with what follows.
        (("a\n", "b\n", "\n"), False, ["a", "\nb", ""]),
    ),
)
@pytest.mark.parametrize("path", ("/v1/completions", "/v1/chat/completions"))
def test_text_ends_where_the_engine_named_stop_string_begins(
    path: str,
    texts: tuple[str, ...],
    include: bool,
    streamed: list[str],
) -> None:
    server = _Server(_stopped(*texts, stop_str="\n\n"))
    body = {
        **(_COMPLETION if path == "/v1/completions" else _CHAT),
        "stop": ["\n\n", "END"],
        "include_stop_str_in_output": include,
        "return_token_ids": True,
    }

    whole = server.post(path, body).json()["choices"][0]
    chunks = [chunk["choices"][0] for chunk in server.stream(path, body)]

    if path == "/v1/completions":
        text, pieces = whole["text"], [chunk["text"] for chunk in chunks[:-1]]
    else:
        text = whole["message"]["content"]
        pieces = [chunk["delta"].get("content", "") for chunk in chunks[:-1]]
    assert pieces == streamed and text == "".join(streamed)
    # The token ids are the engine's output: through the token completing the stop.
    assert whole["token_ids"] == list(range(1, len(texts) + 1))
    assert whole["finish_reason"] == chunks[-1]["finish_reason"] == "stop"


_PROMPT = PromptReply(0, (1, 2, 3), "Hi")
_HE = UserReply(0, "He", False, token_ids=(10,))
_LLO = UserReply(0, "llo", True, TokenUsage(prompt_tokens=3, completion_tokens=2), "length", (11,))


@pytest.mark.parametrize(
    "replies",
    [(_HE, _Late(_PROMPT), _Late(_LLO)), (_HE, _LLO, _Late(_PROMPT))],
    ids=["prompt-between", "prompt-after-the-end"],
)
def test_a_prompt_reply_arriving_after_output_still_comes_first(replies: tuple) -> None:
    server = _Server(replies)
    body = {**_COMPLETION, "echo": True, "return_token_ids": True}

    whole = server.post("/v1/completions", body).json()
    chunks = server.stream("/v1/completions", body)

    choice = whole["choices"][0]
    assert (choice["text"], choice["prompt_token_ids"]) == ("HiHello", [1, 2, 3])
    assert whole["usage"]["completion_tokens"] == 2
    first = chunks[0]["choices"][0]
    assert first["text"] == "HiHe" and first["prompt_token_ids"] == [1, 2, 3]
    assert [c["choices"][0]["finish_reason"] for c in chunks] == [None, None, "length"]


def test_a_prompt_only_completion_waits_for_a_late_prompt() -> None:
    server = _Server((_LLO, _Late(_PROMPT)))
    body = {**_COMPLETION, "echo": True, "max_tokens": 0}

    whole = server.post("/v1/completions", body).json()

    assert whole["choices"][0]["text"] == "Hi"
    assert whole["usage"]["completion_tokens"] == 0


def test_a_stream_waits_past_the_prompt_reply_for_the_engine_answer() -> None:
    rejection = RequestRejectedMsg(0, "too long", RejectionKind.CONTEXT_LENGTH_EXCEEDED)
    server = _Server((PromptReply(0, (1,), "Hi"), rejection))

    response = server.post("/v1/completions", {**_COMPLETION, "echo": True, "stream": True})

    assert _error(response)["code"] == "context_length_exceeded"


# Tool calls: Qwen3's hermes format, its text in two replies split inside the call.
_WEATHER_TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object"}},
}
_TOOLS = [_WEATHER_TOOL, {"type": "function", "function": {"name": "now"}}]
_CALL_TEXT = 'Let me check.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
_CALLING = (
    UserReply(0, _CALL_TEXT[:30], False, token_ids=(10,)),
    UserReply(0, _CALL_TEXT[30:], True, _USAGE, "stop", token_ids=(11,)),
)
_HERMES = "hermes"


def _tool_server(replies: Sequence[object] = _CALLING) -> _Server:
    return _Server(replies, tool_calls=_HERMES)


def test_a_call_in_the_text_is_the_message_s_tool_call() -> None:
    server = _tool_server()

    whole = server.post("/v1/chat/completions", {**_CHAT, "tools": _TOOLS}).json()

    [choice] = whole["choices"]
    [call] = choice["message"]["tool_calls"]
    assert choice["message"]["content"] == "Let me check.\n"
    assert call["id"].startswith("call_") and len(call["id"]) == len("call_") + 24
    assert (call["type"], call["function"]) == (
        "function",
        {"name": "get_weather", "arguments": '{"city": "Paris"}'},
    )
    assert choice["finish_reason"] == "tool_calls"


def test_a_streamed_call_names_itself_once_and_adds_up_to_the_whole() -> None:
    server = _tool_server()

    chunks = server.stream("/v1/chat/completions", {**_CHAT, "tools": _TOOLS})

    deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
    calls = [call for delta in deltas for call in delta.get("tool_calls", ())]
    assert "".join(delta.get("content", "") for delta in deltas) == "Let me check.\n"
    assert {call["index"] for call in calls} == {0}
    assert [("id" in call, "type" in call, "name" in call["function"]) for call in calls] == [
        (True, True, True)
    ] + [(False, False, False)] * (len(calls) - 1)
    assert calls[0]["function"]["name"] == "get_weather"
    assert "".join(call["function"]["arguments"] for call in calls) == '{"city": "Paris"}'
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_a_choice_cut_by_length_keeps_its_reason_and_an_unfinished_call_is_text() -> None:
    cut = (UserReply(0, "<tool_call>\n", True, _USAGE, "length", token_ids=(10,)),)

    choice = _tool_server(cut).post("/v1/chat/completions", {**_CHAT, "tools": _TOOLS}).json()

    assert choice["choices"][0]["message"] == {"role": "assistant", "content": "<tool_call>\n"}
    assert choice["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize(
    ("tool_choice", "offered"),
    (
        (None, _TOOLS),
        ("auto", _TOOLS),
        ("required", _TOOLS),
        ({"type": "function", "function": {"name": "now"}}, _TOOLS[1:]),
        ("none", []),
    ),
)
def test_tool_choice_picks_the_tools_the_template_offers(tool_choice, offered) -> None:
    server = _tool_server()

    body = {**_CHAT, "tools": _TOOLS, "tool_choice": tool_choice}
    response = server.post("/v1/chat/completions", body).json()

    [message] = server.sent
    assert list(message.prompt.tools) == offered
    # A call of a tool the template did not offer is text.
    calls = response["choices"][0]["message"].get("tool_calls")
    assert (calls is None) == (_WEATHER_TOOL not in offered)


def test_a_call_history_reaches_the_template_with_arguments_as_a_mapping() -> None:
    server = _tool_server()
    call = {"id": "call_1", "type": "function", "function": {"name": "now", "arguments": "{}"}}
    messages = [
        {"role": "user", "content": "Time?"},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "noon"},
    ]

    body = {**_CHAT, "messages": messages, "tools": _TOOLS}
    assert server.post("/v1/chat/completions", body).status_code == 200

    [message] = server.sent
    assert message.prompt.messages[1:] == (
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{**call, "function": {"name": "now", "arguments": {}}}],
        },
        {"role": "tool", "content": "noon", "tool_call_id": "call_1"},
    )


def test_tools_for_a_template_that_takes_none_are_a_400() -> None:
    server = _Server()

    error = _error(server.post("/v1/chat/completions", {**_CHAT, "tools": _TOOLS}))
    plain = server.post("/v1/chat/completions", {**_CHAT, "tools": _TOOLS, "tool_choice": "none"})

    assert (error["param"], error["message"]) == (
        "tools",
        "this model's chat template takes no tools",
    )
    assert plain.status_code == 200
    assert server.sent[0].prompt.tools == ()


_CALL = {"id": "c", "type": "function", "function": {"name": "now", "arguments": "{}"}}


@pytest.mark.parametrize(
    ("fields", "param"),
    (
        ({"tools": []}, "tools"),
        ({"tools": [_WEATHER_TOOL, _WEATHER_TOOL]}, "tools"),
        ({"tools": [{"type": "function", "function": {}}]}, "tools[0].function.name"),
        ({"tool_choice": "required"}, "tool_choice"),
        (
            {"tools": _TOOLS, "tool_choice": {"type": "function", "function": {"name": "x"}}},
            "tool_choice.function.name",
        ),
        (
            {"messages": [{"role": "user", "content": "x", "tool_calls": [_CALL]}]},
            "messages[0].tool_calls",
        ),
        ({"messages": [{"role": "tool", "content": "x"}]}, "messages[0].tool_call_id"),
        (
            {"messages": [{"role": "user", "content": "x", "tool_call_id": "c"}]},
            "messages[0].tool_call_id",
        ),
        ({"messages": [{"role": "assistant", "content": None}]}, "messages[0].content"),
        (
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [{**_CALL, "function": {"name": "now", "arguments": "[1]"}}],
                    }
                ]
            },
            "messages[0].tool_calls[0].function.arguments",
        ),
    ),
)
def test_malformed_tools_and_tool_messages_are_400s_naming_the_field(fields, param) -> None:
    server = _tool_server()

    error = _error(server.post("/v1/chat/completions", {**_CHAT, **fields}))

    assert error["param"] == param
    assert server.sent == []
