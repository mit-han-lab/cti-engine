import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from cti.protocol import (
    ChatPrompt,
    Prompt,
    PromptReply,
    SamplingParams,
    TextPrompt,
    TokenizeMsg,
    TokenPrompt,
)
from cti.tokenizer import PromptEncoder
from cti.tokenizer.server import tokenize
from cti.tokenizer.text import TokenizerTextDecoder


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {
        token: token_id
        for token_id, token in enumerate(
            (
                "<unk>",
                "<bos>",
                "system",
                "user",
                "assistant",
                "Be",
                "concise.",
                "First",
                "Second",
                "turn",
                "hello",
                "café",
                "你好",
            )
        )
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(
        single="<bos> $A",
        special_tokens=[("<bos>", vocabulary["<bos>"])],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if loop.first %}{{ bos_token }} {% endif %}"
        "{{ message['role'] }} {{ message['content'] }} "
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant{% endif %}"
    )
    return tokenizer


# The model's vocabulary, padded beyond the tokenizer's 13 tokens as Qwen's is.
_VOCAB_SIZE = 16


def _encoder(tokenizer: PreTrainedTokenizerFast) -> PromptEncoder:
    return PromptEncoder(TokenizerTextDecoder(tokenizer), None, _VOCAB_SIZE)


def _tokenize(prompt: Prompt) -> torch.Tensor:
    message = TokenizeMsg(
        uid=0, prompt=prompt, sampling=SamplingParams(max_new_tokens=1024, temperature=0).snapshot()
    )
    request, prompt = tokenize(_encoder(_tiny_tokenizer()), message)
    assert prompt is None
    return request.input_ids


def test_plain_text_keeps_tokenizer_default_special_token_handling() -> None:
    tokenizer = _tiny_tokenizer()
    text = "hello café 你好"

    actual = _tokenize(TextPrompt(text))

    assert actual.dtype is torch.int32
    assert actual.tolist() == tokenizer.encode(text, add_special_tokens=True)
    assert actual.tolist() == [
        tokenizer.bos_token_id,
        tokenizer.convert_tokens_to_ids("hello"),
        tokenizer.convert_tokens_to_ids("café"),
        tokenizer.convert_tokens_to_ids("你好"),
    ]


def test_chat_template_bos_is_not_duplicated_for_system_and_multiturn_unicode() -> None:
    tokenizer = _tiny_tokenizer()
    conversation = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "First turn 你好"},
        {"role": "assistant", "content": "hello café"},
        {"role": "user", "content": "Second turn"},
    ]
    expected = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    )
    assert isinstance(expected, torch.Tensor)

    actual = _tokenize(ChatPrompt(tuple(conversation)))

    assert actual.dtype is torch.int32
    assert torch.equal(actual, expected.view(-1).to(torch.int32))
    assert actual.tolist().count(tokenizer.bos_token_id) == 1
    for token in ("system", "user", "assistant", "你好", "café"):
        assert tokenizer.convert_tokens_to_ids(token) in actual.tolist()


@pytest.mark.parametrize("skip_special_tokens", [True, False])
def test_token_prompt_passes_through_and_repeats_as_given(skip_special_tokens: bool) -> None:
    tokenizer = _tiny_tokenizer()
    token_ids = tuple(tokenizer.encode("hello café"))
    message = TokenizeMsg(
        uid=3,
        prompt=TokenPrompt(token_ids),
        sampling=SamplingParams(
            max_new_tokens=1024, temperature=0, skip_special_tokens=skip_special_tokens
        ).snapshot(),
        return_prompt=True,
    )

    request, prompt = tokenize(_encoder(tokenizer), message)

    assert request.uid == 3
    assert request.input_ids.dtype is torch.int32
    assert tuple(request.input_ids.tolist()) == token_ids
    # The echo keeps the prompt's special tokens, whatever the output skips.
    text = tokenizer.decode(
        token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    assert tokenizer.bos_token in text
    assert prompt == PromptReply(3, token_ids, text)


@pytest.mark.parametrize("prompt_logprobs", [None, 0])
def test_a_text_prompt_names_its_first_token_for_prompt_logprobs(
    prompt_logprobs: int | None,
) -> None:
    tokenizer = _tiny_tokenizer()
    sampling = SamplingParams(
        max_new_tokens=1024, temperature=0, prompt_logprobs=prompt_logprobs
    ).snapshot()
    message = TokenizeMsg(0, TextPrompt("hello café"), sampling, return_prompt=True)

    request, prompt = tokenize(_encoder(tokenizer), message)

    token_ids = tuple(request.input_ids.tolist())
    texts = {} if prompt_logprobs is None else {token_ids[0]: "<bos>"}
    assert prompt == PromptReply(0, token_ids, "hello café", texts)


@pytest.mark.parametrize("token_id", [-1, _VOCAB_SIZE])
def test_token_prompt_outside_the_model_vocabulary_is_rejected(token_id: int) -> None:
    with pytest.raises(ValueError, match=f"token id {token_id} is outside the vocabulary"):
        _tokenize(TokenPrompt((1, token_id)))


_CHAT = ({"role": "user", "content": "First turn 你好"},)


@pytest.mark.parametrize(
    "online",
    [
        pytest.param(TextPrompt("hello café"), id="text"),
        pytest.param(ChatPrompt(_CHAT), id="chat"),
        # Ids the model may sample beyond the tokenizer's vocabulary come back as prompts.
        pytest.param(TokenPrompt((1, 10, 14)), id="token-ids"),
    ],
)
def test_online_prompts_and_their_offline_encodings_are_the_same_requests(online: Prompt) -> None:
    encoder = _encoder(_tiny_tokenizer())
    request, _ = tokenize(
        encoder,
        TokenizeMsg(0, online, SamplingParams(max_new_tokens=1024, temperature=0).snapshot()),
    )
    offline = encoder.chat(_CHAT) if isinstance(online, ChatPrompt) else online
    message = encoder.request(
        0, offline, SamplingParams(temperature=0, max_new_tokens=1).snapshot()
    )

    assert message.input_ids.dtype is torch.int32
    assert message.input_ids.tolist() == request.input_ids.tolist()
    assert message.images is request.images is None


@pytest.mark.parametrize(
    ("prompt", "message"),
    [
        (TokenPrompt(()), "token prompts must not be empty"),
        (
            ChatPrompt(({"role": "user", "content": [{"type": "image_url", "image_url": {}}]},)),
            "the model takes no images",
        ),
    ],
)
def test_empty_token_prompts_and_images_for_a_text_model_are_rejected(
    prompt: Prompt, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _tokenize(prompt)
