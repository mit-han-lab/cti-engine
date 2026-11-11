from __future__ import annotations

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from cti.protocol import DetokenizeMsg, RequestEnd, TokenUsage
from cti.sampling import TokenLogprobs
from cti.tokenizer.detokenize import IncrementalDetokenizer
from cti.tokenizer.text import TokenizerTextDecoder


@pytest.mark.parametrize("skip_special_tokens", [True, False])
def test_each_request_renders_special_tokens_as_it_asks(skip_special_tokens: bool) -> None:
    backend = Tokenizer(WordLevel({"hello": 0, "world": 1, "<unk>": 2}, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="</s>")
    eos = tokenizer.convert_tokens_to_ids("</s>")
    manager = IncrementalDetokenizer(TokenizerTextDecoder(tokenizer))
    ids = [0, eos, 1]
    end = RequestEnd(TokenUsage(1, len(ids)), len(ids))
    msgs = [
        DetokenizeMsg(
            uid=uid,
            token_ids=(token,),
            skip_special_tokens=skip_special_tokens if uid == 0 else not skip_special_tokens,
            end=end if index == len(ids) - 1 else None,
        )
        for index, token in enumerate(ids)
        for uid in (0, 1)
    ]

    texts = manager.detokenize(msgs)

    rendered = {
        uid: "".join(t for m, t in zip(msgs, texts, strict=True) if m.uid == uid) for uid in (0, 1)
    }
    assert rendered == {
        uid: tokenizer.decode(ids, skip_special_tokens=skip)
        for uid, skip in ((0, skip_special_tokens), (1, not skip_special_tokens))
    }
    assert manager.streams == {}


def test_logprob_tokens_are_each_decoded_alone_with_special_tokens_shown() -> None:
    backend = Tokenizer(WordLevel({"hello": 0, "world": 1, "<unk>": 2}, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="</s>")
    eos = tokenizer.convert_tokens_to_ids("</s>")
    manager = IncrementalDetokenizer(TokenizerTextDecoder(tokenizer))
    msgs = [
        DetokenizeMsg(uid=0, token_ids=(0,)),
        DetokenizeMsg(
            uid=1,
            token_ids=(eos,),
            logprobs=(TokenLogprobs(eos, -0.5, ((1, -0.25), (eos, -0.5))),),
            prompt_logprobs=(None, TokenLogprobs(1, -1.0, ((0, -0.75),))),
            end=RequestEnd(TokenUsage(2, 1), 1, is_stop_token=True),
        ),
    ]

    assert manager.token_texts(msgs) == [{}, {eos: "</s>", 1: "world", 0: "hello"}]
