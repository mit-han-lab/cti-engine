from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tokenizers import Tokenizer, decoders
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from cti.tokenizer.text import TokenizerTextDecoder, load_tokenizer


def test_the_text_decoder_names_exactly_the_tokens_skipping_drops():
    backend = Tokenizer(WordLevel({"hello": 0, "<unk>": 1}, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="</s>")
    tokenizer.add_tokens(["<think>"])
    eos, think = tokenizer.convert_tokens_to_ids(["</s>", "<think>"])

    decoder = TokenizerTextDecoder(tokenizer)

    ids = [0, eos, think, 99]
    kept = [token for token in ids if decoder.renders(token, skip_special_tokens=True)]
    # The special token is skipped, and 99, which the vocabulary does not assign, shows nothing.
    assert kept == [0, think]
    assert decoder.renders(eos, skip_special_tokens=False)
    assert decoder.batch_decode([ids], skip_special_tokens=True) == [
        tokenizer.decode(kept, skip_special_tokens=False)
    ]
    assert "</s>" in decoder.batch_decode([ids], skip_special_tokens=False)[0]


_SPM = decoders.Replace("\u2581", " ")


# Only byte-level BPE alone is proven to keep a prefix's text; the rest go token by token.
_DECODERS = {
    "byte-level": (decoders.ByteLevel(), True),
    "byte-level-in-sequence": (decoders.Sequence([decoders.ByteLevel()]), False),
    "metaspace": (decoders.Metaspace(), False),
    "wordpiece": (decoders.WordPiece(cleanup=False), False),
    "bpe": (decoders.BPEDecoder(), False),
    "ctc": (decoders.CTC(), False),
    "byte-fallback": (
        decoders.Sequence(
            [_SPM, decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(" ", 1, 0)]
        ),
        False,
    ),
    "byte-level-masked": (
        decoders.Sequence([decoders.ByteLevel(), decoders.Replace("\ufffd", "x")]),
        False,
    ),
}


@pytest.mark.parametrize(("decoder", "stable"), _DECODERS.values(), ids=_DECODERS)
def test_the_text_decoder_knows_whether_a_prefix_keeps_its_text(decoder, stable: bool):
    backend = Tokenizer(WordLevel({"hello": 0, "<unk>": 1}, unk_token="<unk>"))
    backend.decoder = decoder
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)

    assert TokenizerTextDecoder(tokenizer).prefix_stable is stable
    # A tokenizer whose decoder cannot be read guarantees nothing.
    assert not TokenizerTextDecoder(SimpleNamespace(added_tokens_decoder={})).prefix_stable


def test_a_byte_level_vocabulary_with_an_empty_token_is_decoded_token_by_token():
    backend = Tokenizer(WordLevel({"hello": 0, "<unk>": 1, "": 2}, unk_token="<unk>"))
    backend.decoder = decoders.ByteLevel()

    assert not TokenizerTextDecoder(PreTrainedTokenizerFast(tokenizer_object=backend)).prefix_stable


def test_load_tokenizer_reads_local_chat_template(tmp_path):
    template_path = tmp_path / "chat_template.json"
    template_path.write_text('{"chat_template": "{{ messages }}"}', encoding="utf-8")
    tokenizer = SimpleNamespace(chat_template=None)

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
        result = load_tokenizer(str(tmp_path))

    assert result is tokenizer
    assert tokenizer.chat_template == "{{ messages }}"


def test_load_tokenizer_allows_missing_local_chat_template(tmp_path):
    tokenizer = SimpleNamespace(chat_template=None)

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
        result = load_tokenizer(str(tmp_path))

    assert result is tokenizer
    assert tokenizer.chat_template is None
