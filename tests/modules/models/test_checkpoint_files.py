import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen2Tokenizer

from cti.models.checkpoint import (
    _load_generation_config,
    _read_tokenizer_eos_token_id,
)

# Qwen/Qwen3.5-9B tokenizer_config.json, trimmed to its EOS and pad tokens.
_QWEN3_5_TOKENIZER_CONFIG = {
    "added_tokens_decoder": {
        "248044": {"content": "<|endoftext|>", "special": True},
        "248046": {"content": "<|im_end|>", "special": True},
    },
    "eos_token": "<|im_end|>",
    "pad_token": "<|endoftext|>",
}


def _write_json(path, document) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def _save_tokenizer(path, **special_tokens) -> None:
    backend = Tokenizer(WordLevel({"hello": 0, "<unk>": 1}, unk_token="<unk>"))
    PreTrainedTokenizerFast(tokenizer_object=backend, **special_tokens).save_pretrained(path)


def test_load_generation_config_reads_a_local_file(tmp_path):
    (tmp_path / "generation_config.json").write_text(
        '{"eos_token_id": [151645, 151643], "top_k": 20}',
        encoding="utf-8",
    )

    assert _load_generation_config(str(tmp_path)) == {
        "eos_token_id": [151645, 151643],
        "top_k": 20,
    }


def test_load_generation_config_allows_a_missing_local_file(tmp_path):
    assert _load_generation_config(str(tmp_path)) == {}


def test_load_generation_config_rejects_a_non_object_file(tmp_path):
    (tmp_path / "generation_config.json").write_text("[151645]", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        _load_generation_config(str(tmp_path))


def test_read_tokenizer_eos_token_id_reads_the_added_tokens_decoder(tmp_path):
    _write_json(tmp_path / "tokenizer_config.json", _QWEN3_5_TOKENIZER_CONFIG)

    assert _read_tokenizer_eos_token_id(str(tmp_path)) == 248046


def test_read_tokenizer_eos_token_id_reads_a_serialized_added_token(tmp_path):
    tokenizer_config = {
        **_QWEN3_5_TOKENIZER_CONFIG,
        "eos_token": {"__type": "AddedToken", "content": "<|endoftext|>", "special": True},
    }
    _write_json(tmp_path / "tokenizer_config.json", tokenizer_config)

    assert _read_tokenizer_eos_token_id(str(tmp_path)) == 248044


def test_read_tokenizer_eos_token_id_falls_back_to_tokenizer_json(tmp_path):
    # Transformers 5.16.1 saves a fast tokenizer's added tokens only in tokenizer.json.
    _write_json(tmp_path / "tokenizer_config.json", {"eos_token": "<|im_end|>"})
    _write_json(
        tmp_path / "tokenizer.json",
        {
            "added_tokens": [
                {"id": 151643, "content": "<|endoftext|>"},
                {"id": 151645, "content": "<|im_end|>"},
            ]
        },
    )

    assert _read_tokenizer_eos_token_id(str(tmp_path)) == 151645


@pytest.mark.parametrize("tokenizer_config", [None, {"pad_token": "<pad>"}, {"eos_token": None}])
def test_read_tokenizer_eos_token_id_allows_a_tokenizer_without_eos(tmp_path, tokenizer_config):
    if tokenizer_config is not None:
        _write_json(tmp_path / "tokenizer_config.json", tokenizer_config)

    assert _read_tokenizer_eos_token_id(str(tmp_path)) is None


@pytest.mark.parametrize("tokenizer_json", [None, {"added_tokens": [{"id": 0, "content": "<s>"}]}])
def test_read_tokenizer_eos_token_id_rejects_an_eos_without_an_added_token(
    tmp_path,
    tokenizer_json,
):
    _write_json(
        tmp_path / "tokenizer_config.json",
        {**_QWEN3_5_TOKENIZER_CONFIG, "eos_token": "</s>"},
    )
    if tokenizer_json is not None:
        _write_json(tmp_path / "tokenizer.json", tokenizer_json)

    with pytest.raises(ValueError, match="eos_token '</s>' has no valid added-token id"):
        _read_tokenizer_eos_token_id(str(tmp_path))


@pytest.mark.parametrize("eos_token", [7, {"special": True}])
def test_read_tokenizer_eos_token_id_rejects_a_malformed_eos(tmp_path, eos_token):
    _write_json(tmp_path / "tokenizer_config.json", {"eos_token": eos_token})

    with pytest.raises(TypeError, match="eos_token in .* must be a string"):
        _read_tokenizer_eos_token_id(str(tmp_path))


@pytest.mark.parametrize(
    ("tokenizer_config", "tokenizer_json"),
    [
        ({"added_tokens_decoder": {"-1": {"content": "<|im_end|>"}}}, None),
        ({}, {"added_tokens": [{"id": -1, "content": "<|im_end|>"}]}),
    ],
)
def test_read_tokenizer_eos_token_id_rejects_a_negative_id(
    tmp_path,
    tokenizer_config,
    tokenizer_json,
):
    _write_json(
        tmp_path / "tokenizer_config.json",
        {**tokenizer_config, "eos_token": "<|im_end|>"},
    )
    if tokenizer_json is not None:
        _write_json(tmp_path / "tokenizer.json", tokenizer_json)

    with pytest.raises(ValueError, match="has no valid added-token id .*; got -1$"):
        _read_tokenizer_eos_token_id(str(tmp_path))


@pytest.mark.parametrize(
    "special_tokens",
    [
        {"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"},
        {"eos_token": "<|endoftext|>", "pad_token": "<|im_end|>"},
        {"pad_token": "<|endoftext|>"},
    ],
)
def test_read_tokenizer_eos_token_id_matches_transformers_for_a_saved_tokenizer(
    tmp_path,
    special_tokens,
):
    _save_tokenizer(tmp_path, **special_tokens)

    expected = AutoTokenizer.from_pretrained(tmp_path).eos_token_id
    assert _read_tokenizer_eos_token_id(str(tmp_path)) == expected


def test_read_tokenizer_eos_token_id_matches_transformers_for_an_added_tokens_decoder(tmp_path):
    _save_tokenizer(tmp_path, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tokenizer_json = json.loads((tmp_path / "tokenizer.json").read_text(encoding="utf-8"))
    # The layout of the Qwen3.5 checkpoints: added tokens also in tokenizer_config.json.
    _write_json(
        tmp_path / "tokenizer_config.json",
        {
            "added_tokens_decoder": {
                str(token["id"]): {"content": token["content"], "special": True}
                for token in tokenizer_json["added_tokens"]
            },
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "tokenizer_class": "TokenizersBackend",
        },
    )

    expected = AutoTokenizer.from_pretrained(tmp_path).eos_token_id
    assert expected is not None
    assert _read_tokenizer_eos_token_id(str(tmp_path)) == expected


def test_read_tokenizer_eos_token_id_matches_transformers_when_the_files_disagree(tmp_path):
    Qwen2Tokenizer(vocab={"hello": 0}, eos_token="<|im_end|>").save_pretrained(tmp_path)
    tokenizer_json = json.loads((tmp_path / "tokenizer.json").read_text(encoding="utf-8"))
    token_ids = {token["content"]: token["id"] for token in tokenizer_json["added_tokens"]}
    tokenizer_config = json.loads((tmp_path / "tokenizer_config.json").read_text(encoding="utf-8"))
    # Contradict tokenizer.json by swapping the two ids; Qwen2Tokenizer follows this file.
    tokenizer_config["added_tokens_decoder"] = {
        str(token_ids["<|endoftext|>"]): {"content": "<|im_end|>", "special": True},
        str(token_ids["<|im_end|>"]): {"content": "<|endoftext|>", "special": True},
    }
    _write_json(tmp_path / "tokenizer_config.json", tokenizer_config)

    expected = AutoTokenizer.from_pretrained(tmp_path).eos_token_id
    assert expected == token_ids["<|endoftext|>"]
    assert _read_tokenizer_eos_token_id(str(tmp_path)) == expected
