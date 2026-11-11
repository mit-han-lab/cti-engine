# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/utils/hf.py`); see THIRD_PARTY_NOTICES.

"""What a checkpoint folder says, read once from its files: ``read_checkpoint`` for the model
(configuration, stop set, image preprocessing)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from cti.multimodal import read_image_preprocessing
from cti.utils import find_model_file

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from cti.config import ModelHub
    from cti.multimodal import ImagePreprocessing

    from ._download import CheckpointSource


@dataclass(frozen=True, kw_only=True)
class CheckpointFacts:
    """A checkpoint's facts, frozen once when it is read from its ``source``, whose folder
    holds its config, weights and tokenizer from one snapshot even if an alias moves."""

    source: CheckpointSource
    hf_config: PretrainedConfig
    # ``config.json`` as written, without Transformers' defaults.
    document: Mapping[str, Any]
    stop_token_ids: tuple[int, ...]
    # The image preprocessing the vision tower is built with and requests are bounded by;
    # None for a checkpoint without one.
    image_preprocessing: ImagePreprocessing | None

    @property
    def architecture(self) -> str:
        """The architecture the checkpoint is served as, its config's first."""

        return self.hf_config.architectures[0]


def read_checkpoint(path: str, *, hub: ModelHub, text_only: bool = False) -> CheckpointFacts:
    """Read the checkpoint at ``path``, a local directory or a repository id on ``hub``; a
    malformed EOS fails here. A repository id downloads only the files that describe the
    checkpoint, at one revision its weights download at once validated (``open_weights``).
    With ``text_only`` an image-text checkpoint reads as its text-only export: no vision
    tower, no image preprocessing."""

    from ._download import fetch_metadata

    source = fetch_metadata(_local_path(path), hub)
    folder = source.folder
    hf_config = _load_hf_config(folder)
    if text_only:
        hf_config = _text_only(hf_config)
    return CheckpointFacts(
        source=source,
        hf_config=hf_config,
        document=MappingProxyType(_load_config_document(folder)),
        stop_token_ids=_stop_token_ids(folder, hf_config),
        image_preprocessing=read_image_preprocessing(
            _architecture(hf_config), folder, hf_config=hf_config
        ),
    )


def _text_only(hf_config: PretrainedConfig) -> PretrainedConfig:
    """The text-only export's config of an image-text ``hf_config``; any other as it is."""

    import copy

    from .register import TEXT_ONLY_ARCHITECTURES

    architecture = TEXT_ONLY_ARCHITECTURES.get(_architecture(hf_config))
    if architecture is None:
        return hf_config
    text = copy.deepcopy(hf_config.get_text_config())
    text.architectures = [architecture]
    # transformers ties the image-text model's head from the top-level field.
    text.tie_word_embeddings = hf_config.tie_word_embeddings
    return text


def _architecture(hf_config: PretrainedConfig) -> str:
    """The config's first architecture, once ``architectures`` is checked to be a non-empty
    list of names."""

    architectures = hf_config.architectures
    if (
        not isinstance(architectures, list)
        or not architectures
        or not all(isinstance(architecture, str) and architecture for architecture in architectures)
    ):
        raise ValueError("Hugging Face model config must declare architectures")
    return architectures[0]


def _local_path(path: str) -> str:
    """``path`` with ``~`` expanded, made absolute when it names a local directory; a
    Hugging Face repository id otherwise. A path spelled as a file path (absolute, or
    starting with ``.`` or ``~``) must name an existing directory."""

    local = Path(path).expanduser()
    if local.is_dir():
        return str(local.resolve(strict=True))
    if path.startswith(("/", ".", "~")):
        raise FileNotFoundError(f"model directory {path} does not exist")
    return str(local)


def _stop_token_ids(model_path: str, hf_config: PretrainedConfig) -> tuple[int, ...]:
    """EOS ids that finish a request: the model's, then the tokenizer's (docs/structures.md)."""

    eos_token_id = _load_generation_config(model_path).get("eos_token_id")
    if eos_token_id is None:
        eos_token_id = getattr(hf_config, "eos_token_id", None)
    if eos_token_id is None:
        text_config = hf_config.get_text_config(decoder=True)
        eos_token_id = getattr(text_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = ()
    token_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else (eos_token_id,)
    for token_id in token_ids:
        if type(token_id) is not int:
            raise TypeError(
                f"eos_token_id must be an integer or a list of integers; got {eos_token_id!r}"
            )
        if token_id < 0:
            raise ValueError(f"eos_token_id must be non-negative; got {eos_token_id!r}")
    stop_token_ids = tuple(dict.fromkeys(token_ids))
    tokenizer_eos_token_id = _read_tokenizer_eos_token_id(model_path)
    if tokenizer_eos_token_id is None or tokenizer_eos_token_id in stop_token_ids:
        return stop_token_ids
    return (*stop_token_ids, tokenizer_eos_token_id)


def _load_hf_config(model_path: str) -> PretrainedConfig:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path)


def _load_json_object(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        document = json.load(f)
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return document


def _load_config_document(model_path: str) -> dict[str, Any]:
    """Return the fields of the model's ``config.json`` as written, without defaults."""

    path = find_model_file(model_path, "config.json")
    if path is None:
        raise ValueError(f"{model_path} has no config.json")
    return _load_json_object(path)


def _load_generation_config(model_path: str) -> dict[str, Any]:
    """Return the fields of the model's ``generation_config.json``, or ``{}`` without one."""

    path = find_model_file(model_path, "generation_config.json")
    return {} if path is None else _load_json_object(path)


def _read_tokenizer_eos_token_id(model_path: str) -> int | None:
    """Return the id of the tokenizer's ``eos_token``, or ``None`` if it declares none.

    Reads the files without building a tokenizer. The id comes from ``tokenizer_config.json``'s
    ``added_tokens_decoder``, else ``tokenizer.json``'s ``added_tokens`` (the
    ``Qwen2Tokenizer`` order); no class-default ``eos_token`` applies. Reference:
    transformers 5.16.1 ``PreTrainedTokenizerBase._from_pretrained``.
    """

    path = find_model_file(model_path, "tokenizer_config.json")
    if path is None:
        return None
    tokenizer_config = _load_json_object(path)
    eos_token = tokenizer_config.get("eos_token")
    if eos_token is None:
        return None
    if isinstance(eos_token, dict):  # a serialized AddedToken
        eos_token = eos_token.get("content")
    if type(eos_token) is not str:
        raise TypeError(f"eos_token in {path} must be a string; got {eos_token!r}")

    added_token_ids = {
        added_token.get("content"): int(token_id)
        for token_id, added_token in tokenizer_config.get("added_tokens_decoder", {}).items()
    }
    if eos_token not in added_token_ids:
        # Transformers 5.16.1 saves a fast tokenizer's added tokens only in tokenizer.json.
        tokenizer_path = find_model_file(model_path, "tokenizer.json")
        if tokenizer_path is not None:
            added_token_ids = {
                added_token.get("content"): added_token.get("id")
                for added_token in _load_json_object(tokenizer_path).get("added_tokens", ())
            }
    token_id = added_token_ids.get(eos_token)
    if type(token_id) is not int or token_id < 0:
        raise ValueError(
            f"the tokenizer's eos_token {eos_token!r} has no valid added-token id in "
            f"{model_path}; got {token_id!r}"
        )
    return token_id
