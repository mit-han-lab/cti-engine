from __future__ import annotations

import os
from threading import Lock
from types import SimpleNamespace
from typing import get_type_hints
from unittest.mock import Mock, sentinel

import pytest

import cti.runtime as runtime_module
from cti.config import EngineOptions, EngineTuning
from cti.llm import LLM
from cti.multimodal import ImageLimits
from cti.protocol import SamplingParams
from cti.runtime import EngineClient, TrainingFuture
from cti.tokenizer import PromptEncoder
from cti.training import (
    ForwardBackwardCompletion,
    OptimizerStepResult,
    TrainingBatch,
)


def test_public_training_annotations_are_runtime_resolvable() -> None:
    constructor_hints = get_type_hints(LLM.__init__)
    forward_hints = get_type_hints(runtime_module.TrainingSession.forward_backward)
    step_hints = get_type_hints(runtime_module.TrainingSession.optim_step)
    save_hints = get_type_hints(runtime_module.TrainingSession.save_checkpoint)

    assert constructor_hints["options"] is EngineOptions
    assert constructor_hints["tuning"] == EngineTuning | None
    assert constructor_hints["resume_from_checkpoint"] == (str | os.PathLike[str] | None)
    assert forward_hints == {
        "batch": TrainingBatch,
        "return": TrainingFuture[ForwardBackwardCompletion],
    }
    assert step_hints == {"return": TrainingFuture[OptimizerStepResult]}
    assert save_hints == {
        "path": str | os.PathLike[str],
        "return": type(None),
    }


def test_the_client_loads_its_prompt_encoder_once_from_the_engine_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits = ImageLimits(max_images=8, max_image_tokens=24)
    # The engine info fields the encoder is loaded from.
    info = SimpleNamespace(
        model_path="/models/vl", hf_config=sentinel.hf_config, image_limits=limits
    )
    client = object.__new__(EngineClient)
    client.info, client._encoder, client._loading = info, None, Lock()
    encoder = SimpleNamespace(tokenizer=SimpleNamespace(added_tokens_decoder={}))
    load = Mock(return_value=encoder)
    monkeypatch.setattr(PromptEncoder, "from_checkpoint", load)

    assert client.prompt_encoder() is encoder
    assert client.prompt_encoder() is encoder
    load.assert_called_once_with("/models/vl", hf_config=sentinel.hf_config, limits=limits)


def test_the_llm_tokenizer_is_its_prompt_encoders(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LLM.tokenizer`` is the tokenizer prompts are encoded with, loaded on first use."""

    info = SimpleNamespace(
        model_path="/models/text", hf_config=sentinel.hf_config, image_limits=sentinel.limits
    )
    client = object.__new__(EngineClient)
    client.info, client._encoder, client._loading = info, None, Lock()
    encoder = SimpleNamespace(tokenizer=sentinel.tokenizer)
    load = Mock(return_value=encoder)
    monkeypatch.setattr(PromptEncoder, "from_checkpoint", load)
    llm = object.__new__(LLM)
    llm._client = client

    assert load.call_count == 0
    assert llm.tokenizer is sentinel.tokenizer
    assert llm.tokenizer is llm.prompt_encoder().tokenizer
    load.assert_called_once()


@pytest.mark.parametrize(
    ("bad", "field"),
    [
        ({"top_p": 1.5}, "top_p"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"top_logprobs": 2}, "top_logprobs"),
        ({"seed": -1}, "seed"),
    ],
)
def test_generate_refuses_a_bad_sampling_field_before_sending_anything(
    bad: dict, field: str
) -> None:
    client = object.__new__(EngineClient)
    client._sampler = Mock()
    llm = object.__new__(LLM)
    llm._client = client

    with pytest.raises(ValueError, match=f"^{field} ") as refused:
        llm.generate(
            [[1, 2, 3], [4]],
            [
                SamplingParams(max_new_tokens=1024, temperature=0),
                SamplingParams(**{"max_new_tokens": 8} | bad),
            ],
        )

    assert refused.value.field == field
    client._sampler.sample.assert_not_called()
