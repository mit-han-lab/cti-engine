import logging
from importlib import import_module

import pytest

import cti.inference as inference
import cti.training as training
import cti.utils as utils


def test_inference_package_exports_only_public_values() -> None:
    assert set(inference.__all__) == {
        "DeviceBudget",
        "Executor",
        "Forward",
        "GraphStats",
        "HostedModels",
        "InferenceEngine",
        "load_speculative",
        "PageMap",
        "RequestSeeds",
        "RequestTable",
        "ResidentBeside",
        "Round",
        "RoundCompletion",
        "Sampled",
        "ServingLimits",
        "TextDecoder",
        "build_executor",
    }
    assert not hasattr(inference, "InferenceRunner")


def test_training_package_exports_only_public_values() -> None:
    assert set(training.__all__) == {
        "CheckpointFormatError",
        "CommandStateError",
        "ForwardBackwardCompletion",
        "ForwardBackwardObservation",
        "GRPO",
        "InvalidGradientError",
        "OptimizerStepResult",
        "PackedTokens",
        "PaddedTokens",
        "Refusal",
        "SFT",
        "SFTDatum",
        "SwitchRejectedError",
        "TrainingBatch",
        "TrainingCheckpoint",
        "TrainingInputError",
        "TrainingVerb",
    }
    assert not hasattr(training, "LoRATrainer")
    assert not hasattr(training, "TrainableParameterSet")
    assert not hasattr(training, "Trainer")
    assert not hasattr(training, "TrainingRunner")


def test_utils_package_exports_every_name_its_users_import() -> None:
    assert set(utils.__all__) == {
        "DataclassTree",
        "DeviceCapability",
        "align_ceil",
        "div_ceil",
        "find_model_file",
        "get_device_capability",
        "init_logger",
        "log_stage",
        "lazy_getattr",
        "mem_GB",
        "nvtx_annotate",
    }


@pytest.mark.parametrize(
    ("package", "name", "owner"),
    (
        *(
            ("cti.multimodal", name, f"cti.multimodal.{owner}")
            for owner, names in (
                ("inputs", ("ImageInputs", "ImageSpan")),
                (
                    "processor",
                    ("InputProcessor", "ProcessedPrompt", "create_input_processor", "rope_layout"),
                ),
                ("rope", ("RopePositionLayout",)),
            )
            for name in names
        ),
        ("cti.models", "ModelCapability", "cti.models.base"),
        ("cti.models", "model_signature", "cti.models.identity"),
        ("cti.inference", "Forward", "cti.inference._step"),
        ("cti.training", "SFTDatum", "cti.training._batch"),
        ("cti.tokenizer", "tokenize_worker", "cti.tokenizer.server"),
    ),
)
def test_lazy_exports_load_from_their_owner_and_unknown_names_raise(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    package: str,
    name: str,
    owner: str,
) -> None:
    namespace = import_module(package)
    expected = getattr(import_module(owner), name)
    monkeypatch.delitem(vars(namespace), name, raising=False)
    caplog.set_level(logging.DEBUG)

    value = getattr(namespace, name)

    assert value is expected
    assert vars(namespace)[name] is value
    with pytest.raises(AttributeError) as error:
        namespace.missing  # noqa: B018
    assert str(error.value) == f"module {package!r} has no attribute 'missing'"
    assert error.value.__cause__ is None and error.value.__context__ is None
    assert not hasattr(error.value, "__notes__")
    assert not caplog.records
