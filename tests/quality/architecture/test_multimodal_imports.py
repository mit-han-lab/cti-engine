from __future__ import annotations

from tests.support.imports import assert_not_loaded


def test_frontend_processor_does_not_load_the_compute_stack() -> None:
    assert_not_loaded(
        "import cti.multimodal.processor, cti.multimodal.qwen3_5.processor",
        "ctikernels",
        "cti.models",
        "cti.layers",
        "cti.forward",
        "cti.inference._execution._breakable_graph",
        "cti.inference._execution._graphs",
        "cti.inference._execution._sampling",
        "cti.inference._execution._speculative",
        "cti.inference._step",
        "cti.inference._execution.runner",
        "cti.plan",
        "cti.resolution",
    )


def test_input_values_load_without_image_preprocessing() -> None:
    # The public contracts and model layout load without image preprocessing.
    assert_not_loaded(
        "from cti.multimodal import ImageInputs, ImageSpan, RopePositionLayout\n"
        "from cti.multimodal.qwen3_5 import Qwen3_5RopeLayout",
        "transformers",
        "PIL",
        "cti.multimodal.processor",
        "cti.multimodal.qwen3_5.processor",
    )


def test_processor_selection_imports_a_processor_only_when_needed() -> None:
    assert_not_loaded(
        "from cti.multimodal.processor import create_input_processor\n"
        "assert create_input_processor('Qwen3ForCausalLM', 'unused',"
        " hf_config=None, limits=None) is None",
        "transformers",
        "PIL",
        "cti.multimodal.qwen3_5",
    )


def test_protocol_import_does_not_load_model_layouts_or_preprocessing() -> None:
    assert_not_loaded(
        "import cti.protocol",
        "cti.multimodal.qwen3_5",
        "transformers",
        "PIL",
    )
