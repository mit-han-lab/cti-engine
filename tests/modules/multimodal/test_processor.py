from __future__ import annotations

import importlib
import inspect
import pkgutil
from unittest import mock

import pytest

import cti.multimodal
import cti.multimodal.processor as processor_module
from cti.multimodal import ImageLimits, rope_layout
from cti.multimodal.processor import InputProcessor, create_input_processor


def test_the_architecture_table_holds_every_processor_and_the_layout_it_declares() -> None:
    for module in pkgutil.walk_packages(cti.multimodal.__path__, "cti.multimodal."):
        importlib.import_module(module.name)
    processors, pending = [], [InputProcessor]
    while pending:
        processor = pending.pop()
        pending.extend(processor.__subclasses__())
        if processor.__module__.startswith("cti.") and not inspect.isabstract(processor):
            processors.append(processor)

    assert processors
    table = processor_module._ARCHITECTURES
    assert {processor_module._processor_class(name) for name in table} == set(processors)
    for name, row in table.items():
        assert processor_module._processor_class(name).ROPE_LAYOUT is rope_layout(
            row.rope_layout[1]
        )


@pytest.mark.parametrize(
    "architecture", ("Qwen3ForCausalLM", "Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM")
)
def test_text_only_architectures_have_no_input_processor(architecture: str) -> None:
    processor = create_input_processor(
        architecture,
        "unused",
        hf_config=mock.sentinel.hf_config,
        limits=ImageLimits(max_images=2, max_image_tokens=1024),
    )

    assert processor is None


@pytest.mark.parametrize(
    ("limits", "reason"),
    [({"max_images": 0}, "max_images=0"), ({"max_image_tokens": -1}, "max_image_tokens=-1")],
)
def test_image_limits_reject_invalid_bounds(limits: dict[str, object], reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        ImageLimits(**{"max_images": 4, "max_image_tokens": 16, **limits})
