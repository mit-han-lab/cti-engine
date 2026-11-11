"""Facade, contract and config imports stay light in a fresh interpreter."""

from __future__ import annotations

import pytest

from tests.support.imports import assert_not_loaded

_MODEL_STACK = ("cti.models.base", "cti.layers", "ctikernels")


def test_llm_facade_loads_no_process_or_model_stack() -> None:
    assert_not_loaded(
        "import cti.llm",
        "cti.runtime",
        "cti.training._checkpoint",
        *_MODEL_STACK,
    )


def test_tokenizer_facade_loads_no_worker() -> None:
    assert_not_loaded("import cti.tokenizer", "cti.tokenizer.server", "zmq")


@pytest.mark.parametrize(
    "setup",
    (
        "import cti.training._protocol",
        "import cti.models",
        "import cti.attention",
    ),
)
def test_contracts_and_configs_load_no_model_stack(setup: str) -> None:
    assert_not_loaded(setup, *_MODEL_STACK)


def test_the_attention_facade_loads_its_families_on_first_resolution() -> None:
    """The family table names its resolvers; importing the facade loads none of them."""

    assert_not_loaded("import cti.attention", "cti.attention.mha", "cti.attention.fixed_query")


def test_options_parse_without_model_training_or_torch() -> None:
    setup = (
        "from cti.config import LaunchFile\n"
        "LaunchFile.parse({'model': {'path': 'm', 'weights': {'kind': 'random_weights'}},"
        " 'lora': {'targets': ['attention']}, 'training': {'trainable_plan': {},"
        " 'optimizer': {}, 'batching': {'layout': 'padded', 'policy': {'kind':"
        " 'static_batching', 'microbatch_size': 1, 'max_tokens_per_sample': 8}}}})"
    )
    assert_not_loaded(setup, "cti.models", "cti.training", "torch", "transformers")


def test_the_step_contract_is_a_value_leaf() -> None:
    """Loading ``_step`` loads neither torch nor either half of inference."""

    assert_not_loaded(
        "import cti.inference._step",
        "torch",
        "cti.inference._control",
        "cti.inference._execution",
    )


# Fails any CUDA probe and stops at the engine child's spawn, keeping the request it got.
_PARENT_UNTIL_SPAWN = """
import multiprocessing as mp

import torch


def probe(*_args, **_kwargs):
    raise AssertionError("the parent probed CUDA")


for name in ("is_available", "init", "device_count", "get_device_capability", "set_device"):
    setattr(torch.cuda, name, probe)
get_context = mp.get_context


class Spawned(Exception):
    pass


class Context:
    def Pipe(self, duplex):
        return get_context("spawn").Pipe(duplex=duplex)

    def Array(self, typecode, size, lock):
        return get_context("spawn").Array(typecode, size, lock=lock)

    def Process(self, **_kwargs):
        raise Spawned


mp.get_context = lambda _method=None: Context()
from cti.config import EngineOptions, ModelOptions, ParallelOptions
from cti.config import EngineTuning
from cti.llm import LLM
from cti.server.args import parse_args

try:
    {entry}
except Spawned:
    pass
else:
    raise AssertionError("the engine child was not spawned")
"""


@pytest.mark.parametrize(
    "entry",
    (
        'LLM(EngineOptions(model=ModelOptions(path="unused")), tuning=EngineTuning(attention_backend="fi"))',
        'LLM(EngineOptions(model=ModelOptions(path="unused"), parallel=ParallelOptions(num_ranks=2)))',
    ),
    ids=("llm", "llm-tensor-parallel"),
)
def test_python_facades_build_the_request_without_model_kernels_or_cuda(entry: str) -> None:
    assert_not_loaded(
        _PARENT_UNTIL_SPAWN.format(entry=entry),
        "cti.plan",
        "cti.resolution",
        "cti.worker",
        "cti.attention",
        "cti.moe",
        *_MODEL_STACK,
    )


def test_cli_builds_the_request_without_the_model_kernels_or_cuda() -> None:
    entry = 'LLM(parse_args(["--model.path", "unused"])[0])'

    assert_not_loaded(
        _PARENT_UNTIL_SPAWN.format(entry=entry),
        "cti.plan",
        "cti.resolution",
        "cti.worker",
        "cti.attention",
        "cti.moe",
        *_MODEL_STACK,
    )


def test_the_parent_loads_neither_the_plan_nor_torch() -> None:
    assert_not_loaded("import cti.llm, cti.runtime, cti.server", "cti.plan", "torch")


def test_the_runtime_loads_the_rank_only_when_it_spawns() -> None:
    assert_not_loaded(
        "import cti.runtime",
        "cti.runtime._core",
        "cti.worker",
        "cti.inference",
        "torch",
        "zmq",
    )


@pytest.mark.parametrize(
    "module",
    (
        "cti.runtime._protocol",
        "cti.runtime._training",
        "cti.utils",
        "cti.training._errors",
    ),
)
def test_leaf_modules_load_no_heavy_third_party_package(module: str) -> None:
    assert_not_loaded(f"import {module}", "torch", "huggingface_hub", "zmq")


def test_the_protocol_loads_neither_torch_nor_zmq() -> None:
    assert_not_loaded("import cti.protocol", "torch", "zmq", "cti.inference")
