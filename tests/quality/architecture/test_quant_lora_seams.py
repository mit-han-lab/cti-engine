"""Quantization, LoRA and MoE providers stay with their owners."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from types import MappingProxyType
from typing import ClassVar

import pytest

from tests.support.architecture import (
    Hit,
    Imports,
    Module,
    Names,
    Package,
    Rule,
    attribute_spellings,
    dotted_spellings,
    isinstance_targets,
    module_spellings,
)

_QUANT = r"(?i)fp8|float8|e4m3|e8m0|mxfp4|nvfp4|w\d+a\d+|scale_inv|weight_scale|(?<!un)quantiz"
_GEMMS = frozenset(
    {"torch.nn.functional.linear", "torch.matmul", "torch.mm", "torch.addmm", "torch.bmm"}
)
_PROJECTION = re.compile(r"\w*proj|\w*_head|gate|experts")
_RAW_WEIGHTS = frozenset({"weight", "bias", "weight_scale_inv"})
_LINEAR_KERNELS = re.compile(r"ctikernels\.linear(\.|$)")
_MOE_KERNELS = re.compile(r"ctikernels\.(moe|fused_topk|fused_moe)")
_PRODUCT_CLASS = re.compile(r"(?i)(fp8|fp4|quant|lora)\w*linear|linear\w*(fp8|fp4|quant|lora)")
_METHOD_CLASS = re.compile(r"(?i)method|quant|fp8|fp4|lora")
_METHOD_OWNERS = MappingProxyType(
    {"LinearMethod": "layers/linear_method.py", "ExpertsMethod": "layers/experts_method.py"}
)
# Format plugins: their methods, recipes, numerics and kernels.
_QUANT_PACKAGE = "layers/quant/"


def _raw_weights_and_gemms(package: Package, module: Module) -> Iterator[Hit]:
    for token in module.tokens:
        if token.kind not in {"name", "attribute"}:
            continue
        owner = token.chain.rpartition(".")[0].rpartition(".")[2]
        if token.ident in _RAW_WEIGHTS and _PROJECTION.fullmatch(owner):
            yield token.qualname, f"{owner}.{token.ident}"
        elif token.chain in _GEMMS:
            yield token.qualname, token.chain


def _moe_threading(package: Package, module: Module) -> Iterator[Hit]:
    if module.path.startswith("models/"):
        yield from Names(r"^(moe_backend|BaseMoeBackend|create_moe_backend)$")(package, module)
    yield from Names(r"^is_moe$")(package, module)


def _kernel_imports(package: Package, module: Module) -> Iterator[Hit]:
    if module.path.startswith(_QUANT_PACKAGE):
        return
    for scope, chain in Imports("ctikernels")(package, module):
        moe_kernel = _MOE_KERNELS.match(chain) and not module.path.startswith("moe/")
        if _LINEAR_KERNELS.match(chain) or moe_kernel:
            yield scope, chain


def _linear_structure(package: Package, module: Module) -> Iterator[Hit]:
    if module.path.startswith(_QUANT_PACKAGE):
        return
    for scope, node in module.nodes:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {module.ident(base) for base in node.bases}
        if bases & package.subclasses("LinearBase") and module.path != "layers/linear.py":
            yield scope, "LinearBase"
        if _PRODUCT_CLASS.search(node.name):
            yield scope, node.name
    if module.path not in _METHOD_OWNERS.values():
        for scope, chain in isinstance_targets(module):
            if _METHOD_CLASS.search(name := chain.rpartition(".")[2]):
                yield scope, name


def _quant_entry(package: Package, module: Module) -> Iterator[Hit]:
    """Imports of the format plugins, but the loader's of ``CheckpointFormats``."""

    if module.path.startswith(_QUANT_PACKAGE):
        return
    for scope, chain in Imports("cti.layers.quant")(package, module):
        if not (
            module.path == "models/weight.py" and chain == "cti.layers.quant.CheckpointFormats"
        ):
            yield scope, chain


def _plan_owners(package: Package, module: Module) -> Iterator[Hit]:
    if not module.path.startswith("layers/"):
        yield from Names(r"^physical_plan$")(package, module)
    for scope, node in module.nodes:
        if isinstance(node, ast.ClassDef):
            for root, owner in _METHOD_OWNERS.items():
                derived = {module.ident(base) for base in node.bases} & package.subclasses(root)
                if derived and module.path != owner and not module.path.startswith(_QUANT_PACKAGE):
                    yield scope, root


class QuantLoraSeams:
    """The Q rules, with their probes and ledgers."""

    # Q1 axis-agnostic packages name no LoRA; their quantization names fall under Q2. The
    # runner and its cost model build the adapters.
    Q1: ClassVar[Rule] = Rule(
        "Q1",
        Names(r"(?i)lora"),
        probes=(("kvcache/radix_cache.py", attribute_spellings("lora_rank")),),
        scope=(
            "attention/",
            "forward.py",
            "inference/",
            "kvcache/",
            "moe/",
            "multimodal/",
            "recurrent.py",
            "tokenizer/",
            "training/",
        ),
        owners=("inference/_execution/_hosting.py", "inference/_execution/runner.py"),
    )
    # Q2 quantization names stay in the format plugins.
    Q2: ClassVar[Rule] = Rule(
        "Q2",
        Names(_QUANT),
        probes=(("layers/linear.py", attribute_spellings("weight_scale_inv")),),
        owners=(_QUANT_PACKAGE,),
        ledger="",
    )
    # Q3 models read no projection weight and call no GEMM.
    Q3: ClassVar[Rule] = Rule(
        "Q3",
        _raw_weights_and_gemms,
        probes=(
            ("models/qwen3/model.py", dotted_spellings("torch.nn.functional.linear", "{}(_x, _w)")),
            (
                "models/qwen3/model.py",
                (
                    "self.o_proj.weight",
                    "_bias = self.lm_head.bias",
                    "from torch import matmul\nmatmul(_x, _w)",
                ),
            ),
        ),
        scope=("models/",),
        ledger="",
    )
    # Q4 the LoRA pipeline threaded through model constructors.
    Q4: ClassVar[Rule] = Rule(
        "Q4",
        Names(r"^(lora_adapters|LoRALinear|wrap)$|forward_lora"),
        probes=(("models/qwen3_moe/model.py", attribute_spellings("lora_adapters")),),
        scope=("models/",),
        ledger="",
    )
    # Q5 the MoE provider threaded through models, and is_moe anywhere.
    Q5: ClassVar[Rule] = Rule(
        "Q5",
        _moe_threading,
        probes=(
            ("models/qwen3/model.py", attribute_spellings("moe_backend")),
            ("worker/__init__.py", attribute_spellings("is_moe")),
        ),
        ledger="",
    )
    # Q6 linear kernels come in through layers only, MoE kernels through moe/.
    Q6: ClassVar[Rule] = Rule(
        "Q6",
        _kernel_imports,
        probes=(
            ("models/qwen3/model.py", module_spellings("ctikernels.linear.inference")),
            ("layers/moe.py", module_spellings("ctikernels.moe_impl")),
        ),
        ledger="""
layers/linear_method.py
    ctikernels.linear.Linear
layers/lora.py
    ctikernels.linear.lora.LoRAInferencePlan
    ctikernels.linear.lora.lora_projection
    ctikernels.linear.lora.try_prepare_lora_inference_plan
    ctikernels.linear.training.lora_projection_training
layers/prepare.py
    ctikernels.linear.release_cublas_workspaces
    ctikernels.linear.release_scratch
""",
    )
    # Q7 one linear class family: LinearBase subclasses in layers/linear.py, no product
    # classes such as Fp8RowLinear, no isinstance on methods outside their owners.
    Q7: ClassVar[Rule] = Rule(
        "Q7",
        _linear_structure,
        probes=(
            (
                "models/qwen3/model.py",
                dotted_spellings(
                    "cti.layers.linear.LinearRowParallel", "class _Probe({}):\n    pass"
                )
                + ("class Fp8RowLinear:\n    pass", "class LinearWithLoRA:\n    pass"),
            ),
            (
                "inference/_execution/runner.py",
                dotted_spellings("cti.layers.linear_method.Fp8Method", "isinstance(_method, {})"),
            ),
        ),
        ledger="",
    )
    # Q8 one physical-plan owner: physical_plan() in layers, method subclasses in their
    # modules.
    Q8: ClassVar[Rule] = Rule(
        "Q8",
        _plan_owners,
        probes=(
            ("models/qwen3/model.py", attribute_spellings("physical_plan")),
            (
                "layers/linear.py",
                dotted_spellings(
                    "cti.layers.linear_method.LinearMethod", "class _Probe({}):\n    pass"
                ),
            ),
        ),
        ledger="",
    )

    # Q9 the format plugins have one entry, the loader's CheckpointFormats.
    Q9: ClassVar[Rule] = Rule(
        "Q9",
        _quant_entry,
        probes=(
            (
                "inference/_execution/runner.py",
                module_spellings("cti.layers.quant.CheckpointFormats"),
            ),
            ("models/weight.py", module_spellings("cti.layers.quant.fp8_block")),
        ),
        ledger="",
    )

    @classmethod
    def rules(cls) -> tuple[Rule, ...]:
        return (cls.Q1, cls.Q2, cls.Q3, cls.Q4, cls.Q5, cls.Q6, cls.Q7, cls.Q8, cls.Q9)


@pytest.mark.parametrize("rule", QuantLoraSeams.rules(), ids=lambda rule: rule.name)
def test_positions_match_the_ledger(rule: Rule, package: Package) -> None:
    rule.check(package)


@pytest.mark.parametrize("rule", QuantLoraSeams.rules(), ids=lambda rule: rule.name)
def test_rule_catches_every_spelling(rule: Rule, package: Package) -> None:
    rule.check_probes(package)
