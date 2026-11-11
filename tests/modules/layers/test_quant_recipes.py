"""Quantization recipes parse the real target checkpoints and refuse what they cannot serve,
by name."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch

from cti.layers import quant
from cti.layers.quant import CheckpointIndex
from cti.layers.quant.fp8_block import Fp8BlockRecipe
from cti.layers.quant.modelopt_mixed import ModelOptMixedRecipe
from tests.support import quant_checkpoints

_NVFP4 = "qwen3_8_27b_nvfp4"
_QWEN3_8_FP8 = "qwen3_8_27b_fp8"
_QWEN3_5_MOE_FP8 = "qwen3_5_35b_a3b_fp8"
_QWEN3_FP8 = "qwen3_8b_fp8"


def _checkpoint(stem: str) -> tuple[dict[str, Any], CheckpointIndex]:
    documents, headers = quant_checkpoints.load(stem)
    return documents, CheckpointIndex.from_headers(headers)


def _mutated(stem: str, edit: Any) -> dict[str, Any]:
    documents = copy.deepcopy(quant_checkpoints.load(stem)[0])
    edit(documents)
    return documents


@pytest.mark.parametrize(
    ("stem", "tensors"),
    [(_QWEN3_FP8, 651), (_QWEN3_8_FP8, 1606), (_NVFP4, 2194), (_QWEN3_5_MOE_FP8, 64196)],
)
def test_each_target_parses_into_its_recipe_and_matches_its_headers(
    stem: str, tensors: int
) -> None:
    documents, index = _checkpoint(stem)

    recipe = quant.parse(documents)
    recipe.verify(index)

    assert len(index.tensors) == tensors
    assert recipe.NAME == ("modelopt_mixed" if stem == _NVFP4 else "fp8_block")


def test_the_fp8_recipes_list_their_unconverted_modules_and_record_stale_ones() -> None:
    stale = {}
    for stem, listed in ((_QWEN3_FP8, 0), (_QWEN3_8_FP8, 882), (_QWEN3_5_MOE_FP8, 287)):
        documents, index = _checkpoint(stem)
        recipe = quant.parse(documents)
        assert len(recipe.unconverted) == listed
        stale[stem] = len(recipe.stale(index))

    # Qwen3.8-27B-FP8 lists 311 modules it does not have (unprefixed vision, in_proj_ba, ...).
    assert stale == {_QWEN3_FP8: 0, _QWEN3_8_FP8: 311, _QWEN3_5_MOE_FP8: 0}


def test_an_fp8_weight_inside_an_unconverted_module_is_refused_by_name() -> None:
    documents, index = _checkpoint(_QWEN3_8_FP8)
    recipe = quant.parse(documents)
    module = "model.language_model.layers.3.self_attn.q_proj"

    with pytest.raises(ValueError, match=rf"^{module}\.weight is F8_E4M3, but .* {module!r}$"):
        Fp8BlockRecipe((*recipe.unconverted, module)).verify(index)


def test_the_modelopt_recipe_reads_every_module_precision() -> None:
    documents, index = _checkpoint(_NVFP4)

    recipe = quant.parse(documents)

    assert isinstance(recipe, ModelOptMixedRecipe)
    assert Counter(recipe.algorithms.values()) == {"FP8": 208, "NVFP4": 193}
    assert recipe.algorithms["lm_head"] == "NVFP4"
    assert recipe.algorithms["model.language_model.layers.3.self_attn.q_proj"] == "FP8"
    assert recipe.excluded == ("mtp*", "mtp.layers.0*")


def _config(key: str, value: Any) -> Any:
    def edit(documents: dict[str, Any]) -> None:
        documents["config.json"]["quantization_config"][key] = value

    return edit


def _declared(key: str, value: Any) -> Any:
    def edit(documents: dict[str, Any]) -> None:
        documents["hf_quant_config.json"]["quantization"][key] = value

    return edit


def _move_target(documents: dict[str, Any]) -> None:
    groups = documents["config.json"]["quantization_config"]["config_groups"]
    module = "model.language_model.layers.3.self_attn.q_proj"
    groups["group_0"]["targets"].remove(module)
    groups["group_1"]["targets"].append(module)


def _relabel(document: str, path: tuple[str, ...]) -> Any:
    def edit(documents: dict[str, Any]) -> None:
        node = documents[document]
        for key in path:
            node = node[key]
        node["model.language_model.layers.3.self_attn.q_proj"] = {
            "quant_algo": "NVFP4",
            "group_size": 16,
        }

    return edit


def _drop(document: str, path: tuple[str, ...]) -> Any:
    def edit(documents: dict[str, Any]) -> None:
        node = documents[document]
        for key in path:
            node = node[key]
        del node["model.language_model.layers.0.mlp.up_proj"]

    return edit


_LAYERS = ("quantization_config", "quantized_layers")
_DECLARED_LAYERS = ("quantization", "quantized_layers")


@pytest.mark.parametrize(
    ("stem", "edit", "message"),
    [
        (_QWEN3_FP8, _config("quant_method", "awq"), r"no recipe claims .*'quant_method': 'awq'"),
        (_QWEN3_FP8, _config("weight_block_size", [64, 64]), r"weight_block_size=\[64, 64\]"),
        (_QWEN3_FP8, _config("activation_scheme", "static"), r"activation_scheme='static'"),
        (_QWEN3_FP8, _config("act_per_tensor", True), r"act_per_tensor=True: per-tensor dynamic"),
        (_QWEN3_FP8, _config("weight_per_tensor", True), r"weight_per_tensor=True"),
        (_QWEN3_FP8, _config("fmt", "e5m2"), r"fmt='e5m2'"),
        (_QWEN3_FP8, _config("ignored_layers", []), r"unknown keys \['ignored_layers'\]"),
        (
            _QWEN3_FP8,
            lambda d: d["config.json"]["quantization_config"].pop("weight_block_size"),
            r"lacks \['weight_block_size'\]",
        ),
        (_NVFP4, _config("quant_algo", "NVFP4"), r"'quant_algo': 'NVFP4'"),
        (_NVFP4, _config("kv_cache_scheme", {}), r"unknown keys \['kv_cache_scheme'\]"),
        (_NVFP4, _config("sparsity", None), r"unknown keys \['sparsity'\]"),
        (_NVFP4, _declared("kv_cache_quant_algo", "FP8"), r"kv_cache_quant_algo='FP8'"),
        (_NVFP4, _declared("awq_block_size", 64), r"unknown keys \['awq_block_size'\]"),
        (
            _NVFP4,
            lambda d: d["hf_quant_config.json"]["producer"].update(name="other"),
            r"producer name='other'",
        ),
        (_NVFP4, _config("ignore", ["mtp*"]), r"ignore \['mtp\*'\] differs from .*exclude_modules"),
        (
            _NVFP4,
            lambda d: (_config("ignore", "lm_head")(d), _declared("exclude_modules", "lm_head")(d)),
            r"ignore must be a list of module names, not 'lm_head'",
        ),
        (
            _NVFP4,
            lambda d: d["config.json"].pop("quantization_config"),
            r"hf_quant_config\.json declares a ModelOpt export that config\.json lacks",
        ),
        (
            _NVFP4,
            lambda d: d.pop("hf_quant_config.json"),
            r"needs hf_quant_config\.json",
        ),
        (_NVFP4, _move_target, r"layers\.3\.self_attn\.q_proj: the declarations disagree"),
        (_NVFP4, _relabel("config.json", _LAYERS), r"layers\.3\.self_attn\.q_proj: the decl"),
        (
            _NVFP4,
            _relabel("hf_quant_config.json", _DECLARED_LAYERS),
            r"layers\.3\.self_attn\.q_proj: the declarations disagree",
        ),
        (_NVFP4, _drop("config.json", _LAYERS), r"layers\.0\.mlp\.up_proj: the declarations"),
        (
            _NVFP4,
            _drop("hf_quant_config.json", _DECLARED_LAYERS),
            r"layers\.0\.mlp\.up_proj: the declarations disagree",
        ),
        (
            _NVFP4,
            lambda d: d["config.json"]["quantization_config"]["config_groups"]["group_1"][
                "weights"
            ].update(group_size=32),
            r"group_1: weights .* are neither static FP8 nor NVFP4",
        ),
    ],
)
def test_a_declaration_the_recipes_cannot_serve_is_refused_by_name(
    stem: str, edit: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        quant.parse(_mutated(stem, edit))


def test_the_modelopt_exclusions_compare_as_sets() -> None:
    documents = _mutated(_NVFP4, _config("ignore", ["mtp.layers.0*", "mtp*"]))

    assert quant.parse(documents).excluded == ("mtp*", "mtp.layers.0*")


def test_a_checkpoint_two_recipes_claim_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    class Twin(Fp8BlockRecipe):
        NAME = "twin"

    recipes = {**quant.RECIPES, "twin": quant.RecipeEntry(lambda: Twin)}
    monkeypatch.setattr(quant, "RECIPES", recipes)

    with pytest.raises(ValueError, match=r"claimed by several recipes: \['fp8_block', 'twin'\]"):
        quant.parse(quant_checkpoints.load(_QWEN3_FP8)[0])


def test_an_unquantized_checkpoint_has_no_recipe() -> None:
    assert quant.parse({"config.json": {"model_type": "qwen3"}}) is None
    assert {name: entry.load().NAME for name, entry in quant.RECIPES.items()} == {
        name: name for name in quant.RECIPES
    }


def test_the_modelopt_recipe_refuses_tensors_its_declarations_miss() -> None:
    documents, index = _checkpoint(_NVFP4)
    recipe = quant.parse(documents)
    algorithms = dict(recipe.algorithms)
    del algorithms["model.language_model.layers.0.mlp.up_proj"]

    up_proj = r"model\.language_model\.layers\.0\.mlp\.up_proj"
    with pytest.raises(ValueError, match=rf"^{up_proj}\.weight\w* is \w+, but no .* {up_proj}$"):
        ModelOptMixedRecipe(algorithms, recipe.excluded).verify(index)
    with pytest.raises(ValueError, match=r"^lm_head is listed as NVFP4 and excluded by 'lm\*'$"):
        ModelOptMixedRecipe(recipe.algorithms, ("lm*",)).verify(index)


def test_the_format_table_refuses_modelopt_routed_experts_by_name() -> None:
    # Qwen3.5-35B-A3B in ModelOpt's NVFP4: an expert's projection stops at the table, by name,
    # instead of an AttributeError.
    documents, headers = quant_checkpoints.load(_NVFP4)
    module = "model.language_model.layers.0.mlp.up_proj"

    with pytest.raises(ValueError, match=rf"^{module}: NVFP4 routed experts are not supported"):
        quant.CheckpointFormats(documents.get, headers, {module}, torch.device("cpu"))


def test_a_modelopt_checkpoint_names_its_recipe_as_untrainable(tmp_path: Path) -> None:
    from cti.models.weight import CheckpointWeights, SeededWeights

    quant_checkpoints.write_headers_only(_NVFP4, nvfp4 := tmp_path / "nvfp4")
    quant_checkpoints.write_headers_only(_QWEN3_FP8, fp8 := tmp_path / "fp8")
    cpu = torch.device("cpu")
    assert CheckpointWeights(str(nvfp4)).methods(cpu).untrainable == "modelopt_mixed"
    assert CheckpointWeights(str(fp8)).methods(cpu).untrainable is None
    assert SeededWeights(0).methods(cpu).untrainable is None
