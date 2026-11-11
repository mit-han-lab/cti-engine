"""What each rank of tiny Qwen3, Qwen3.5, Qwen3-MoE and Qwen3.5-MoE (experts split by expert
parallelism) and the DFlash draft at TP 1, 2 and 4 places, as the text ``tests/modules/models/goldens/<model>.txt``
records.

Regenerate the goldens on purpose with ``python -m tests.support.tp_goldens``.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import pytest
import torch
from transformers import Qwen3Config, Qwen3MoeConfig

from cti.distributed import Topology
from cti.layers.linear import LinearBase
from cti.models.dflash import DFlashConfig, DFlashDraftModel
from tests.support.dflash import TINY_DRAFT_CONFIG, qwen3_hf_config, target_engine_config
from tests.support.moe import ReferenceMoe
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5 import TINY_QWEN3_5_CONFIG, load_hf_config
from tests.support.qwen3_5_engine import bind_reference_kernels
from tests.support.qwen3_5_moe import TINY_QWEN3_5_MOE_CONFIG
from tests.support.tiny_configs import TINY_GEOMETRY
from tests.support.tp_shards import build_model, default_dtype, site

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cti.layers.base import BaseOP
    from cti.layers.placement import WeightShard


class PlacementGoldens:
    """The goldens' models, tensor-parallel sizes and head counts, and the placement lines
    they render."""

    FOLDER = REPO_ROOT / "tests" / "modules" / "models" / "goldens"
    MODELS = ("qwen3", "qwen3_5", "qwen3_moe", "qwen3_5_moe", "dflash")
    TP_SIZES = (1, 2, 4)
    # Two KV heads, so TP 4 replicates each on two ranks; the vocabulary pads at TP 2 and 4.
    HEADS = MappingProxyType({"vocab_size": 67, "num_attention_heads": 4, "num_key_value_heads": 2})
    # The draft's KV head counts: one, which every rank replicates, and two.
    DRAFT_KV_HEADS = (1, 2)

    def render(self, folder: Path) -> dict[str, list[str]]:
        """Every model's placement lines under a ``[tp n, rank r]`` header per rank."""

        moe = {"moe_backend": ReferenceMoe()}
        configs = {
            "qwen3": (self._qwen3(), {}),
            "qwen3_5": (load_hf_config(self._qwen3_5(TINY_QWEN3_5_CONFIG), folder / "qwen3_5"), {}),
            "qwen3_moe": (self._qwen3_moe(), moe),
            "qwen3_5_moe": (
                load_hf_config(self._qwen3_5(TINY_QWEN3_5_MOE_CONFIG), folder / "qwen3_5_moe"),
                moe,
            ),
        }
        result: dict[str, list[str]] = {name: [] for name in self.MODELS}
        with pytest.MonkeyPatch.context() as monkeypatch:
            bind_reference_kernels(monkeypatch)
            for size in self.TP_SIZES:
                for name, (config, options) in configs.items():
                    topologies = self._topologies(size)
                    ranks = [build_model(config, topology=t, **options) for t in topologies]
                    result[name] += self._ranks(f"tp {size}", ranks)
            for kv_heads in self.DRAFT_KV_HEADS:
                draft = self._draft(kv_heads)
                for size in self.TP_SIZES:
                    ranks = [self._build_draft(draft, t) for t in self._topologies(size)]
                    result["dflash"] += self._ranks(f"kv {kv_heads}, tp {size}", ranks)
        return result

    @staticmethod
    def _topologies(size: int) -> list[Topology]:
        return [Topology(rank, size) for rank in range(size)]

    @staticmethod
    def _ranks(header: str, models: list[BaseOP]) -> list[str]:
        """Each rank's lines under ``[<header>, rank r]``; a window that n ranks hold alike
        reads ``*n@i`` on the i-th of them."""

        placements = [
            [line.split(" ") for line in PlacementGoldens._placement(model)] for model in models
        ]
        held = Counter(
            (row, column, word)
            for lines in placements
            for row, words in enumerate(lines)
            for column, word in enumerate(words)
            if word.startswith(("rows[", "cols["))
        )
        seen: Counter[tuple[int, int, str]] = Counter()
        result = []
        for rank, lines in enumerate(placements):
            result.append(f"[{header}, rank {rank}]")
            for row, words in enumerate(lines):
                for column, word in enumerate(words):
                    key = (row, column, word)
                    if held[key] > 1:
                        words[column] = f"{word}*{held[key]}@{seen[key]}"
                        seen[key] += 1
                result.append(" ".join(words))
        return result

    @staticmethod
    def _draft(kv_heads: int) -> DFlashConfig:
        # One full layer of four query heads.
        document = {
            **TINY_DRAFT_CONFIG,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": kv_heads,
            "layer_types": ["full_attention"],
        }
        return DFlashConfig.from_hf(document, qwen3_hf_config(document), target_engine_config())

    @staticmethod
    def _build_draft(config: DFlashConfig, topology: Topology) -> DFlashDraftModel:
        with torch.device("meta"), default_dtype(torch.float32):
            return DFlashDraftModel(config, round_width=config.block_size, site=site(topology))

    def _qwen3(self) -> Qwen3Config:
        # Tied like Qwen3-0.6B.
        return Qwen3Config(
            **{**TINY_GEOMETRY, **self.HEADS, "num_hidden_layers": 1},
            tie_word_embeddings=True,
            architectures=["Qwen3ForCausalLM"],
        )

    def _qwen3_moe(self) -> Qwen3MoeConfig:
        # Eight query heads over two KV heads and eight experts: TP 4 replicates each KV head
        # on two ranks and places two whole experts per rank.
        return Qwen3MoeConfig(
            **{**TINY_GEOMETRY, **self.HEADS, "num_hidden_layers": 1, "num_attention_heads": 8},
            moe_intermediate_size=16,
            num_experts=8,
            num_experts_per_tok=2,
            architectures=["Qwen3MoeForCausalLM"],
        )

    def _qwen3_5(self, tiny: Mapping[str, Any]) -> dict[str, object]:
        # Four Gated DeltaNet key heads, each read by two value heads; the MoE model's eight
        # experts place two whole ones per rank at TP 4.
        return {
            **tiny,
            "text_config": {
                **tiny["text_config"],
                **self.HEADS,
                "linear_num_key_heads": 4,
                "linear_num_value_heads": 8,
            },
        }

    @staticmethod
    def _window(shard: WeightShard) -> str:
        padding = "" if shard.padded_size is None else f"/{shard.padded_size}"
        return f"{('rows', 'cols')[shard.axis]}[{shard.start}:{shard.stop}]{padding}"

    @staticmethod
    def _placement(model: BaseOP) -> list[str]:
        """One line per parameter: name, local shape, the global experts a stacked expert tensor
        holds, the checkpoint windows it loads (``/n``: zero-padded to n) and, after ``|``, a linear's output slices; then the local mixer
        geometry the runner sizes the KV and recurrent-state pools from."""

        ops = dict(model.named_ops())
        lines = []
        for param in model.params():
            windows = map(PlacementGoldens._window, param.windows)
            experts = (
                ()
                if param.experts is None
                else (f"experts[{param.experts.start}:{param.experts.stop}]",)
            )
            words = [param.path, str(param.shape), *experts, *windows]
            owner = ops[param.path.rpartition(".")[0]]
            if isinstance(owner, LinearBase) and param.path.endswith(".weight"):
                words.append("|")
                words += (f"{s.name}[{s.offset}:{s.offset + s.size}]" for s in owner.output_slices)
            lines.append(" ".join(words))
        lines += (
            f"attention {index}: {layer.num_qo_heads} query, "
            f"{layer.num_kv_heads} KV heads of {layer.head_dim}"
            for index, layer in enumerate(model.iter_attention_layers())
        )
        recurrent = getattr(model, "iter_recurrent_layers", tuple)()
        lines += (
            f"gated delta net {index}: {layer.num_key_heads} key, "
            f"{layer.num_value_heads} value heads; state {state}, conv {history} x {channels}"
            for index, layer in enumerate(recurrent)
            for (_, state, _), (_, (history, channels), _) in (layer.state_spec.tensors,)
        )
        return lines


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        for model, lines in PlacementGoldens().render(Path(folder)).items():
            path = PlacementGoldens.FOLDER / f"{model}.txt"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
