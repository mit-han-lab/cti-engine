from __future__ import annotations

import re
import sys
import threading
from collections.abc import Callable, Mapping
from fnmatch import fnmatch
from pathlib import Path
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
from torch.overrides import TorchFunctionMode

import cti.models._download as download_module
import cti.models.weight as weight_module
from cti.config import Checkpoint as CheckpointOption
from cti.distributed import Topology
from cti.layers import LinearQKVMerged, LinearRowParallel, Param
from cti.models.mapping import CheckpointMapping, CheckpointRule, MergeRule, Reshape, RowSplit
from cti.models.weight import open_weights
from tests.support.tp_shards import site

_PROJECTION_MERGES = (
    MergeRule("qkv_proj", ("q_proj", "k_proj", "v_proj")),
    MergeRule("gate_up_proj", ("gate_proj", "up_proj")),
)


class _Intercept(TorchFunctionMode):
    """Run each handled torch function's handler in its place while the mode is active.

    A mode rather than a patched ``torch.empty``: torch caches its tensor constructors the
    first time a device context dispatches one, and a patched constructor cached then would
    keep every later ``with torch.device(...)`` from placing tensors, past this test.
    """

    def __init__(self, handlers: Mapping[Callable, Callable]) -> None:
        super().__init__()
        self._handlers = handlers

    def __torch_function__(self, func, types, args=(), kwargs=None):
        return self._handlers.get(func, func)(*args, **(kwargs or {}))


@pytest.fixture
def final_cuda_allocations():
    """Observe final CUDA storage while executing copies with real CPU tensors."""
    target = torch.device("cuda", 0)
    allocations = []

    def observe_transfer(self, *args, **kwargs):
        if args and args[0] == target:
            assert self.device.type == "cpu"
            result = self.clone()
            allocations.append(result)
            return result
        return self.to(*args, **kwargs)

    def observe_allocation(*args, **kwargs):
        if kwargs.get("device") == target:
            result = torch.empty(*args, **{**kwargs, "device": "cpu"})
            allocations.append(result)
            return result
        return torch.empty(*args, **kwargs)

    with _Intercept({torch.Tensor.to: observe_transfer, torch.empty: observe_allocation}):
        yield target, allocations


def _record_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every checkpoint tensor the loader reads."""
    reads: list[str] = []
    open_checkpoint = safetensors.safe_open

    class RecordingCheckpoint:
        def __init__(self, handle) -> None:
            self._handle = handle

        def __enter__(self) -> RecordingCheckpoint:
            self._handle.__enter__()
            return self

        def __exit__(self, *exc_info) -> None:
            self._handle.__exit__(*exc_info)

        def keys(self) -> list[str]:
            return self._handle.keys()

        def get_tensor(self, name: str) -> torch.Tensor:
            reads.append(name)
            return self._handle.get_tensor(name)

    monkeypatch.setattr(
        safetensors,
        "safe_open",
        lambda path, *, framework, device: RecordingCheckpoint(
            open_checkpoint(path, framework=framework, device=device)
        ),
    )
    return reads


def _declare(tensors: dict[str, torch.Tensor], dtype: torch.dtype) -> tuple[Param, ...]:
    """Params shaped like the expected runtime tensors."""
    return tuple(Param(name, tuple(t.shape), dtype) for name, t in tensors.items())


def _load(
    folder: Path,
    *,
    mapping: CheckpointMapping,
    params: tuple[Param, ...],
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    return dict(
        weight_module.load_weight(
            str(folder),
            torch.device("cpu") if device is None else device,
            mapping=mapping,
            params=params,
        )
    )


@pytest.mark.parametrize("rank,size", [(0, 1), (1, 2), (3, 4)])
def test_weight_loading_transfers_only_final_sharded_and_merged_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    final_cuda_allocations,
    rank: int,
    size: int,
) -> None:
    """Loading must not leave temporary checkpoint/merge/cast allocations on the GPU."""
    num_kv_heads = 2
    at = site(Topology(rank, size))
    with torch.device("meta"):
        qkv_proj = LinearQKVMerged(
            hidden_size=4,
            head_dim=2,
            num_qo_heads=4,
            num_kv_heads=num_kv_heads,
            has_bias=False,
            site=at,
        )
        down_proj = LinearRowParallel(8, 4, has_bias=False, site=at)
    windows = {
        param.path: param.windows
        for param in (
            *qkv_proj.params(prefix="model.qkv_proj"),
            *down_proj.params(prefix="model.down_proj"),
        )
    }
    # Distinct integers below 256 stay exact in bf16, so the cast cannot hide a wrong shard.
    q = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    k = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 32
    v = k + 16
    down = torch.arange(32, dtype=torch.float32).reshape(4, 8) + 64
    norm = torch.arange(4, dtype=torch.float32) + 96
    safetensors.torch.save_file(
        {"model.q_proj.weight": q, "model.k_proj.weight": k}, tmp_path / "part-1.safetensors"
    )
    safetensors.torch.save_file(
        {"model.v_proj.weight": v, "model.down_proj.weight": down, "model.norm.weight": norm},
        tmp_path / "part-2.safetensors",
    )
    open_checkpoint = safetensors.safe_open
    opened = []

    def cpu_checkpoint(path, *, framework, device):
        opened.append(path)
        assert str(device) == "cpu", "checkpoint temporaries must stay on CPU"
        return open_checkpoint(path, framework=framework, device=device)

    monkeypatch.setattr(safetensors, "safe_open", cpu_checkpoint)
    target, allocations = final_cuda_allocations
    kv_parts = min(size, num_kv_heads)
    kv_rank = rank * num_kv_heads // size if size > num_kv_heads else rank
    expected = {
        "model.qkv_proj.weight": torch.cat(
            (
                q.chunk(size)[rank],
                k.chunk(kv_parts)[kv_rank],
                v.chunk(kv_parts)[kv_rank],
            )
        ),
        "model.down_proj.weight": down.chunk(size, dim=1)[rank],
        "model.norm.weight": norm,
    }
    actual = _load(
        tmp_path,
        mapping=CheckpointMapping(merges=_PROJECTION_MERGES),
        params=tuple(
            Param(name, tuple(t.shape), torch.bfloat16, windows.get(name, ()))
            for name, t in expected.items()
        ),
        device=target,
    )
    assert len(opened) == 2
    assert actual.keys() == expected.keys()
    assert len(allocations) == len(expected)
    for name, tensor in actual.items():
        torch.testing.assert_close(tensor, expected[name].bfloat16(), rtol=0, atol=0)
        assert tensor.is_contiguous()
    assert {t.data_ptr() for t in actual.values()} == {t.data_ptr() for t in allocations}


def test_weight_loading_stacks_experts_before_device_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    final_cuda_allocations,
) -> None:
    weights = {
        f"model.experts.{expert}.{projection}.weight": torch.full(shape, value)
        for expert in range(2)
        for projection, shape, value in (
            ("gate_proj", (8, 4), 10.0 + expert),
            ("up_proj", (8, 4), 20.0 + expert),
            ("down_proj", (4, 8), 30.0 + expert),
        )
    }
    safetensors.torch.save_file(weights, tmp_path / "experts.safetensors")
    target, allocations = final_cuda_allocations
    open_checkpoint = safetensors.safe_open

    def cpu_checkpoint(path, *, framework, device):
        assert str(device) == "cpu"
        return open_checkpoint(path, framework=framework, device=device)

    monkeypatch.setattr(safetensors, "safe_open", cpu_checkpoint)
    expected_gate_up = torch.stack(
        [
            torch.cat(
                [weights[f"model.experts.{i}.{name}.weight"] for name in ("gate_proj", "up_proj")]
            )
            for i in range(2)
        ]
    )
    expected_down = torch.stack([weights[f"model.experts.{i}.down_proj.weight"] for i in range(2)])
    actual = _load(
        tmp_path,
        mapping=CheckpointMapping(merges=_PROJECTION_MERGES, stack_experts=True),
        params=_declare(
            {
                "model.experts.gate_up_proj": expected_gate_up,
                "model.experts.down_proj": expected_down,
            },
            torch.float32,
        ),
        device=target,
    )
    assert set(actual) == {"model.experts.gate_up_proj", "model.experts.down_proj"}
    assert len(allocations) == 2
    assert {t.data_ptr() for t in actual.values()} == {t.data_ptr() for t in allocations}
    torch.testing.assert_close(
        actual["model.experts.gate_up_proj"], expected_gate_up, rtol=0, atol=0
    )
    torch.testing.assert_close(actual["model.experts.down_proj"], expected_down, rtol=0, atol=0)


def test_weight_loading_stages_every_part_through_one_buffer_it_frees_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = {
        f"model.experts.{expert}.{projection}.weight": torch.full(shape, 10.0 * expert + index)
        for expert in range(2)
        for index, (projection, shape) in enumerate(
            (("gate_proj", (8, 4)), ("up_proj", (8, 4)), ("down_proj", (4, 8)))
        )
    }
    safetensors.torch.save_file(weights, tmp_path / "experts.safetensors")
    stagings: list[weight_module._HostStaging] = []
    staged: list[int] = []

    class RecordingStaging(weight_module._HostStaging):
        def __init__(self) -> None:
            super().__init__()
            stagings.append(self)

        def stage(self, part: torch.Tensor) -> torch.Tensor:
            result = super().stage(part)
            staged.append(self._buffer.data_ptr())
            return result

    monkeypatch.setattr(weight_module, "_HostStaging", RecordingStaging)
    gate_up = torch.stack(
        [
            torch.cat([weights[f"model.experts.{i}.{p}.weight"] for p in ("gate_proj", "up_proj")])
            for i in range(2)
        ]
    )
    down = torch.stack([weights[f"model.experts.{i}.down_proj.weight"] for i in range(2)])
    loading = weight_module.load_weight(
        str(tmp_path),
        torch.device("cpu"),
        mapping=CheckpointMapping(merges=_PROJECTION_MERGES, stack_experts=True),
        params=_declare(
            {"model.experts.gate_up_proj": gate_up, "model.experts.down_proj": down},
            torch.float32,
        ),
    )
    actual = {}
    for name, tensor in loading:
        assert stagings[0]._buffer is not None
        actual[name] = tensor.clone()

    # Six equal parts pass through one buffer, which is gone once the last tensor is out.
    assert len(stagings) == 1 and len(staged) == 6 and len(set(staged)) == 1
    assert stagings[0]._buffer is None
    torch.testing.assert_close(actual["model.experts.gate_up_proj"], gate_up, rtol=0, atol=0)
    torch.testing.assert_close(actual["model.experts.down_proj"], down, rtol=0, atol=0)


def test_weight_loading_reads_only_the_experts_in_its_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shapes = {"gate_proj": (2, 4), "up_proj": (2, 4), "down_proj": (4, 2)}
    weights = {
        (expert, name): torch.full(shape, 10.0 * expert + index)
        for expert in range(4)
        for index, (name, shape) in enumerate(shapes.items())
    }
    safetensors.torch.save_file(
        {f"model.experts.{e}.{name}.weight": t for (e, name), t in weights.items()},
        tmp_path / "experts.safetensors",
    )
    reads = _record_reads(monkeypatch)
    local = (2, 3)
    expected = {
        "model.experts.gate_up_proj": torch.stack(
            [torch.cat([weights[e, "gate_proj"], weights[e, "up_proj"]]) for e in local]
        ),
        "model.experts.down_proj": torch.stack([weights[e, "down_proj"] for e in local]),
    }

    actual = _load(
        tmp_path,
        mapping=CheckpointMapping(merges=_PROJECTION_MERGES, stack_experts=True),
        params=tuple(
            Param(name, tuple(t.shape), torch.float32, experts=range(2, 4))
            for name, t in expected.items()
        ),
    )

    assert sorted(reads) == sorted(
        f"model.experts.{e}.{name}.weight" for e in local for name in shapes
    )
    assert actual.keys() == expected.keys()
    for name, tensor in actual.items():
        assert torch.equal(tensor, expected[name]), name


@pytest.mark.parametrize(
    "mismatch,message",
    [
        ("rows", r"model\.experts\.gate_up_proj: expert 1 has 1 rows, expected 2"),
        ("trailing_dimensions", r"model\.experts\.gate_up_proj: part .*\(2, 4\) does not match"),
        ("dtype", r"model\.experts\.gate_up_proj: part torch\.float16 .* does not match"),
    ],
)
def test_stacked_experts_are_validated_before_allocation(
    tmp_path: Path, mismatch: str, message: str
) -> None:
    second = {
        "rows": torch.ones(1, 3),
        "trailing_dimensions": torch.ones(2, 4),
        "dtype": torch.ones(2, 3, dtype=torch.float16),
    }[mismatch]
    safetensors.torch.save_file(
        {
            "model.experts.0.gate_up_proj.weight": torch.ones(2, 3),
            "model.experts.1.gate_up_proj.weight": second,
        },
        tmp_path / "experts.safetensors",
    )

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("invalid groups must fail before device allocation")

    with _Intercept({torch.empty: unexpected_allocation}), pytest.raises(ValueError, match=message):
        _load(
            tmp_path,
            mapping=CheckpointMapping(stack_experts=True),
            params=(Param("model.experts.gate_up_proj", (2, 2, 3), torch.float32),),
        )


def test_weight_loading_casts_each_tensor_to_the_dtype_its_target_declares(tmp_path: Path) -> None:
    # 1 + 2**-10 rounds to 1 in bf16 but is exact in fp16 and fp32.
    value = torch.full((4,), 1 + 2**-10)
    names = ("model.norm.weight", "model.layers.0.linear_attn.A_log", "model.scale")
    safetensors.torch.save_file(
        {name: value.clone() for name in names}, tmp_path / "model.safetensors"
    )
    dtypes = dict(zip(names, (torch.bfloat16, torch.float32, torch.float16), strict=True))
    params = tuple(Param(name, (4,), dtype) for name, dtype in dtypes.items())

    actual = _load(tmp_path, mapping=CheckpointMapping(), params=params)

    assert {name: tensor.dtype for name, tensor in actual.items()} == dtypes
    for name, dtype in dtypes.items():
        assert torch.equal(actual[name], value.to(dtype))
    assert torch.equal(actual["model.layers.0.linear_attn.A_log"], value)


def test_weight_loading_reports_each_shard_as_its_read_begins(tmp_path: Path) -> None:
    names = ("model.a", "model.b")
    for index, name in enumerate(names):
        safetensors.torch.save_file(
            {name: torch.ones(4)}, tmp_path / f"model-{index + 1:05d}-of-00002.safetensors"
        )
    params = tuple(Param(name, (4,), torch.float32) for name in names)
    seen: list[object] = []

    loaded = weight_module.load_weight(
        str(tmp_path),
        torch.device("cpu"),
        mapping=CheckpointMapping(),
        params=params,
        progress=seen.append,
    )
    for name, _ in loaded:
        seen.append(name)

    assert seen == [
        "loading weights: shard 1/2 (model-00001-of-00002.safetensors)",
        "model.a",
        "loading weights: shard 2/2 (model-00002-of-00002.safetensors)",
        "model.b",
    ]


@pytest.mark.parametrize(
    ("checkpoint_dtype", "target_dtype"),
    [(torch.float8_e4m3fn, torch.bfloat16), (torch.bfloat16, torch.float8_e4m3fn)],
)
def test_weight_loading_never_converts_to_or_from_a_quantized_dtype(
    tmp_path: Path, checkpoint_dtype: torch.dtype, target_dtype: torch.dtype
) -> None:
    safetensors.torch.save_file(
        {"model.q_proj.weight": torch.ones(128, 128, dtype=checkpoint_dtype)},
        tmp_path / "model.safetensors",
    )
    params = (Param("model.q_proj.weight", (128, 128), target_dtype),)

    with pytest.raises(
        ValueError,
        match=rf"^model\.q_proj\.weight: checkpoint dtype {checkpoint_dtype} cannot load as "
        rf"{target_dtype}$",
    ):
        _load(tmp_path, mapping=CheckpointMapping(), params=params)


def test_weight_loading_keeps_a_quantized_weight_exact_and_merges_its_scale_with_it(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    rows = {"q_proj": 256, "k_proj": 128, "v_proj": 128}
    weights = {name: torch.randn(n, 256).to(torch.float8_e4m3fn) for name, n in rows.items()}
    scales = {name: torch.rand(n // 128, 2, dtype=torch.bfloat16) for name, n in rows.items()}
    safetensors.torch.save_file(
        {f"model.{name}.weight": tensor for name, tensor in weights.items()}
        | {f"model.{name}.weight_scale_inv": tensor for name, tensor in scales.items()},
        tmp_path / "model.safetensors",
    )
    params = (
        Param("model.qkv_proj.weight", (512, 256), torch.float8_e4m3fn),
        # A BF16 scale widens to FP32 exactly.
        Param("model.qkv_proj.weight_scale_inv", (4, 2), torch.float32),
    )

    actual = _load(tmp_path, mapping=CheckpointMapping(merges=_PROJECTION_MERGES), params=params)

    weight = actual["model.qkv_proj.weight"]
    assert weight.dtype == torch.float8_e4m3fn
    assert torch.equal(weight.view(torch.uint8), torch.cat([*weights.values()]).view(torch.uint8))
    assert torch.equal(
        actual["model.qkv_proj.weight_scale_inv"], torch.cat([*scales.values()]).float()
    )


@pytest.mark.parametrize(
    "transform",
    [RowSplit((64, 64)), Reshape(lambda tensor: tensor.chunk(2))],
    ids=["unaligned", "callable"],
)
def test_a_rule_that_cannot_cut_scale_blocks_names_itself(transform) -> None:
    mapping = CheckpointMapping(
        rules=(CheckpointRule("fused.weight", ("a.weight", "b.weight"), transform),)
    )

    with pytest.raises(
        ValueError,
        match=r"^model\.fused\.weight_scale_inv: rule 'fused\.weight' cannot cut 128-row blocks$",
    ):
        mapping.split("model.fused.weight_scale_inv", torch.ones(1, 2), 128)


def test_weight_loading_applies_model_rules_before_merging(tmp_path: Path) -> None:
    hidden = 4
    # Two heads, each storing its two query rows before its two gate rows.
    query_gate = torch.arange(8 * hidden, dtype=torch.float32).reshape(8, hidden)
    key = torch.full((2, hidden), 100.0)
    value = torch.full((2, hidden), 200.0)
    conv = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1)
    safetensors.torch.save_file(
        {
            "layers.0.self_attn.q_proj.weight": query_gate,
            "layers.0.self_attn.k_proj.weight": key,
            "layers.0.self_attn.v_proj.weight": value,
            "layers.0.linear_attn.conv1d.weight": conv,
        },
        tmp_path / "model.safetensors",
    )
    transformed: list[torch.Tensor] = []

    def split_query_gate(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transformed.append(tensor)
        heads = tensor.view(2, 2, 2, hidden)
        return heads[:, 0].reshape(4, hidden), heads[:, 1].reshape(4, hidden)

    mapping = CheckpointMapping(
        rules=(
            CheckpointRule(
                "self_attn.q_proj.weight",
                ("self_attn.q_proj.weight", "self_attn.output_gate.weight"),
                Reshape(split_query_gate),
            ),
            CheckpointRule("linear_attn.conv1d.weight", ("linear_attn.gdn.conv_weight",)),
        ),
        merges=(MergeRule("qkv_gate_proj", ("q_proj", "k_proj", "v_proj", "output_gate")),),
    )
    heads = query_gate.view(2, 2, 2, hidden)
    expected = {
        "layers.0.self_attn.qkv_gate_proj.weight": torch.cat(
            (heads[:, 0].reshape(4, hidden), key, value, heads[:, 1].reshape(4, hidden))
        ),
        "layers.0.linear_attn.gdn.conv_weight": conv,
    }

    actual = _load(tmp_path, mapping=mapping, params=_declare(expected, torch.float32))

    assert actual.keys() == expected.keys()
    for name, tensor in actual.items():
        assert torch.equal(tensor, expected[name]), name
    assert [tensor.device.type for tensor in transformed] == ["cpu"]


@pytest.mark.parametrize("rank", [0, 1])
def test_weight_loading_selects_rank_windows_of_rule_split_parts(tmp_path: Path, rank: int) -> None:
    hidden = 4
    # Two query heads, each storing its two query rows before its two gate rows.
    query_gate = torch.arange(8 * hidden, dtype=torch.float32).reshape(8, hidden)
    key = torch.full((2, hidden), 100.0)
    value = torch.full((2, hidden), 200.0)
    safetensors.torch.save_file(
        {"attn.q_proj.weight": query_gate, "attn.k_proj.weight": key, "attn.v_proj.weight": value},
        tmp_path / "model.safetensors",
    )
    with torch.device("meta"):
        projection = LinearQKVMerged(
            hidden_size=hidden,
            head_dim=2,
            num_qo_heads=2,
            num_kv_heads=1,
            has_bias=False,
            with_output_gate=True,
            site=site(Topology(rank, 2)),
        )
    heads = query_gate.view(2, 2, 2, hidden)

    def split_query_gate(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = tensor.view(2, 2, 2, hidden)
        return rows[:, 0].reshape(4, hidden), rows[:, 1].reshape(4, hidden)

    mapping = CheckpointMapping(
        rules=(
            CheckpointRule(
                "q_proj.weight", ("q_proj.weight", "output_gate.weight"), Reshape(split_query_gate)
            ),
        ),
        merges=(MergeRule("qkv_gate_proj", ("q_proj", "k_proj", "v_proj", "output_gate")),),
    )

    actual = _load(
        tmp_path,
        mapping=mapping,
        params=projection.params(prefix="attn.qkv_gate_proj"),
    )

    # This rank's query head and its gate rows; the single KV head is replicated.
    expected = torch.cat((heads[rank, 0], key, value, heads[rank, 1]))
    assert torch.equal(actual["attn.qkv_gate_proj.weight"], expected)


def test_weight_loading_renames_then_skips_without_reading_skipped_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    norm = torch.arange(4, dtype=torch.float32)
    safetensors.torch.save_file(
        {
            "model.language_model.norm.weight": norm,
            "model.visual.patch_embed.proj.weight": torch.ones(2, 2),
            "mtp.fc.weight": torch.ones(2, 2),
        },
        tmp_path / "model.safetensors",
    )
    reads = _record_reads(monkeypatch)
    mapping = CheckpointMapping(
        prefix_renames=(("model.language_model.", "model."), ("model.visual.", "visual.")),
        skip_prefixes=("visual.", "mtp."),
    )

    actual = _load(
        tmp_path, mapping=mapping, params=_declare({"model.norm.weight": norm}, norm.dtype)
    )

    assert actual.keys() == {"model.norm.weight"}
    assert torch.equal(actual["model.norm.weight"], norm)
    assert reads == ["model.language_model.norm.weight"]


def test_weight_loading_drops_an_optional_tensor_only_when_the_model_lacks_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedding = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    head = {"lm_head.weight": embedding + 8, "lm_head.weight_scale_inv": torch.ones(1, 1)}
    safetensors.torch.save_file(
        {"model.embed_tokens.weight": embedding, **head}, tmp_path / "model.safetensors"
    )
    reads = _record_reads(monkeypatch)
    mapping = CheckpointMapping(optional_names=("lm_head.weight",))
    tied_params = _declare({"model.embed_tokens.weight": embedding}, torch.float32)

    tied = _load(tmp_path, mapping=mapping, params=tied_params)
    untied = _load(tmp_path, mapping=mapping, params=tied_params + _declare(head, torch.float32))

    assert tied.keys() == {"model.embed_tokens.weight"}
    assert all(reads.count(name) == 1 for name in head)  # by the untied load only
    for name, tensor in head.items():
        assert torch.equal(untied[name], tensor), name
    # The scale of a declared head is never optional.
    with pytest.raises(ValueError, match=r"'lm_head\.weight_scale_inv', which the model does not"):
        _load(
            tmp_path,
            mapping=mapping,
            params=tied_params
            + _declare({"lm_head.weight": head["lm_head.weight"]}, torch.float32),
        )


@pytest.mark.parametrize(
    "checkpoint,message",
    [
        (
            {"model.rotary_emb.inv_freq": torch.ones(2)},
            r"checkpoint tensor 'model\.rotary_emb\.inv_freq' maps to "
            r"'model\.rotary_emb\.inv_freq', which the model does not declare",
        ),
        (
            {"model.mlp.gate_proj.weight": torch.ones(2, 2)},
            r"checkpoint tensor 'model\.mlp\.gate_proj\.weight' maps to "
            r"'model\.mlp\.gate_up_proj\.weight', which the model does not declare",
        ),
        (
            {"model.norm.weight": torch.ones(3)},
            r"model\.norm\.weight: checkpoint gives shape \(3,\), the model declares \(4,\)",
        ),
        (
            {"model.q_proj.weight": torch.ones(2, 2), "model.k_proj.weight": torch.ones(2, 2)},
            r"Incomplete merge or expert groups in checkpoint: \['model\.qkv_proj\.weight'\]",
        ),
        (
            {"model.experts.0.down_proj.weight": torch.ones(2, 2)},
            r"Incomplete merge or expert groups in checkpoint: \['model\.experts\.down_proj'\]",
        ),
        (
            {f"model.experts.{expert}.down_proj.weight": torch.ones(2, 2) for expert in (0, 2)},
            r"model\.experts\.down_proj: expert 2 is outside 2 experts",
        ),
        (
            {"model.norm.weight": torch.ones(4, dtype=torch.float8_e4m3fn)},
            r"model\.norm\.weight: checkpoint dtype torch\.float8_e4m3fn cannot load as "
            r"torch\.float32",
        ),
    ],
    ids=[
        "unknown",
        "unknown-merge",
        "shape",
        "incomplete-merge",
        "incomplete-experts",
        "expert-range",
        "quantized-dtype",
    ],
)
def test_weight_loading_rejects_a_checkpoint_that_does_not_fit_the_model(
    tmp_path: Path,
    checkpoint: dict[str, torch.Tensor],
    message: str,
) -> None:
    safetensors.torch.save_file(checkpoint, tmp_path / "model.safetensors")
    params = (
        Param("model.norm.weight", (4,), torch.float32),
        Param("model.qkv_proj.weight", (6, 2), torch.float32),
        Param("model.experts.down_proj", (2, 2, 2), torch.float32),
    )

    def no_cpu_allocation(*args, **kwargs):
        if kwargs.get("device") == torch.device("cpu"):
            raise AssertionError("a misfit checkpoint must fail before allocating its tensor")
        return torch.empty(*args, **kwargs)

    with _Intercept({torch.empty: no_cpu_allocation}), pytest.raises(ValueError, match=message):
        _load(
            tmp_path,
            mapping=CheckpointMapping(merges=_PROJECTION_MERGES, stack_experts=True),
            params=params,
        )


def test_checkpoint_rule_matches_only_whole_trailing_name_components() -> None:
    mapping = CheckpointMapping(rules=(CheckpointRule("q_proj.weight", ("q.weight",)),))
    declared = {"model.q.weight", "q.weight"}

    (nested,) = mapping.placements("model.q_proj.weight", declared)
    (bare,) = mapping.placements("q_proj.weight", declared)

    assert (nested.name, bare.name) == ("model.q.weight", "q.weight")
    with pytest.raises(
        ValueError,
        match=r"'model\.xq_proj\.weight' maps to 'model\.xq_proj\.weight', which the model",
    ):
        mapping.placements("model.xq_proj.weight", declared)


def test_checkpoint_mapping_applies_only_the_first_matching_prefix_rename() -> None:
    mapping = CheckpointMapping(prefix_renames=(("a.", "b."), ("b.", "c.")))
    declared = {"b.x", "c.x", "c.y"}

    (renamed_once,) = mapping.placements("a.x", declared)
    (second_rename,) = mapping.placements("b.y", declared)

    assert (renamed_once.name, second_rename.name) == ("b.x", "c.y")


@pytest.mark.parametrize(
    "rule,message",
    [
        (lambda: CheckpointRule("q_proj.weight", ()), r"'q_proj\.weight' has no targets"),
        (
            lambda: CheckpointRule("q_proj.weight", ("q.weight", "gate.weight")),
            r"'q_proj\.weight' needs a transform to produce 2 tensors",
        ),
    ],
)
def test_checkpoint_rule_rejects_targets_it_cannot_produce(rule, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rule()


def test_checkpoint_rule_rejects_a_transform_that_returns_the_wrong_count(tmp_path: Path) -> None:
    safetensors.torch.save_file(
        {"model.q_proj.weight": torch.ones(2, 2)}, tmp_path / "x.safetensors"
    )
    rule = CheckpointRule("q_proj.weight", ("q.weight", "gate.weight"), Reshape(lambda t: (t,)))

    with pytest.raises(ValueError, match=r"model\.q_proj\.weight: .* returned 1 tensors for 2"):
        _load(
            tmp_path,
            mapping=CheckpointMapping(rules=(rule,)),
            params=_declare(
                {"model.q.weight": torch.ones(2, 2), "model.gate.weight": torch.ones(2, 2)},
                torch.float32,
            ),
        )


_LISTING = (
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "pytorch_model.bin",
    "README.md",
)


class _Hub:
    """A hub serving ``_LISTING`` at revision ``abc``."""

    def __init__(self, monkeypatch) -> None:
        self.fetched: list[tuple[str, str | None]] = []
        hub = self

        class Api:
            def model_info(self, repo_id, *, revision):
                siblings = [SimpleNamespace(rfilename=name) for name in _LISTING]
                return SimpleNamespace(sha=revision, siblings=siblings)

        def snapshot_download(repo_id, *, allow_patterns, tqdm_class, revision=None):
            hub.fetched.extend(
                (f, revision) for f in _LISTING if any(fnmatch(f, p) for p in allow_patterns)
            )
            return f"/cache/{repo_id}/snapshots/abc"

        def hf_hub_download(repo_id, filename, *, revision, tqdm_class):
            hub.fetched.append((filename, revision))

        monkeypatch.setattr(download_module, "HfApi", Api)
        monkeypatch.setattr(download_module, "snapshot_download", snapshot_download)
        monkeypatch.setattr(download_module, "hf_hub_download", hf_hub_download)


def test_a_hub_checkpoint_reads_its_files_first_and_its_weights_at_the_same_revision(
    monkeypatch,
) -> None:
    hub, reports = _Hub(monkeypatch), []

    source = download_module.fetch_metadata("org/model", "huggingface")
    described = {name for name, _ in hub.fetched}
    weights = open_weights(source, CheckpointOption(), reports.append)

    assert source.identity() == {"hub": "huggingface", "repo_id": "org/model", "revision": "abc"}
    assert weights.folder == "/cache/org/model/snapshots/abc"
    assert described == set(_LISTING) - {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "pytorch_model.bin",
        "README.md",
    }
    shards = {(name, "abc") for name in _LISTING if name.endswith(".safetensors")}
    assert set(hub.fetched[len(described) :]) == shards
    parsed = [re.fullmatch(r"downloading weights: file (\d)/2 \((.+)\)", r) for r in reports]
    assert [int(match[1]) for match in parsed] == [1, 2]


def test_a_local_checkpoint_downloads_nothing(tmp_path: Path) -> None:
    source = download_module.fetch_metadata(str(tmp_path), "modelscope")

    assert source.identity() == {"folder": str(tmp_path)}
    assert open_weights(source, CheckpointOption(), pytest.fail).folder == str(tmp_path)


def test_a_modelscope_download_renews_progress_while_it_runs(monkeypatch) -> None:
    heard = threading.Event()

    def snapshot_download(repo_id, *, allow_patterns):
        assert allow_patterns == ["*.safetensors"] and heard.wait(5)
        return f"/ms/{repo_id}"

    monkeypatch.setitem(
        sys.modules, "modelscope", SimpleNamespace(snapshot_download=snapshot_download)
    )
    monkeypatch.setattr(download_module, "BYTE_PROGRESS_SECONDS", 0.01)
    source = download_module.CheckpointSource("/ms/org/model", "modelscope", "org/model")
    reports: list[str] = []

    open_weights(source, CheckpointOption(), lambda report: (reports.append(report), heard.set()))

    assert reports and reports[0].startswith("downloading weights: ")


def _one_shard_hub(monkeypatch, hf_hub_download) -> None:
    class Api:
        def model_info(self, repo_id, *, revision):
            return SimpleNamespace(
                sha=revision, siblings=[SimpleNamespace(rfilename="big.safetensors")]
            )

    monkeypatch.setattr(download_module, "HfApi", Api)
    monkeypatch.setattr(download_module, "hf_hub_download", hf_hub_download)


def test_a_slow_file_reports_its_bytes_every_30_seconds_while_it_downloads(monkeypatch) -> None:
    now = [0.0]

    def hf_hub_download(repo_id, filename, *, revision, tqdm_class):
        bar = tqdm_class(total=4_000_000_000, initial=0, desc=filename, unit="B")
        for _ in range(100):  # 4 GB over 1000 s of fake time
            now[0] += 10.0
            bar.update(40_000_000)
        bar.close()

    _one_shard_hub(monkeypatch, hf_hub_download)
    reports: list[str] = []

    download_module._download_each_file("org/model", "abc", reports.append, clock=lambda: now[0])

    *bytes_reports, done = reports
    assert len(bytes_reports) == 1000 // download_module.BYTE_PROGRESS_SECONDS
    assert bytes_reports[0] == "downloading weights: big.safetensors 0.1/4.0 GB"
    assert done == "downloading weights: file 1/1 (big.safetensors)"


def test_a_hub_without_tqdm_class_still_reports_each_file(monkeypatch) -> None:
    fetched = []
    _one_shard_hub(monkeypatch, lambda repo_id, filename, *, revision: fetched.append(filename))
    reports: list[str] = []

    download_module._download_each_file("org/model", "abc", reports.append)

    assert fetched == ["big.safetensors"]
    assert reports == ["downloading weights: file 1/1 (big.safetensors)"]


def test_an_unreachable_hub_serves_the_cached_revision(monkeypatch) -> None:
    hub, reports = _Hub(monkeypatch), []

    class Offline:
        def model_info(self, repo_id, *, revision):
            raise download_module.httpx.ConnectError("offline")

    monkeypatch.setattr(download_module, "HfApi", Offline)
    source = download_module.CheckpointSource("/cache/snap", "huggingface", "org/model", "abc")

    assert open_weights(source, CheckpointOption(), reports.append).folder == "/cache/snap"
    assert reports == []
    assert {revision for _, revision in hub.fetched} == {"abc"}
