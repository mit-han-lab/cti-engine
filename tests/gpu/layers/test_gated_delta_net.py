"""Gated DeltaNet layers on the GPU: the layer over the recurrent-state pool, with the
recurrent backend's metadata, on the ``ctikernels.gdn`` operations its ``LayerServices``
resolve and checked against the torch reference operations of
``tests/support/gated_delta_net.py``; prompt rows beside blocks that K2 verifies, in one
layer call, against the two apart. The backend's decode-graph participation is checked in
``tests/gpu/inference/test_gated_delta_net_decode_graphs.py``.

Each test runs one child process that prints a receipt; the parent asserts on it.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from tests.support.gated_delta_net_gpu import (
    NUM_SLOTS,
    bind_layer,
    component_receipt,
    gdn_layer,
    pools_equal,
    projected_rows,
    relative_error,
    run_component,
    run_layer,
)
from tests.support.gpu import engine_checks_off

pytestmark = pytest.mark.checkpoints()

if TYPE_CHECKING:
    import torch

    from cti.kvcache import RecurrentStatePool
    from cti.layers.gated_delta_net import GatedDeltaNet
    from cti.recurrent import RecurrentBackend
    from tests.support.forward_rows import Entry

PREFILL_TOKENS = 70
CHUNK_SPLITS = (1, 3, 4, 63, 64, 65)
# Max |a - b| / max |b| against the FP32 reference. The kernels carry BF16 activations
# between their stages; their own GPU tests bound y's peak error at 3e-2. What this test
# checks, slot routing and splitting, fails by O(1) when wrong.
REFERENCE_TOLERANCE = 3e-2


# ============================== Child process ==============================


def _operations(device: torch.device) -> dict[str, Any]:
    """The kernels against the reference, chunking, mixed batches, padding and NaN rows."""

    import torch

    from tests.support.forward_rows import decode_entry, extend_entry

    receipt: dict[str, Any] = {}
    projected = projected_rows(PREFILL_TOKENS, 1, device)

    kernel_layer = gdn_layer(device)
    kernel_backend, kernel_pool = bind_layer(kernel_layer, device)
    reference_layer = gdn_layer(device, kernels=False)
    reference_backend, reference_pool = bind_layer(reference_layer, device)
    one_shot = run_layer(
        kernel_layer, kernel_backend, projected, extend_entry(1, 0, PREFILL_TOKENS)
    )
    reference = run_layer(
        reference_layer, reference_backend, projected, extend_entry(1, 0, PREFILL_TOKENS)
    )
    receipt["kernels_vs_reference"] = {
        "y": relative_error(one_shot, reference),
        "state": relative_error(
            kernel_pool.layer_views(0)[0][1], reference_pool.layer_views(0)[0][1]
        ),
        "conv_history_equal": torch.equal(
            kernel_pool.layer_views(0)[1][1], reference_pool.layer_views(0)[1][1]
        ),
    }

    # A chunked prefill (two chunks) and four decodes against the one-shot prefill.
    decode_start = PREFILL_TOKENS - 4
    chunked: dict[str, float] = {}
    for split in CHUNK_SPLITS:
        layer = gdn_layer(device)
        backend, _ = bind_layer(layer, device)
        parts = [
            run_layer(layer, backend, projected[:split], extend_entry(1, 0, split)),
            run_layer(
                layer, backend, projected[split:decode_start], extend_entry(1, split, decode_start)
            ),
        ]
        parts.extend(
            run_layer(layer, backend, projected[token : token + 1], decode_entry(1, token))
            for token in range(decode_start, PREFILL_TOKENS)
        )
        chunked[str(split)] = relative_error(torch.cat(parts), one_shot)
    receipt["chunked_vs_one_shot"] = chunked

    # One mixed batch (a fresh extend, a continued extend, two decodes) against each
    # request alone. The mixed forward must not synchronise with the host.
    first, second = projected_rows(9, 2, device), projected_rows(12, 3, device)
    decode_a, decode_b = projected_rows(5, 4, device), projected_rows(4, 5, device)

    def warmed() -> tuple[GatedDeltaNet, RecurrentBackend, RecurrentStatePool]:
        layer = gdn_layer(device)
        backend, pool = bind_layer(layer, device)
        run_layer(layer, backend, first[:4], extend_entry(1, 0, 4))
        run_layer(layer, backend, decode_a[:4], extend_entry(0, 0, 4))
        run_layer(layer, backend, decode_b[:3], extend_entry(2, 0, 3))
        return layer, backend, pool

    alone_layer, alone_backend, alone_pool = warmed()

    def alone_decode(row: torch.Tensor, entry: Entry) -> torch.Tensor:
        # Padded to the mixed batch's two decodes: the decode kernel may choose its launch
        # configuration, and so the FP32 state's rounding, by batch size.
        padded = torch.cat((row, torch.zeros_like(row)))
        return run_layer(alone_layer, alone_backend, padded, entry, padding=1)[:1]

    alone = torch.cat(
        (
            run_layer(alone_layer, alone_backend, second, extend_entry(3, 0, 12)),
            run_layer(alone_layer, alone_backend, first[4:], extend_entry(1, 4, 9)),
            alone_decode(decode_a[4:], decode_entry(0, 4)),
            alone_decode(decode_b[3:], decode_entry(2, 3)),
        )
    )
    mixed_layer, mixed_backend, mixed_pool = warmed()
    mixed_rows = torch.cat((second, first[4:], decode_a[4:], decode_b[3:]))
    entries = (
        extend_entry(3, 0, 12),
        extend_entry(1, 4, 9),
        decode_entry(0, 4),
        decode_entry(2, 3),
    )
    torch.cuda.synchronize(device)
    # A host synchronisation inside raises, so reaching the receipt proves there was none.
    torch.cuda.set_sync_debug_mode("error")
    try:
        with engine_checks_off():
            mixed = run_layer(mixed_layer, mixed_backend, mixed_rows, *entries)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    receipt["mixed_vs_alone"] = {
        "y": relative_error(mixed, alone),
        "bitwise": torch.equal(mixed, alone),
        "pools_equal": pools_equal(mixed_pool, alone_pool),
        "synchronization_free": True,
    }

    # Padding rows give zero and leave every pool row but the real slot's untouched.
    layer = gdn_layer(device)
    backend, pool = bind_layer(layer, device)
    run_layer(layer, backend, projected_rows(3, 6, device), extend_entry(2, 0, 3))
    before = [view.clone() for view in pool.layer_views(0)]
    y = run_layer(layer, backend, projected_rows(3, 7, device), decode_entry(2, 3), padding=2)
    untouched = [slot for slot in range(NUM_SLOTS) if slot != 2]
    after = pool.layer_views(0)
    receipt["padding"] = {
        "padding_rows_zero": bool(torch.count_nonzero(y[1:]) == 0),
        "real_row_nonzero": bool(torch.count_nonzero(y[0]) > 0),
        "other_slots_untouched": all(
            torch.equal(now[untouched], then[untouched])
            for now, then in zip(after, before, strict=True)
        ),
        "real_slot_advanced": not torch.equal(after[0][2], before[0][2]),
    }

    # A fresh request in a slot left full of NaN equals one in a clean slot.
    layer = gdn_layer(device)
    backend, pool = bind_layer(layer, device)
    rows = projected_rows(6, 8, device)
    clean = run_layer(layer, backend, rows, extend_entry(0, 0, 6))
    state, conv_state = pool.layer_views(0)
    state[3].fill_(float("nan"))
    conv_state[3].fill_(float("nan"))
    reused = run_layer(layer, backend, rows, extend_entry(3, 0, 6))
    receipt["nan_canary"] = {
        "bitwise": torch.equal(reused, clean),
        "state_finite": bool(torch.isfinite(state[3]).all()),
        "conv_history_finite": bool(torch.isfinite(conv_state[3].float()).all()),
    }
    return receipt


def _blocks(device: torch.device) -> dict[str, Any]:
    """Prompt extensions (fresh and continued) and verified blocks in one layer call against
    the prompt forward and the verify apart: outputs and pools must match bitwise."""

    import torch

    from cti.forward import ForwardInputs, InlineBreaks, RoundShape
    from cti.recurrent import RecurrentGeometry, RecurrentPhase, RecurrentRounds
    from tests.support.forward_rows import extend_entry, forward_rows

    width = 16
    frontiers = {0: 20, 1: 33}
    # (slot, start, end): a fresh prompt and a chunk continuing slot 5's first 12 tokens.
    prompts = ((4, 0, 9), (5, 12, 30))
    extensions = torch.cat(
        [projected_rows(end - start, 40 + slot, device) for slot, start, end in prompts]
    )
    blocks_rows = projected_rows(len(frontiers) * width, 50, device)
    slot_ids = torch.tensor(list(frontiers), dtype=torch.int32, device=device)
    positions = torch.stack(
        [torch.arange(f, f + width, device=device) for f in frontiers.values()]
    ).to(torch.int64)
    frontier_table = torch.zeros(NUM_SLOTS, dtype=torch.int64, device=device)
    for slot, frontier in frontiers.items():
        frontier_table[slot] = frontier

    def setup() -> tuple[GatedDeltaNet, RecurrentBackend, RecurrentStatePool, Any]:
        layer = gdn_layer(device)
        backend, pool = bind_layer(layer, device)
        for slot, frontier in frontiers.items():
            run_layer(
                layer,
                backend,
                projected_rows(frontier, 10 + slot, device),
                extend_entry(slot, 0, frontier),
            )
        run_layer(layer, backend, projected_rows(12, 20, device), extend_entry(5, 0, 12))
        shape = RoundShape(width, 2, len(frontiers))
        geometry = RecurrentGeometry(pool.num_slots, len(pool.snapshot_rows), pool.device)
        replay = layer.create_replay(max_batch=len(frontiers), rounds=shape, geometry=geometry)
        return layer, backend, pool, RecurrentRounds([replay], shape=shape)

    def verify(rounds, run) -> torch.Tensor:
        rounds.mark_stale(frontiers)
        status_slot = rounds.begin(list(frontiers), frontier_table)
        with torch.inference_mode():
            y = run(rounds.prepare_verify(status_slot, slot_ids, positions))
        status = rounds.end(status_slot, len(frontiers))
        for source, target in status.copies:
            target.copy_(source)
        torch.cuda.synchronize(device)
        status.settle()
        return y

    entries = tuple(extend_entry(slot, start, end) for slot, start, end in prompts)
    layer, backend, apart_pool, rounds = setup()
    prompt_y = run_layer(layer, backend, extensions, *entries)
    block_y = verify(
        rounds,
        lambda rows: layer.forward(
            blocks_rows, SimpleNamespace(recurrent_metadata=rows, graph_breaks=InlineBreaks())
        ),
    )
    apart = torch.cat((prompt_y, block_y))

    layer, backend, joint_pool, rounds = setup()
    prompt_rows = forward_rows(*entries)
    prompt_metadata = backend.prepare_metadata(prompt_rows)

    def joint(rows):
        batch = ForwardInputs(
            None,
            None,
            None,
            prompt_rows.size,
            ragged=True,
            recurrent_metadata=replace(
                prompt_metadata, phase=RecurrentPhase.VERIFY, verify=rows.verify
            ),
        )
        return layer.forward(torch.cat((extensions, blocks_rows)), batch)

    together = verify(rounds, joint)
    return {
        "bitwise": torch.equal(together, apart),
        "pools_equal": pools_equal(joint_pool, apart_pool),
        "finite": bool(torch.isfinite(together).all()),
    }


# ============================== Tests ==============================


def _handoff(device: torch.device) -> dict[str, Any]:
    """Verified blocks and a commit with the rows' handoff eager or captured (replaying a
    graph): outputs and pools must match bitwise."""

    import torch

    from cti.forward import InlineBreaks, RoundShape
    from cti.memory import GraphPool
    from cti.recurrent import RecurrentGeometry, RecurrentRounds
    from tests.support.forward_rows import extend_entry

    width = 16
    frontiers = {0: 20, 1: 33}
    blocks_rows = projected_rows(len(frontiers) * width, 50, device)
    slot_ids = torch.tensor(list(frontiers), dtype=torch.int32, device=device)
    positions = torch.stack(
        [torch.arange(f, f + width, device=device) for f in frontiers.values()]
    ).to(torch.int64)
    frontier_table = torch.zeros(NUM_SLOTS, dtype=torch.int64, device=device)
    for slot, frontier in frontiers.items():
        frontier_table[slot] = frontier

    def setup() -> tuple[GatedDeltaNet, RecurrentBackend, RecurrentStatePool, Any]:
        layer = gdn_layer(device)
        backend, pool = bind_layer(layer, device)
        for slot, frontier in frontiers.items():
            run_layer(
                layer,
                backend,
                projected_rows(frontier, 10 + slot, device),
                extend_entry(slot, 0, frontier),
            )
        shape = RoundShape(width, 2, len(frontiers))
        geometry = RecurrentGeometry(pool.num_slots, len(pool.snapshot_rows), pool.device)
        replay = layer.create_replay(max_batch=len(frontiers), rounds=shape, geometry=geometry)
        return layer, backend, pool, RecurrentRounds([replay], shape=shape)

    lengths = torch.tensor([5, width], dtype=torch.int32, device=device)

    def verify(rounds, run, *, commit=True) -> torch.Tensor:
        rounds.mark_stale(frontiers)
        status_slot = rounds.begin(list(frontiers), frontier_table)
        rows = rounds.prepare_verify(status_slot, slot_ids, positions)
        with torch.inference_mode():
            y = run(rows).clone()
            if commit:
                rounds.commit(status_slot, lengths)
        status = rounds.end(status_slot, len(frontiers))
        for source, target in status.copies:
            target.copy_(source)
        torch.cuda.synchronize(device)
        status.settle()
        return y

    from types import SimpleNamespace

    def alone(layer, rounds, commit=True) -> torch.Tensor:
        metadata = SimpleNamespace(recurrent_metadata=None, graph_breaks=InlineBreaks())

        def run(rows):
            metadata.recurrent_metadata = rows
            return layer.forward(blocks_rows, metadata)

        return verify(rounds, run, commit=commit)

    layer, _, eager_pool, rounds = setup()
    eager = alone(layer, rounds)
    layer, _, captured_pool, rounds = setup()
    # Eager verifies without a commit warm every call shape the captures take, in each
    # status slot.
    for _ in range(rounds.depth):
        alone(layer, rounds, commit=False)
    rounds.capture_handoffs(GraphPool(), torch.cuda.Stream(device))
    captured = alone(layer, rounds)
    return {
        "finite": bool(torch.isfinite(eager).all()),
        "captured_bitwise": torch.equal(captured, eager),
        "captured_pools_equal": pools_equal(captured_pool, eager_pool),
    }


# Qwen3.5 9B's and 27B's Gated DeltaNet key and value heads.
_PREFIX_GEOMETRIES = {"9B": (16, 32), "27B": (16, 48)}


def _prefix_states(device: torch.device) -> dict[str, Any]:
    """Resolution's answer whether the prefill exports states against the built layer's, at
    each geometry's heads for TP1 and for a TP2 rank, and the cache it plans then."""

    import torch
    from transformers import Qwen3_5Config

    from cti.config import CacheOptions, EngineOptions, EngineTuning, ModelOptions
    from cti.distributed import Topology
    from cti.layers import LayerServices
    from cti.layers.gated_delta_net import GatedDeltaNet
    from cti.models import get_model_class
    from cti.resolution import ResolvedModel
    from cti.utils import get_device_capability
    from tests.support.checkpoint_facts import checkpoint_facts
    from tests.support.tp_shards import default_dtype, site

    model_class = get_model_class("Qwen3_5ForConditionalGeneration")
    answers: dict[str, dict[str, Any]] = {}
    for name, (keys, values) in _PREFIX_GEOMETRIES.items():
        hf_config = Qwen3_5Config(
            text_config={"linear_num_key_heads": keys, "linear_num_value_heads": values}
        )
        for num_ranks in (1, 2):
            resolved = model_class.exports_prefill_states(hf_config, device, num_ranks=num_ranks)
            with default_dtype(torch.bfloat16):
                layer = GatedDeltaNet(
                    num_key_heads=keys,
                    num_value_heads=values,
                    key_head_dim=GatedDeltaNet.HEAD_DIM,
                    value_head_dim=GatedDeltaNet.HEAD_DIM,
                    conv_kernel_size=GatedDeltaNet.CONV_KERNEL_SIZE,
                    site=site(Topology(0, num_ranks), device=device, services=LayerServices()),
                )
            answers[f"{name}-tp{num_ranks}"] = {
                "resolved": resolved,
                "built": layer.exports_prefill_states,
            }
    checkpoint = ResolvedModel(
        facts=checkpoint_facts(hf_config),
        model_config=SimpleNamespace(
            architectures=["Qwen3_5ForConditionalGeneration"],
            model_type="qwen3_5",
            head_dim=256,
            max_position=4096,
            rotary_config=SimpleNamespace(max_position=4096),
        ),
        model_class=model_class,
    )
    plans: dict[str, bool | str] = {}
    for prefix in ("auto", "radix"):
        options = EngineOptions(
            model=ModelOptions(path="unused"), cache=CacheOptions(prefix=prefix)
        )
        try:
            plan = checkpoint.plan(
                checkpoint.check(options, EngineTuning(), speculative=None),
                device=device,
                capability=get_device_capability(),
                device_memory=torch.cuda.get_device_properties(device).total_memory,
                online=False,
            )
        except ValueError as error:
            plans[prefix] = str(error)
        else:
            plans[prefix] = plan.prefix_cache == "radix"
    return {"answers": answers, "plans": plans}


def _prefix_states_without_cute(device: torch.device) -> dict[str, Any]:
    """``_prefix_states`` with the CuTe prefill switched off: TP1 falls back to Triton, which
    exports no states; a TP2 rank to fla-core, which exports them where cti-kernels' CuTe
    capture kernel runs."""

    import os

    os.environ["CTI_GDN_PREFILL_CUTE"] = "0"
    return _prefix_states(device)


def _receipt(tmp_path: Path, component: str) -> dict[str, Any]:
    return component_receipt(tmp_path, "tests.gpu.layers.test_gated_delta_net", component)


def test_kernel_operations_serve_the_backend_pool_like_the_reference(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path, "operations")

    reference = receipt["kernels_vs_reference"]
    assert reference["y"] < REFERENCE_TOLERANCE
    assert reference["state"] < REFERENCE_TOLERANCE
    assert reference["conv_history_equal"] is True
    assert set(receipt["chunked_vs_one_shot"]) == {str(split) for split in CHUNK_SPLITS}
    assert max(receipt["chunked_vs_one_shot"].values()) < REFERENCE_TOLERANCE
    mixed = receipt["mixed_vs_alone"]
    assert mixed["bitwise"] is True, mixed
    assert mixed["pools_equal"] is True
    assert mixed["synchronization_free"] is True
    assert receipt["padding"] == {
        "padding_rows_zero": True,
        "real_row_nonzero": True,
        "other_slots_untouched": True,
        "real_slot_advanced": True,
    }
    assert receipt["nan_canary"] == {
        "bitwise": True,
        "state_finite": True,
        "conv_history_finite": True,
    }


def test_prompt_rows_and_verified_blocks_share_one_layer_call(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path, "blocks")

    assert {key: receipt[key] for key in ("bitwise", "pools_equal", "finite")} == {
        "bitwise": True,
        "pools_equal": True,
        "finite": True,
    }


def test_a_captured_handoff_verifies_and_commits_like_the_eager_one(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path, "handoff")

    keys = ("finite", "captured_bitwise", "captured_pools_equal")
    assert {key: receipt[key] for key in keys} == dict.fromkeys(keys, True)


@pytest.mark.parametrize("component", ["prefix_states", "prefix_states_without_cute"])
def test_resolution_answers_whether_the_built_prefill_exports_states(
    tmp_path: Path, component: str
) -> None:
    receipt = _receipt(tmp_path, component)

    answers = receipt["answers"]
    assert set(answers) == {f"{name}-tp{size}" for name in _PREFIX_GEOMETRIES for size in (1, 2)}
    assert all(answer["resolved"] is answer["built"] for answer in answers.values()), answers
    # The plan follows the answer: auto drops to the naive cache, radix is refused.
    exports = answers["27B-tp1"]["built"]
    plans = receipt["plans"]
    assert plans["auto"] is exports
    if exports:
        assert plans["radix"] is True
    else:
        assert plans["radix"].endswith('; use "naive"'), plans


def main() -> None:
    components = {
        "operations": _operations,
        "blocks": _blocks,
        "handoff": _handoff,
        "prefix_states": _prefix_states,
        "prefix_states_without_cute": _prefix_states_without_cute,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", required=True, choices=tuple(components))
    run_component(parser.parse_args().component, components)


if __name__ == "__main__":
    main()
