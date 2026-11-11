"""Vision graphs of the Qwen3.5 tower through ``runner.graphs.images``: a pack that fills its
budget is bitwise the eager tower's encoding of its images; a padded pack is bitwise the
eager encoding of the same padded pack (its padding one sequence of its own); an image past
the largest budget is bitwise its eager encoding alone. The tower runs no video. A second
child records capture seconds and pool bytes per budget and the encode latency of eager vs
graphs (measured, not gated)."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.support.checkpoints import QWEN3_5_9B
from tests.support.gpu import (
    assert_cuda_device_metadata,
    fixed_kv_page_capacity,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.paths import PYTHON_ROOT, REPO_ROOT, source_pythonpath
from tests.support.qwen3_5_gpu import selected_checkpoint, write_receipt

pytestmark = pytest.mark.checkpoints(QWEN3_5_9B)

if TYPE_CHECKING:
    import torch

    from cti.forward import ImageBatch, VisionTower
    from cti.inference._execution._vision_graph import VisionGraph

RECEIPT_PREFIX = "CTI_VISION_GRAPH="
NUM_KV_PAGES = 4096
# Qwen3.5's smallest image is 64 tokens: budgets 64, 128 and 256.
PARITY_BATCH_TOKENS = 256
BUDGETS = (64, 128, 256)
# Patch grids (one frame, rows, columns) of 4 patches a token, per budget, filling it.
EXACT_FITS = {
    64: ((1, 16, 16),),
    128: ((1, 16, 16), (1, 16, 16)),
    256: ((1, 16, 32), (1, 8, 32), (1, 8, 32)),
}
# 16, 60 and 64 tokens: one pack of 140 tokens padded to 256.
PADDED = ((1, 8, 8), (1, 12, 20), (1, 16, 16))
# 300 tokens past the largest budget, then one of 64.
OVERFLOW = ((1, 20, 60), (1, 16, 16))
# Measured: images of 448, 896 and 1344 pixels a side, 1, 2 and 4 at a time.
MEASURED_SIDES = (448, 896, 1344)
MEASURED_COUNTS = (1, 2, 4)
MEASURE_BATCH_TOKENS = 8192
MEASURE_REPEATS = 20


def _images(
    grids: tuple[tuple[int, int, int], ...], tower: VisionTower, *, seed: int
) -> ImageBatch:
    import torch

    from cti.forward import ImageBatch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    patches = [frames * height * width for frames, height, width in grids]
    pixels = torch.randint(
        0, 256, (sum(patches), tower.patch_embed.frame_dim), dtype=torch.uint8, generator=generator
    )
    bounds = [0]
    for count in patches:
        bounds.append(bounds[-1] + count)
    empty = torch.empty(0, dtype=torch.int64)
    batch = ImageBatch(
        pixels, grids, torch.tensor(bounds, dtype=torch.int32), max(patches), empty, empty, empty, 0
    )
    return batch.to(torch.device("cuda"))


def _token_slices(grids: tuple[tuple[int, int, int], ...], merge: int) -> list[slice]:
    slices, start = [], 0
    for frames, height, width in grids:
        tokens = frames * height * width // merge**2
        slices.append(slice(start, start + tokens))
        start += tokens
    return slices


def _padded_reference(
    tower: VisionTower, vision: VisionGraph, images: ImageBatch, budget: int
) -> torch.Tensor:
    """The eager encoding of ``images`` packed as ``budget``'s graph packs them, zero padding
    included, back in image order."""

    import torch

    patches = [frames * height * width for frames, height, width in images.grids]
    per_token = tower.merge_size**2
    (pack,) = vision.budgets.pack([count // per_token for count in patches])
    assert pack.budget == budget
    inputs = tower.embed(images)
    starts = [sum(patches[:image]) for image in range(len(patches))]
    padded = budget * per_token
    packed = tuple(
        torch.cat(
            [
                *(tensor[starts[image] : starts[image] + patches[image]] for image in pack.images),
                tensor.new_zeros((padded - sum(patches), *tensor.shape[1:])),
            ]
        )
        for tensor in inputs
    )
    bounds = vision.budgets.boundaries(budget, [patches[image] for image in pack.images], per_token)
    cu_patches = torch.tensor(bounds, dtype=torch.int32, device=inputs[0].device)
    encoded = tower.encode(packed, cu_patches, padded)
    slices = _token_slices(images.grids, tower.merge_size)
    output = torch.empty_like(encoded[: slices[-1].stop])
    offset = 0
    for image in pack.images:
        count = slices[image].stop - slices[image].start
        output[slices[image]] = encoded[offset : offset + count]
        offset += count
    return output


def _exercise_parity(
    tower: VisionTower, vision: VisionGraph, _stream: torch.cuda.Stream
) -> dict[str, object]:
    import torch

    receipt: dict[str, object] = {}
    with torch.inference_mode():
        for budget, grids in EXACT_FITS.items():
            images = _images(grids, tower, seed=budget)
            before = vision.replayed
            graphed = vision.encode(tower, images)
            assert vision.replayed == before + len(grids)
            assert torch.equal(graphed, tower.forward(images)), budget
        receipt["exact_fits"] = list(EXACT_FITS)

        images = _images(PADDED, tower, seed=7)
        graphed = vision.encode(tower, images)
        assert torch.equal(graphed, _padded_reference(tower, vision, images, BUDGETS[-1]))
        diff = (graphed.float() - tower.forward(images).float()).abs().max()
        # cuBLAS picks algorithms by row count: unpadded eager is a diagnostic only.
        assert bool(torch.isfinite(graphed).all())
        receipt["padded_unpadded_eager_max_abs"] = float(diff)

        images = _images(OVERFLOW, tower, seed=11)
        replayed, eager = vision.replayed, vision.eager
        graphed = vision.encode(tower, images)
        assert (vision.replayed - replayed, vision.eager - eager) == (1, 1)
        # Each image alone, from the batch's own embedding: the overflow eagerly, the other
        # filling the smallest budget.
        inputs = tower.embed(images)
        start = 0
        for rows_slice, (frames, height, width) in zip(
            _token_slices(OVERFLOW, tower.merge_size), OVERFLOW, strict=True
        ):
            count = frames * height * width
            rows = tuple(tensor[start : start + count] for tensor in inputs)
            cu_patches = torch.tensor([0, count], dtype=torch.int32, device=inputs[0].device)
            assert torch.equal(graphed[rows_slice], tower.encode(rows, cu_patches, count))
            start += count
        receipt["overflow"] = {"replayed": 1, "eager": 1}
    return receipt


def _encode_ms(encode, images: ImageBatch) -> float:
    import torch

    with torch.inference_mode():
        for _ in range(3):
            encode(images)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(MEASURE_REPEATS):
            encode(images)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / MEASURE_REPEATS


def _exercise_measure(
    tower: VisionTower, vision: VisionGraph, stream: torch.cuda.Stream
) -> dict[str, object]:
    import torch

    from cti.forward import EagerImages
    from cti.inference._execution._vision_graph import VisionBudgets, VisionGraph
    from cti.memory import GraphPool

    per_budget = {}
    for budget in vision.budgets.sizes:
        pool = GraphPool()
        single = VisionGraph(
            tower=tower,
            budgets=VisionBudgets((budget,)),
            pool=pool,
            stream=stream,
            device=torch.device("cuda"),
        )
        per_budget[budget] = {"seconds": single.captured.seconds, "bytes": single.captured.nbytes}
        single.close()
        pool.close()
        del single
        gc.collect()
        torch.cuda.empty_cache()
    eager = EagerImages()
    latency = {}
    for side in MEASURED_SIDES:
        grid = (1, side // 16, side // 16)
        for count in MEASURED_COUNTS:
            images = _images((grid,) * count, tower, seed=side + count)
            latency[f"{side}x{count}"] = {
                "eager_ms": _encode_ms(lambda batch: eager.encode(tower, batch), images),
                "graph_ms": _encode_ms(lambda batch: vision.encode(tower, batch), images),
            }
    return {
        "budgets": list(vision.budgets.sizes),
        "all_budgets": {"seconds": vision.captured.seconds, "bytes": vision.captured.nbytes},
        "per_budget": per_budget,
        "latency": latency,
    }


def _run_child(mode: str, snapshot: str) -> None:
    import torch

    import cti
    from cti.config import (
        CacheOptions,
        EngineOptions,
        EngineTuning,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
    )
    from tests.support.gpu import start_rank

    max_batch_tokens = PARITY_BATCH_TOKENS if mode == "parity" else MEASURE_BATCH_TOKENS
    with fixed_kv_page_capacity(NUM_KV_PAGES):
        worker = start_rank(
            EngineOptions(
                model=ModelOptions(path=snapshot, max_seq_len=max_batch_tokens),
                memory=MemoryOptions(inference_ratio=0.85),
                scheduling=SchedulingOptions(
                    max_running_requests=4, max_batch_tokens=max_batch_tokens
                ),
                cache=CacheOptions(prefix="naive"),
            ),
            EngineTuning(cuda_graph_bs=(1,), graph_row_buckets=(), vision_graphs=True),
            checks=True,
        )
    try:
        runner = worker.executor
        vision = runner.graphs.images
        tower = runner.model.vision_tower()
        captures = {capture.kind: capture for capture in runner.graph_stats().captures}
        assert captures["vision"] is vision.captured
        assert captures["vision"].nbytes > 0, captures
        exercise = _exercise_parity if mode == "parity" else _exercise_measure
        receipt = {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "source_root": str(Path(cti.__file__).resolve().parent),
            "sizes": list(captures["vision"].sizes),
            **exercise(tower, vision, runner.execution_stream),
        }
    finally:
        worker.close()
    gc.collect()
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))


def _child_receipt(tmp_path: Path, mode: str, *, timeout: float) -> dict:
    # A missing pinned checkpoint raises QualificationBlocked: it fails the run, never skips.
    snapshot = str(selected_checkpoint().snapshot())
    env, log_root = isolated_runtime_env(tmp_path, "vision-graph")
    env.update(
        PYTHONPATH=source_pythonpath(REPO_ROOT),
        PYTHONNOUSERSITE="1",
        PYTHONHASHSEED="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.inference.test_vision_graph", mode, snapshot),
        log_root / f"vision-graph-{mode}.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=timeout,
    )
    (receipt,) = (
        json.loads(line.removeprefix(RECEIPT_PREFIX))
        for line in output.splitlines()
        if line.startswith(RECEIPT_PREFIX)
    )
    assert_cuda_device_metadata(receipt)
    assert Path(receipt["source_root"]).is_relative_to(PYTHON_ROOT)
    return receipt


def test_vision_graph_replays_match_the_eager_tower(tmp_path: Path) -> None:
    receipt = _child_receipt(tmp_path, "parity", timeout=900)
    assert receipt["sizes"] == list(BUDGETS)
    assert receipt["exact_fits"] == list(BUDGETS)
    assert receipt["overflow"] == {"replayed": 1, "eager": 1}
    assert receipt["padded_unpadded_eager_max_abs"] >= 0
    write_receipt("vision-graph-parity", receipt)


def test_vision_graph_capture_and_latency_are_recorded(tmp_path: Path) -> None:
    receipt = _child_receipt(tmp_path, "measure", timeout=1500)
    assert receipt["budgets"][-1] == MEASURE_BATCH_TOKENS
    assert set(map(int, receipt["per_budget"])) == set(receipt["budgets"])
    write_receipt("vision-graph-measure", receipt)


if __name__ == "__main__":
    _run_child(*sys.argv[1:])
