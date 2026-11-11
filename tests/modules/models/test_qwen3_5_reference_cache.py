import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tests.support.qwen3_5_hf_reference import reference_placement
from tests.support.qwen3_5_reference_cache import ReferenceCache


def test_cache_identity_validation_and_eviction(tmp_path, monkeypatch):
    monkeypatch.setenv("CTI_REFERENCE_CACHE_DIR", str(tmp_path))
    recipe = {**reference_placement(torch.nn.Linear(1, 1)), "evidence_sha256": "a" * 64}
    cache = ReferenceCache(recipe)
    names = "model_revision config_hash weights_hash dtype body head attention ieee torch_version transformers_version implementation_hash tokenizer_hash template_hash processor_hash images adapter"
    identity = dict.fromkeys(names.split(), "pinned")
    identity["images"] = [b"image", torch.tensor([0.25])]
    key = cache.key([1, 2], [3, 4], **identity)
    for name in identity:
        assert cache.key([1, 2], [3, 4], **{**identity, name: "changed"}) != key
    assert cache.key([1, 2], [3, 5], **identity) != key
    identity["images"][1].add_(1)
    assert cache.key([1, 2], [3, 4], **identity) != key
    rows = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    proof = dict.fromkeys(["ieee", "torch_paths", "indexing"], True)
    proof.update(dtype="torch.float32", max_nats=0)
    cache.store(key, {"logits": rows}, proof, recipe)
    assert torch.equal(cache.load(key)["logits"], rows)
    with pytest.raises(ValueError):
        cache.store(key, {"logits": rows}, proof, {**recipe, "execution": {"w": "cuda:0"}})
    path = tmp_path / f"{key}.safetensors"
    with safe_open(path, framework="pt") as stored:
        metadata = stored.metadata()
    for field, value in [(0, 99), (1, "wrong"), (2, "wrong"), (3, "wrong"), (4, {})]:
        entry = json.loads(metadata["entry"])
        entry[field] = value
        save_file({"logits": rows}, path, metadata={"entry": json.dumps(entry)})
        assert cache.load(key) is None
    path.write_bytes(b"corrupt")
    assert cache.load(key) is None
    cache.store(key, {"logits": rows}, proof, recipe)
    cache.max_bytes = 1
    cache.evict()
    assert cache.bytes == 0 and cache.load(key) is None
    cache.store(key, {"logits": rows}, proof, recipe)
    assert cache.bytes == 0 and cache.load(key) is None


def test_same_version_source_patch_invalidates_identity(tmp_path, monkeypatch):
    import transformers

    from tests.support import hf_truth, qwen3_5_hf_reference
    from tests.support import qwen3_5_reference_cache as cache

    monkeypatch.setenv("CTI_REFERENCE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(torch.version, "cuda", "test")
    names = "libtorch_cpu.so libtorch_cuda.so libcublas.so.13 libcublasLt.so.13"
    libs = [tmp_path / name for name in names.split()]
    read = Path.read_text
    maps = "\n".join(f"0-1 r-xp 0 0:0 0 {lib}" for lib in libs)
    proc = Path("/proc/self/maps")
    monkeypatch.setattr(Path, "read_text", lambda p: maps if p == proc else read(p))
    files = [
        tmp_path / name for name in ("modeling.py", "processing.py", "truth.py", "reference.py")
    ]
    for p in [*files, *libs]:
        p.write_text("original")
    monkeypatch.setattr(transformers, "__file__", str(tmp_path / "__init__.py"))
    monkeypatch.setattr(hf_truth, "__file__", str(files[-2]))
    monkeypatch.setattr(qwen3_5_hf_reference, "__file__", str(files[-1]))
    before = cache.implementation_identity()
    for p in [*files, *libs]:
        p.write_text("patched!")
        assert cache.implementation_identity() != before
        p.write_text("original")
    monkeypatch.setattr(torch.version, "git_version", "different-build")
    assert cache.implementation_identity() != before
    hashes = cache.library_identity()
    assert {name for name, _ in hashes} == set(names.split())
    monkeypatch.setattr(
        hf_truth, "sha256_file", lambda _: pytest.fail("memoised library was rehashed")
    )
    assert cache.library_identity() == hashes
    maps = maps.splitlines()[0]
    with pytest.raises(ValueError, match="not all mapped"):
        cache.library_identity()


IDENTITY = dict.fromkeys(
    "model_revision config_hash weights_hash dtype body head attention ieee torch_version "
    "transformers_version implementation_hash tokenizer_hash template_hash processor_hash "
    "images adapter".split(),
    "pinned",
)


def _proof():
    proof = dict.fromkeys(["ieee", "torch_paths", "indexing"], True)
    proof.update(dtype="torch.float32", max_nats=0)
    return proof


def test_top4_rows_take_no_recipe_and_key_apart_from_logits(tmp_path, monkeypatch):
    from tests.support.qwen3_5_reference_cache import top4_rows

    monkeypatch.setenv("CTI_REFERENCE_CACHE_DIR", str(tmp_path))
    recipe = {**reference_placement(torch.nn.Linear(1, 1)), "evidence_sha256": "a" * 64}
    with pytest.raises(ValueError, match="take no recipe"):
        ReferenceCache(recipe, rows="top4")
    with pytest.raises(ValueError, match="recipe is required"):
        ReferenceCache(rows="logits")
    truth = torch.randn(3, 10, generator=torch.Generator().manual_seed(0))
    entry = top4_rows(truth)
    assert torch.equal(entry["topk_ids"], truth.topk(4).indices)
    assert torch.equal(entry["logsumexp"], truth.logsumexp(-1))
    cache, logits = ReferenceCache(rows="top4"), ReferenceCache(recipe)
    key = cache.key([1, 2], [3, 4, 5], **IDENTITY)
    assert logits.key([1, 2], [3, 4, 5], **IDENTITY) != key
    with pytest.raises(ValueError, match="placement mismatch"):
        cache.store(key, entry, _proof(), recipe)
    cache.store(key, entry, _proof())
    assert cache.load(key).keys() == entry.keys()
    assert logits.load(key) is None
    for broken in (
        {name: tensor for name, tensor in entry.items() if name != "logsumexp"},
        {**entry, "logsumexp": entry["logsumexp"][:2]},
        {**entry, "topk_logits": entry["topk_logits"].bfloat16()},
        {**entry, "topk_logits": torch.full_like(entry["topk_logits"], float("nan"))},
    ):
        with pytest.raises(ValueError, match="unverified"):
            cache.store(key, broken, _proof())


@pytest.mark.parametrize(
    ("weights", "free", "cpu", "streams", "placement"),
    [
        (20, 40, 0, False, "whole"),
        (35, 40, 0, True, "stream"),
        (35, 40, 100, False, "offload"),
        (35, 40, 2, False, None),
        (35, 8, 100, False, None),
    ],
    ids=["fits", "streams", "offloads", "too-little-ram", "no-gpu-room"],
)
def test_truth_placement_follows_the_hosts_memory(weights, free, cpu, streams, placement):
    from tests.support.qwen3_5_reference_cache import TruthUnavailable, truth_placement

    gib = 1 << 30
    if placement is None:
        with pytest.raises(TruthUnavailable):
            truth_placement(weights * gib, free * gib, cpu * gib, streams=streams)
        return
    chosen, max_memory = truth_placement(weights * gib, free * gib, cpu * gib, streams=streams)
    assert chosen == placement
    assert (max_memory is None) == (placement != "offload")
    if max_memory is not None:
        assert max_memory[0] == (free - 8) * gib and sum(max_memory.values()) >= weights * gib
