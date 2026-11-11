"""External, bounded HF row cache of teacher-forced sequences, in two row formats.

``logits`` rows, the FP32 logits the accuracy tier compares below placement-level rounding,
come only from a fixed, previously verified recipe, which precedes lookup. ``top4`` rows, the
FP32 truth's top-4 ids and logits and logsumexp that the default tier judges far above it, come
from any placement a host affords; each entry's proof names it.
``python -m tests.support.qwen3_5_reference_cache JOB`` computes a consumer's misses."""

import fcntl
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from tests.support import hf_layer_stream, hf_truth
from tests.support import qwen3_5_hf_reference as hf
from tests.support.checkpoints import pinned
from tests.support.hf_truth import content_digest as digest
from tests.support.paths import REPO_ROOT

# Marks the child's one stdout receipt line.
RECEIPT_PREFIX = "CTI_REFERENCE_TRUTH="


@dataclass(frozen=True)
class RowFormat:
    # Tensor names and dimensions; every tensor has one row per position.
    tensors: dict[str, int]
    # Computed only on the cache's verified recipe, which keys every entry.
    pinned: bool


ROW_FORMATS = {
    "logits": RowFormat({"logits": 2}, pinned=True),
    "top4": RowFormat({"topk_ids": 2, "topk_logits": 2, "logsumexp": 1}, pinned=False),
}


class TruthUnavailable(RuntimeError):
    """This host's GPU and CPU memory together cannot hold the FP32 model."""


class ReferenceCache:
    def __init__(self, recipe=None, *, rows="logits", max_bytes=8 << 30):
        self.folder = Path(os.environ["CTI_REFERENCE_CACHE_DIR"]).resolve()
        if self.folder.is_relative_to(REPO_ROOT) or max_bytes <= 0:
            raise ValueError("cache must be outside the repository, with a positive size cap")
        if rows not in ROW_FORMATS:
            raise ValueError(f"row format {rows!r} is not one of {sorted(ROW_FORMATS)}")
        if not ROW_FORMATS[rows].pinned:
            if recipe is not None:
                raise ValueError(f"{rows} rows come from any placement; they take no recipe")
        elif recipe is None or set(recipe) != {
            "parameters",
            "execution",
            "architecture",
            "evidence_sha256",
        }:
            raise ValueError("a verified parameter/execution/device recipe is required")
        elif not recipe["parameters"] or len(recipe["evidence_sha256"]) != 64:
            raise ValueError("recipe must carry a verified placement and evidence digest")
        self.recipe, self.rows = json.loads(json.dumps(recipe)), rows
        self.max_bytes = max_bytes
        self.proofs = {}
        self.folder.mkdir(parents=True, exist_ok=True)
        self.evict()

    def key(self, prompt_ids, continuation_ids, **identity):
        """images: original bytes, FP32 pixels, grid/M-RoPE; adapter: factors/config or merge."""
        required = "model_revision config_hash weights_hash dtype body head attention ieee torch_version transformers_version implementation_hash tokenizer_hash template_hash processor_hash images adapter"
        if set(identity) != set(required.split()):
            raise ValueError("incomplete reference identity")
        return digest([2, self.recipe, self.rows, identity, prompt_ids, continuation_ids])

    def _valid(self, tensors, proof):
        dims = ROW_FORMATS[self.rows].tensors
        floats = [tensor for tensor in tensors.values() if tensor.is_floating_point()]
        dtype = floats[0].dtype if floats else None
        bound = hf.FP32_FIRST_ROW_MAX_NATS if dtype == torch.float32 else 0.4
        return (
            set(tensors) == set(dims)
            and all(tensors[name].ndim == ndim for name, ndim in dims.items())
            and len({tensor.shape[0] for tensor in tensors.values()}) == 1
            and next(iter(tensors.values())).shape[0] > 0
            and dtype in (torch.float32, torch.bfloat16)
            and all(tensor.dtype == dtype for tensor in floats)
            and all(tensor.isfinite().all().item() for tensor in floats)
            and proof["dtype"] == str(dtype)
            and all(proof[k] is True for k in ("torch_paths", "ieee", "indexing"))
            and 0 <= proof["max_nats"] <= bound
        )

    def load(self, key):
        """The entry's tensors by name, or ``None`` if it is absent or fails its proof."""
        try:
            path = self.folder / f"{key}.safetensors"
            with safe_open(path, framework="pt") as stored:
                meta = stored.metadata()
                tensors = {name: stored.get_tensor(name) for name in stored.keys()}
            schema, saved_key, recipe, payload, proof = json.loads(meta["entry"])
            expected = [2, key, digest(self.recipe), digest(tensors)]
            if [schema, saved_key, recipe, payload] != expected or not self._valid(tensors, proof):
                raise ValueError("invalid reference proof")
            os.utime(path)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, SafetensorError):
            return None
        self.proofs[key] = proof
        return tensors

    def store(self, key, tensors, proof, actual_recipe=None):
        tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
        if actual_recipe != self.recipe or not self._valid(tensors, proof):
            raise ValueError("unverified reference or placement mismatch")
        metadata = {"entry": json.dumps([2, key, digest(self.recipe), digest(tensors), proof])}
        with tempfile.TemporaryDirectory(dir=self.folder) as temporary:
            partial = Path(temporary) / "rows"
            save_file(tensors, partial, metadata=metadata)
            os.replace(partial, self.folder / f"{key}.safetensors")
        self.evict()

    def evict(self):
        """Enforce the size cap by recency and refresh the on-disk byte count."""
        with (self.folder / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            entries = sorted(
                (p.stat().st_mtime_ns, p.stat().st_size, p)
                for p in self.folder.glob("*.safetensors")
            )
            self.bytes = sum(size for _, size, _ in entries)
            for _, size, path in entries:
                if self.bytes <= self.max_bytes:
                    break
                path.unlink(missing_ok=True)
                self.bytes -= size


def reference_identity(snapshot, checkpoint):
    files = {
        p.name: hf_truth.sha256_file(p)
        for p in sorted([*snapshot.glob("*.json"), *snapshot.glob("*.jinja")])
    }
    weights = sorted(snapshot.glob("*.safetensors"))
    if (
        snapshot.name != checkpoint.revision
        or not weights
        or any(
            not p.is_symlink() or p.resolve().parent.name != "blobs" or len(p.resolve().name) != 64
            for p in weights
        )
    ):
        raise ValueError("reference requires the pinned immutable HF snapshot/blob layout")
    return {
        **dict.fromkeys("dtype body head".split(), "float32"),
        **dict.fromkeys("tokenizer_hash template_hash processor_hash".split(), digest(files)),
        "model_revision": checkpoint.key,
        "config_hash": files["config.json"],
        "weights_hash": digest([(p.name, p.resolve().name) for p in weights]),
        "attention": "eager",
        "ieee": True,
        "torch_version": str(torch.__version__),
        "transformers_version": hf_truth.code_version()["transformers"],
        "implementation_hash": implementation_identity(),
        "adapter": None,
    }


def top4_rows(logits):
    """The ``top4`` rows of one sequence from its FP32 logits rows [N, V]."""
    logits = logits.float()
    values, ids = logits.topk(4, dim=-1)
    return {"topk_ids": ids, "topk_logits": values, "logsumexp": logits.logsumexp(-1)}


def truth_placement(weight_bytes, free_gpu_bytes, available_cpu_bytes, *, streams):
    """Where a host computes the FP32 truth: ``("whole", None)`` when the weights and the
    activations' headroom fit on the GPU, ``("stream", None)`` for an attention-only decoder
    that does not, else ``("offload", max_memory)`` with what does not fit in CPU RAM. Raises
    ``TruthUnavailable`` when not even that holds the weights."""
    if weight_bytes + (hf_truth.AUTO_GPU_HEADROOM_GIB << 30) <= free_gpu_bytes:
        return "whole", None
    if streams:
        return "stream", None
    try:
        max_memory = hf_truth.offload_max_memory(
            torch.device("cuda:0"),
            None,
            free_gpu_bytes=[free_gpu_bytes],
            available_cpu_bytes=available_cpu_bytes,
        )
    except ValueError as error:
        raise TruthUnavailable(str(error)) from error
    if weight_bytes > sum(max_memory.values()):
        raise TruthUnavailable(
            f"{weight_bytes / (1 << 30):.1f} GiB of FP32 weights exceed the "
            f"{sum(max_memory.values()) / (1 << 30):.1f} GiB this host's GPU and CPU RAM hold"
        )
    return "offload", max_memory


def fp32_truth(snapshot, requests, *, device, placement=None):
    """Each request's FP32 logits rows over its continuation [N, V] and their proof, from the
    checkpoint's own Transformers class on ``placement`` (``truth_placement``'s for this host
    unless given); the proof names the placement. Image requests never stream: the
    layer-streamed path is text-only."""
    architecture = json.loads((snapshot / "config.json").read_text())["architectures"][0]
    text = all(request.get("images") is None for request in requests)
    streams = text and hf_truth.reference_architecture(architecture).streams
    max_memory = None
    if placement is None and device.type == "cuda":
        import psutil

        weights = 4 * _parameter_count(snapshot)
        free = torch.cuda.mem_get_info(device)[0]
        available = psutil.virtual_memory().available
        placement, max_memory = truth_placement(weights, free, available, streams=streams)
    placement = placement or "whole"
    if placement == "stream":
        rows, proof, loaded = _streamed_truth(snapshot, requests, device)
    else:
        model = hf_truth.load_reference(snapshot, device, torch.float32, max_memory)
        if max_memory is not None:
            hf_truth.pin_offloaded_weights(model)
        rows, proof = hf.score_continuations(model, requests, device=device)
        loaded = hf_truth.parameter_bytes(model)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    where = {"mode": placement, "device": name, "parameter_bytes": loaded}
    return rows, {**proof, "placement": where}


def _parameter_count(snapshot):
    count = 0
    for file in snapshot.glob("*.safetensors"):
        with safe_open(file, framework="pt") as stored:
            count += sum(math.prod(stored.get_slice(name).get_shape()) for name in stored.keys())
    return count


def _streamed_truth(snapshot, requests, device):
    """``fp32_truth`` one decoder layer at a time, with ``score_continuations``' checks: the
    torch paths, IEEE FP32, and the shortest request's first row against its prompt alone."""
    from transformers.integrations import hub_kernels

    hf_truth.check_ieee_fp32()
    stream = hf_layer_stream.LayerStream(snapshot, device=device, dtype=torch.float32)
    shortest = min(range(len(requests)), key=lambda i: len(requests[i]["prompt_ids"]))
    sequences = [request["prompt_ids"] + request["continuation_ids"] for request in requests]
    with torch.inference_mode():
        *hidden, alone = stream.forward([*sequences, requests[shortest]["prompt_ids"]])
        rows = []
        for request, states in zip(requests, hidden, strict=True):
            first, count = len(request["prompt_ids"]) - 1, len(request["continuation_ids"])
            rows.append(_projected(stream, states[:, first : first + count]))
        prompt_row = _projected(stream, alone[:, -1:])[0]
    stream.drop_head()
    if (
        hub_kernels._kernels_enabled
        or hf_truth.loaded_blocked_modules()
        or any(sys.modules.get(p, ...) is not None for p in hf_truth.BLOCKED_PACKAGES)
    ):
        raise RuntimeError("reference must use isolated torch kernels")
    hf_truth.check_ieee_fp32()
    first_row = rows[shortest][0]
    ids = list({*first_row.topk(8).indices.tolist(), requests[shortest]["continuation_ids"][0]})
    error = float((first_row.log_softmax(-1)[ids] - prompt_row.log_softmax(-1)[ids]).abs().max())
    if not error <= hf.FP32_FIRST_ROW_MAX_NATS:
        raise RuntimeError(f"reference first-row self-check failed: {error} nats")
    proof = dict.fromkeys(("torch_paths", "ieee", "indexing"), True)
    proof.update(dtype=str(torch.float32), max_nats=error, shortest=shortest)
    return rows, proof, {"streamed": stream.loaded_bytes}


def _projected(stream, states):
    """The LM head's FP32 logits of ``[1, T, hidden]`` states, on the CPU."""
    return torch.cat(list(stream.logits(states))).float().cpu()


def _compute_misses(job, *, device):
    """Compute and store a consumer's job; its receipt: the rows computed and where, or why
    this host cannot compute them."""
    data = torch.load(job, weights_only=True)
    cache = ReferenceCache(data["manifest"]["recipe"], rows=data["rows"])
    checkpoint = pinned(data["manifest"]["model_key"]).snapshot()
    requests = list(data["requests"].values())
    started = time.monotonic()
    if ROW_FORMATS[cache.rows].pinned:
        placement = {
            name.removeprefix("device_map:"): int(device) if device.isdecimal() else device
            for name, device in cache.recipe["parameters"].items()
            if name.startswith("device_map:")
        }
        model = hf_truth.load_reference(checkpoint, device, torch.float32, device_map=placement)
        actual = {
            **hf.reference_placement(model),
            "evidence_sha256": cache.recipe["evidence_sha256"],
        }
        if actual != cache.recipe:
            raise ValueError("actual reference placement differs from verified recipe")
        rows, proof = hf.score_continuations(model, requests, device=device)
        entries = [{"logits": logits} for logits in rows]
    else:
        try:
            rows, proof = fp32_truth(checkpoint, requests, device=device)
        except TruthUnavailable as error:
            return {"unavailable": str(error)}
        actual = None
        entries = [top4_rows(logits) for logits in rows]
    if implementation_identity() != data["implementation"]:
        raise ValueError("reference child implementation differs from cache identity")
    for key, entry in zip(data["requests"], entries, strict=True):
        cache.store(key, entry, proof, actual)
    return {
        "computed": len(entries),
        "placement": proof.get("placement"),
        "seconds": round(time.monotonic() - started, 3),
    }


def configure_reference():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    hf_truth.use_ieee_fp32()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def library_identity():
    required = {"libtorch_cpu.so"}
    if torch.version.cuda is not None:
        required.update(("libtorch_cuda.so", "libcublas.so", "libcublasLt.so"))
    paths = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6:
            name = fields[5].rsplit("/", 1)[-1].split(".so")[0] + ".so"
            if name in required:
                paths[Path(fields[5]).resolve(strict=True)] = name
    if set(paths.values()) != required:
        raise ValueError("reference implementation libraries are not all mapped")
    folder = Path(os.environ["CTI_REFERENCE_CACHE_DIR"]) / ".library-digests"
    folder.mkdir(exist_ok=True)
    hashes = set()
    for path in sorted(paths):
        stat = path.stat()
        memo = folder / digest([str(path), stat.st_size, stat.st_mtime_ns])
        value = memo.read_text() if memo.is_file() else ""
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            value = hf_truth.sha256_file(path)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError("reference library changed while hashing")
            with tempfile.NamedTemporaryFile(mode="w", dir=folder, delete=False) as partial:
                partial.write(value)
            os.replace(partial.name, memo)
        hashes.add((path.name, value))
    return sorted(hashes)


def implementation_identity():
    import transformers

    build = [str(torch.__version__), torch.version.git_version, torch.version.cuda]
    package = Path(transformers.__file__).parent
    sources = {str(p.relative_to(package)): hf_truth.sha256_file(p) for p in package.rglob("*.py")}
    return digest(
        {
            "transformers": sources,
            "reference": [
                hf_truth.sha256_file(Path(source))
                for source in (hf.__file__, __file__, hf_truth.__file__, hf_layer_stream.__file__)
            ],
            "torch": [*build, torch.backends.cudnn.version(), library_identity()],
            "cublas": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        }
    )


if __name__ == "__main__":
    hf_truth.block_optional_kernels()
    configure_reference()
    report = _compute_misses(Path(sys.argv[1]), device=torch.device("cuda:0"))
    print(RECEIPT_PREFIX + json.dumps(report), flush=True)
