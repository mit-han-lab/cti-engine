# cti performance benchmarks

This directory contains cti-only performance entry points. It is intentionally
not a cross-engine comparison framework. Correctness belongs to `tests/`; these
drivers reject malformed measurements but do not duplicate numerical or lifecycle
proofs.

Operation-level profiling and tuning live in the independent
[`cti-kernel`](https://github.com/mit-han-lab/cti-kernel) repository. Engine
benchmarks measure the installed kernel artifact as part of the complete engine.

| Workload | Entry point | Primary measurements |
| --- | --- | --- |
| Offline inference | `python -m benchmark.inference.offline` | TTFT, TPOT, E2E, throughput |
| Online inference | `python -m benchmark.inference.online` | arrival-to-TTFT/TPOT/E2E, throughput |
| SFT | `python -m benchmark.training.sft` | step, forward/backward, optimizer, training peak memory |
| GRPO | `python -m benchmark.training.grpo` | rollout, switching, preparation, policy update, cycle, training peak memory |
| Residency switching | `python -m benchmark.switching` | both transition directions, round trip, first post-switch request |

Inference profiles start and stop an OpenAI-compatible server themselves and write
their artifacts to an empty directory outside the checkout. Training and switching
profiles run the typed Python API in one process and write one sealed JSON receipt.
Use `--help` on an entry point for its workload controls. A formal result must come
from a clean checkout; `--allow-dirty` exists only on inference profiles for local
tooling smoke tests.

Correctness coverage is deliberately separate:

- `tests/gpu/inference/` verifies generation, CUDA graphs, cache behavior, and
  numerical parity.
- `tests/gpu/training/` verifies padded and packed SFT/GRPO, optimizer updates,
  activation checkpointing, and reference/oracle parity.
- `tests/gpu/switching/` and `tests/gpu/integration/` verify KV invalidation,
  revision publication, online pause/resume, repeated switching, and clean shutdown.

Benchmark receipts are evidence about the measured checkout and hardware, not
correctness certificates and not portable performance baselines across GPUs.
SFT, GRPO and switching receipts record both installed package versions under
`runtime.packages`, plus available wheel hashes or resolved VCS commits. Index
installs do not require direct-URL metadata. Editable mode is recorded but does
not establish immutable kernel contents; use a built wheel for a reproducible
candidate comparison. Private installation URLs and local paths are omitted.
The V0 release reports RTX 5090 and DGX Spark independently; it never compares their
latencies or throughput. A DGX Spark receipt leaves physical peak memory unavailable
when GB10 unified-memory telemetry is not exposed by the platform.
