# Engine and kernel packages

The engine (this repository) owns the APIs, scheduling, model lifecycle, training and
integration tests. The kernel package (`mit-han-lab/cti-kernel`) owns operations, providers,
numerical/JIT source, operation tests and tuning. Preserve source headers, copyright and
applicable notices. The engine's distribution and import name is `cti`; the kernel's are
`cti-kernels` / `ctikernels`.

## Independent versions

Kernel and engine release independently, with no shared version number or release
cadence. Each project's version is defined once in its own `pyproject.toml`.
The engine selects one exact kernel dependency there and keeps it until an
explicit dependency-upgrade PR passes integration qualification. A kernel release
does not itself require any engine change. Kernels never depend on the engine.

Published versions identify immutable releases: do not replace an existing wheel's
bytes. New code needs a new version; development candidates may use standard
prerelease/development versions. Record actual versions and available artifact
hashes with validation. Hashes identify artifacts, not a second manually selected
kernel revision. Native Torch/CUDA/provider compatibility still needs qualification.

The engine requires `cti-kernels[cute,gdn,varlen]==0.0.1`, published only
as the [`v0.0.1` release](https://github.com/mit-han-lab/cti-kernel/releases/tag/v0.0.1)
of `mit-han-lab/cti-kernel`, not on a package index: download it, check its SHA256, and install it
with `--kernel-wheel` (below). Its artifact is:

| Artifact | SHA256 |
|---|---|
| `cti_kernels-0.0.1-py3-none-any.whl` | `6df0cc5b13bcef44a5672c215eb19ac4b23f010ccb7da53fdcba1e99e4317f9c` |

The same release carries the prebuilt flash-attn wheels that
[`environments/flash-attention-prebuilt.toml`](../environments/flash-attention-prebuilt.toml)
pins by SHA256.

There is no nested kernel source in the engine. Kernel tests run in the kernel repository;
engine and cross-package integration tests consume an installed kernel artifact,
not a sibling source PYTHONPATH.

The kernel repository owns operation correctness, gradients, mutation/aliasing contracts,
provider availability and operation-local routing, supported shapes/dtypes, workspace requirements,
operation-level graph replay and tuning. Engine operation tests use public kernel
interfaces or doubles at that boundary; they do not mock native operation symbols
or assert native symbol names. Engine retains model-level correctness, metadata,
resource lifetime, asynchronous ordering and end-to-end switching tests. Installer
checks of operation availability also use public operations; native package/ABI
and source provenance checks remain part of environment qualification, including
tests with simulated package and binary metadata.

Engine retains execution-backend selection, prefill/decode composition, cache layout,
graph strategy and TP/EP orchestration. Kernel support facts do not replace engine
integration qualification. Workspace requirements belong to the operation contract;
allocation, initialization and lifetime belong to its caller. Keep these boundaries
explicit without a cross-repository router or duplicate distributed runtime.

## Build and install

From clean, reviewed checkouts, build each wheel through its own sdist:

```bash
python -m pip install build
python -m build --outdir /path/to/artifacts/kernel /path/to/cti-kernel
python -m build --outdir /path/to/artifacts/engine /path/to/cti
sha256sum /path/to/artifacts/kernel/* /path/to/artifacts/engine/*
```

Native JIT sources, headers and their notices travel in kernel artifacts. Engine
artifacts contain only engine code. In a provisioned compatible environment:

```bash
python -m pip install --no-build-isolation \
  /path/to/artifacts/kernel/*.whl /path/to/artifacts/engine/*.whl
python -m pip check
```

Remove an earlier unpublished candidate first if it uses the same version, so pip
cannot silently keep it. To create a complete engine development Conda environment,
run from the engine checkout using Python 3.11 or newer:

```bash
python -m pip install packaging
python scripts/setup_env.py create --name cti \
  --kernel-wheel /path/to/cti_kernels-0.0.1-py3-none-any.whl
```

Setup installs the kernel version required by the engine and never builds kernel source
implicitly. `--kernel-wheel` takes the wheel downloaded from the kernel's GitHub Release,
or a locally built candidate (exactly one wheel) for qualification before publication. The
wheel's name/version are checked before changing an environment and its hash must match
after installation. [`scripts/install.sh`](../scripts/install.sh) does this download,
checks the pinned SHA256 and runs `setup_env.py create` in one step. The container image
([`docker/Dockerfile`](../docker/Dockerfile)) takes the same wheels through
`install.sh --fetch-only` and runs `setup_env.py create --no-gpu`, which leaves `verify` to
the first run on a GPU host. Both index and local-wheel installs request
`[cute,gdn,varlen]`; the formal profile verifies
`fla-core==0.5.2` and its `einops==0.8.2` dependency, and pins
`nvidia-cutlass-dsl[cu13]==4.8.0`, the `cute` extra's. The `varlen` extra's flash-attn is
the profile's locked version, installed before the kernel. On SM120/SM121 GPUs (RTX 5090,
RTX PRO 6000, DGX Spark), `--flash-attn-wheel` installs it from a prebuilt wheel: built
from the same locked source (`environments/flash-attention-source.txt`) against the
profile's Torch/CUDA with SM120 images, and pinned per machine by
`environments/flash-attention-prebuilt.toml`. Without it, setup builds FlashAttention from
that source for the visible GPUs. Verification accepts either provenance and records
which.

Verification checks installed versions, package ownership and native compatibility.
An ordinary index install is valid without `direct_url.json`; when origin metadata
exists it is recorded. An explicit `--kernel-wheel` requires a matching artifact
hash. Editable kernel development is allowed in a suitably provisioned environment,
but formal paired qualification uses an installed noneditable distribution.

## Tests and publication

Engine and integration tests run against the published, hash-checked kernel wheel that
the engine's exact kernel requirement names; they never clone or build kernel source. CPU
and GPU tests use the same artifact. A GPU environment must match the formal profile:
reinstalling the wheel offline with `--no-deps` keeps its GDN dependencies, and
environment verification rejects missing or mismatched provider pins. The kernel
repository qualifies its own changes independently.

CPU contract tests omit native GPU providers; successful import is not native
qualification. Complete GPU logs and cleanup checks remain required.

Release each repository after its own review and qualification. Publish a new
kernel version before an engine release that newly requires it; unrelated engine
and kernel releases need not wait for each other. Keep development-only
test/provenance checks out of release artifacts.

## Dependency checks and provenance

Project metadata owns package requirements; benchmark profiles freeze a qualified
environment. Operation-specific provider contracts and numerical tests belong to
the kernel repository, not the engine installer. Tests compare built/installed metadata with
the project's declarations instead of repeating current version literals.

The TMS requirement in engine `pyproject.toml` remains an immutable upstream commit
because the required MemPool fix was not distinguishable by the older wheel's
version. The benchmark profile separately records its reported metadata version.
Replace this source exception only after qualifying a release containing the fix.
FlashAttention's hashed source and prebuilt wheel locks and reviewed build targets remain
native-build requirements, not kernel package version selectors.

Keep artifact integrity, model/checkpoint identity, benchmark evidence and legal
attribution hashes. Benchmark receipts record installed engine/kernel versions and
available origin identity; editable mode is not a proof of immutable source content.
