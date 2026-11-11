# cti container image

`ghcr.io/mit-han-lab/cti-engine:<version>`, where `<version>` is the engine version in
`pyproject.toml`, holds the environment [`scripts/install.sh`](../scripts/install.sh) creates
for SM120/SM121 GPUs: RTX 5090 and RTX PRO 6000 on `linux/amd64`, DGX Spark (GB10) on
`linux/arm64`. Its FlashAttention is the prebuilt SM120 wheel, so other GPUs need the install
script instead.

## Host requirements

- Linux with an NVIDIA driver 580 or newer.
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  which mounts the driver into the container.

## Run

```bash
IMAGE=ghcr.io/mit-han-lab/cti-engine:<version>
docker run --rm --gpus all "$IMAGE"
```

The default command prints the installed versions and the first GPU. Rootless Docker can
select GPUs through CDI instead: with the toolkit's
[CDI specification](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
in place, pass `--device nvidia.com/gpu=all`:

```bash
docker run --rm --device nvidia.com/gpu=all "$IMAGE"
```

CDI mounts the driver at its host path, and the image expects it in
`/usr/lib/<arch>-linux-gnu`, as on Ubuntu, Debian and DGX OS. On other hosts use `--gpus all`,
which rootless Docker supports with the toolkit's `no-cgroups = true`.

The image is built without a GPU, so verify it once on each host:

```bash
docker run --rm --gpus all --ipc=host "$IMAGE" python scripts/setup_env.py verify
```

`--qualification` also checks the host against a configuration in
[`environments/gpu-qualifications.toml`](../environments/gpu-qualifications.toml): `rtx5090`
needs exactly one visible RTX 5090 (`--gpus '"device=0"'` selects one), `rtx-pro-6000-pair`
exactly two RTX PRO 6000 and `dgx-spark` a DGX Spark. Omit it when none describes the host:

```bash
docker run --rm --gpus '"device=0"' --ipc=host "$IMAGE" \
  python scripts/setup_env.py verify --qualification rtx5090
```

Generate a few tokens, then serve on port 1919:

```bash
docker run --rm --gpus all --ipc=host -v cti-cache:/root/.cache "$IMAGE" \
  python scripts/smoke_check.py --model Qwen/Qwen3-0.6B
docker run --rm --gpus all --ipc=host -p 1919:1919 -v cti-cache:/root/.cache "$IMAGE" \
  python -m cti --model.path Qwen/Qwen3-0.6B --server.host 0.0.0.0
```

`docker run` starts every command in the activated environment from `/opt/cti`, the
source tree. `docker exec` skips that activation, which sets the compilers, CUDA paths and JIT
cache directories the kernels need, so activate first:

```bash
docker exec -it CONTAINER bash -c \
  'source /opt/conda/etc/profile.d/conda.sh && conda activate cti && exec python ...'
```

The image has no `benchmark/`: benchmarks record their Git state, so they run from a clone
installed with [`scripts/install.sh`](../scripts/install.sh).

`/root/.cache` holds Hugging Face downloads and the kernels compiled on first use (FlashInfer,
Triton, CuTe DSL, TVM FFI, Torch extensions); the `cti-cache` volume keeps them across
containers. `--ipc=host` gives the engine's processes and NCCL more shared memory than Docker's
64 MB default. Pass `-e HF_TOKEN` for gated models, or mount a local model directory with `-v`.

## Build

From the root of a clean clone (or a `git worktree add` of the commit passed as `REVISION`,
which the image records without checking the files against it), with Docker Buildx, one
platform at a time on a host of that architecture:

```bash
VERSION=$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml)
docker buildx build -f docker/Dockerfile --platform linux/amd64 --load \
  --build-arg REVISION="$(git rev-parse HEAD)" \
  -t "ghcr.io/mit-han-lab/cti-engine:$VERSION" .
```

`--platform linux/amd64,linux/arm64 --push` builds and pushes both under one tag when the
builder has a node of each architecture (`docker buildx create --append`); QEMU emulation
works too, but slowly. `REVISION` is required: the image carries no `.git`, and `verify`
records this commit as its `source_revision`.

The build takes the pinned cti-kernels wheel and this architecture's prebuilt flash-attn wheel
through `scripts/install.sh --fetch-only`, so the pins and SHA256 checks are the install
script's. For an offline or private build, place both release assets in `docker/wheels/`
first; the build checks their SHA256 and does not download them. The `MINIFORGE_VERSION` and
`MINIFORGE_SHA256_X86_64`/`MINIFORGE_SHA256_AARCH64` build arguments select another Miniforge
release.
