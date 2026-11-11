# syntax=docker/dockerfile:1
# The cti image for SM120/SM121 GPUs (RTX 5090, RTX PRO 6000, DGX Spark), built for
# linux/amd64 and linux/arm64 from the repository root as docker/README.md shows.
FROM ubuntu:24.04

# Miniforge, and its installer's SHA256 for each machine `uname -m` names.
ARG MINIFORGE_VERSION=26.9.4-0
ARG MINIFORGE_SHA256_X86_64=621428f40496a90702753b09237794518a081d9979d9f845a525de761b6056f7
ARG MINIFORGE_SHA256_AARCH64=d265b10be348403a3bb8d475ff989beba48db9baa5fe7ee28eb38213538056bb

# git: pip installs torch-memory-saver from its locked commit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

RUN set -eu; \
    machine="$(uname -m)"; \
    case "$machine" in \
        x86_64) sha256="$MINIFORGE_SHA256_X86_64" ;; \
        aarch64) sha256="$MINIFORGE_SHA256_AARCH64" ;; \
        *) echo "unsupported machine: $machine" >&2; exit 1 ;; \
    esac; \
    installer="Miniforge3-$MINIFORGE_VERSION-Linux-$machine.sh"; \
    curl -fsSL --retry 3 -o "/tmp/$installer" \
        "https://github.com/conda-forge/miniforge/releases/download/$MINIFORGE_VERSION/$installer"; \
    echo "$sha256  /tmp/$installer" | sha256sum -c -; \
    bash "/tmp/$installer" -b -p /opt/conda; \
    rm "/tmp/$installer"; \
    /opt/conda/bin/conda clean --all --yes

WORKDIR /opt/cti
# .dockerignore admits only the sources the image needs; the wheels in docker/wheels/ stay out
# of every layer (the fetch step below mounts them).
COPY --exclude=docker/wheels . .

# verify records the source revision, which a tree without .git reads from REVISION.
ARG REVISION
LABEL org.opencontainers.image.source="https://github.com/mit-han-lab/cti-engine" \
      org.opencontainers.image.revision="$REVISION"
RUN test -n "$REVISION" || { echo 'pass --build-arg REVISION="$(git rev-parse HEAD)"' >&2; exit 1; }; \
    echo "$REVISION" > REVISION

# The pinned cti-kernels and flash-attn wheels, checked against scripts/install.sh's SHA256
# pins: those placed in docker/wheels/ before the build, else downloaded from the kernel
# release into the writable mount, which discards them. The build has no GPU, so the
# environment records the library directory the NVIDIA Container Toolkit mounts the driver
# into, and verification waits for a GPU host.
RUN --mount=type=bind,source=docker,target=/tmp/docker,rw \
    CTI_CACHE_DIR=/tmp/docker bash scripts/install.sh --fetch-only /tmp/wheels \
    && /opt/conda/bin/python scripts/setup_env.py create --no-gpu --name cti \
        --conda /opt/conda/bin/conda \
        --kernel-wheel /tmp/wheels/cti_kernels-*.whl \
        --flash-attn-wheel /tmp/wheels/flash_attn-*.whl \
        --cuda-driver-dir "/usr/lib/$(uname -m)-linux-gnu" \
        --cache-dir /root/.cache/cti \
    && rm -rf /tmp/wheels /root/.cache/pip \
    && /opt/conda/bin/conda clean --all --yes

# Hugging Face downloads and JIT-compiled kernels go under /root/.cache: mount a volume there
# to keep them across containers.
ENV PATH=/opt/conda/envs/cti/bin:/opt/conda/bin:$PATH \
    HF_HOME=/root/.cache/huggingface \
    TRITON_CACHE_DIR=/root/.cache/triton

# Every command runs in the activated environment, which sets the variables setup_env.py
# recorded in it (CUDA_HOME, the library paths, the JIT cache directories).
ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate cti && exec \"$@\"", "--"]
CMD ["python", "scripts/smoke_check.py"]
