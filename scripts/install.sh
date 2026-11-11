#!/usr/bin/env bash
# One-step cti install: checks prerequisites, fetches the pinned cti-kernels wheel
# (and a prebuilt flash-attn wheel on SM120/SM121 GPUs), creates the Conda environment
# through scripts/setup_env.py, and runs a smoke check. With --fetch-only it only fetches
# and checks the wheels, as the container image's build does.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The wheel's SHA256 belongs to this exact version; a pin change in pyproject.toml must
# update both lines.
PINNED_KERNEL_VERSION="0.0.1"
PINNED_KERNEL_SHA256="6df0cc5b13bcef44a5672c215eb19ac4b23f010ccb7da53fdcba1e99e4317f9c"
# Prebuilt flash-attn wheels, assets of every cti-kernel release beside the kernel wheel.
# These must match environments/flash-attention-prebuilt.toml.
PINNED_FLASH_ATTN_VERSION="2.8.3.post1"
PINNED_FLASH_ATTN_SHA256_X86_64="3e8f6edd3c2f20ec3a668f7c60c6da8ec0adfb5517979bb1ee0b50e06927e3b7"
PINNED_FLASH_ATTN_SHA256_AARCH64="d29013ab8bbcfab10f1df2b9e0d834f5ff807e6d4004fd1b0a4ec5c0bb235493"
MIN_DRIVER_MAJOR=580 # CUDA 13 (the profile's PyTorch cu130 build)

ENV_NAME="${CTI_ENV:-cti}"
KERNEL_REPO="${KERNEL_REPO:-mit-han-lab/cti-kernel}"
KERNEL_WHEEL="${KERNEL_WHEEL:-}"
KERNEL_WHEEL_URL="${KERNEL_WHEEL_URL:-}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-}"
BUILD_FLASH_ATTN="${BUILD_FLASH_ATTN:-0}"
CACHE_DIR="${CTI_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/cti}"
QUALIFICATION=""
MODEL=""
FETCH_ONLY=""
RECREATE=0
SMOKE=1

usage() {
  cat <<'EOF'
Usage: bash scripts/install.sh [options]

Creates a Conda environment with CUDA, PyTorch, the pinned cti-kernels wheel and an
editable cti, verifies it, and imports both packages. Rerunning it verifies an
existing environment instead of rebuilding it.

Options:
  -n, --name NAME          Conda environment name (default: cti; env CTI_ENV)
      --kernel-wheel PATH  install this local cti-kernels wheel (env KERNEL_WHEEL)
      --kernel-wheel-url URL
                           download the wheel from URL (env KERNEL_WHEEL_URL)
      --flash-attn-wheel PATH
                           install this prebuilt flash-attn wheel (env FLASH_ATTN_WHEEL)
      --build-flash-attn   build flash-attn from source on any GPU (env BUILD_FLASH_ATTN=1)
      --qualification Q    also require a GPU qualification from
                           environments/gpu-qualifications.toml (rtx5090, dgx-spark, ...)
      --model PATH         after installing, generate a few tokens offline with this
                           model (a local directory or a Hugging Face id)
      --recreate           remove an existing environment of the same name first
      --no-smoke           skip the import check (--model still runs it)
      --fetch-only DIR     only fetch the cti-kernels wheel and this machine's prebuilt
                           flash-attn wheel, check their SHA256, copy them into DIR and
                           exit; no GPU, driver or conda checks
  -h, --help               show this help

Any conda installation works: Miniforge (which also ships mamba), Miniconda or Anaconda.
By default the wheel comes from the kernel repository's GitHub release
(KERNEL_REPO, default mit-han-lab/cti-kernel) over HTTPS, or through `gh release download`
when that fails and the GitHub CLI is installed (gh auth login or GH_TOKEN authenticates it).
The kernel wheel must match the pinned SHA256; KERNEL_WHEEL_SHA256 overrides it.
When every GPU reports compute capability 12.0 or 12.1 (RTX 5090, RTX PRO 6000, DGX Spark),
flash-attn comes from a prebuilt wheel on the same KERNEL_REPO release as the kernel wheel,
checked against the SHA256 environments/flash-attention-prebuilt.toml pins. On other GPUs, or when that download fails, flash-attn is built
from source: the slow step (about 10-20 minutes), whose parallelism MAX_JOBS caps.
Downloads are cached in $XDG_CACHE_HOME/cti (or CTI_CACHE_DIR).
EOF
}

log() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    -n | --name) ENV_NAME="${2:?--name needs a value}"; shift 2 ;;
    --kernel-wheel) KERNEL_WHEEL="${2:?--kernel-wheel needs a path}"; shift 2 ;;
    --kernel-wheel-url) KERNEL_WHEEL_URL="${2:?--kernel-wheel-url needs a URL}"; shift 2 ;;
    --flash-attn-wheel) FLASH_ATTN_WHEEL="${2:?--flash-attn-wheel needs a path}"; shift 2 ;;
    --build-flash-attn) BUILD_FLASH_ATTN=1; shift ;;
    --qualification) QUALIFICATION="${2:?--qualification needs a name}"; shift 2 ;;
    --model) MODEL="${2:?--model needs a path or id}"; shift 2 ;;
    --recreate) RECREATE=1; shift ;;
    --no-smoke) SMOKE=0; shift ;;
    --fetch-only) FETCH_ONLY="${2:?--fetch-only needs a directory}"; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
done
[[ -z "$FLASH_ATTN_WHEEL" || "$BUILD_FLASH_ATTN" != 1 ]] \
  || die "--flash-attn-wheel and --build-flash-attn are mutually exclusive"
[[ -z "$FETCH_ONLY" || (-z "$FLASH_ATTN_WHEEL" && "$BUILD_FLASH_ATTN" != 1) ]] \
  || die "--fetch-only fetches this machine's prebuilt flash-attn wheel; drop --flash-attn-wheel and --build-flash-attn"

# --- prerequisites -------------------------------------------------------------------
[[ "$(uname -s)" == Linux ]] || die "cti supports Linux only (found $(uname -s))"
case "$(uname -m)" in
  x86_64 | aarch64) ;;
  *) die "unsupported CPU architecture $(uname -m); x86_64 or aarch64 is required" ;;
esac

if command -v sha256sum >/dev/null; then
  sha256() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null; then
  sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  die "sha256sum (coreutils) or shasum is required"
fi

# Only the install needs the GPU, conda and a checkout; --fetch-only needs none of them.
if [[ -z "$FETCH_ONLY" ]]; then
  CONDA="${CONDA_EXE:-$(command -v conda || true)}"
  [[ -n "$CONDA" && -x "$CONDA" ]] \
    || die "conda was not found; install any conda (Miniforge: https://github.com/conda-forge/miniforge) and open a new shell"

  command -v nvidia-smi >/dev/null || die "nvidia-smi was not found; install the NVIDIA driver (>= $MIN_DRIVER_MAJOR)"
  GPUS="$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null)" \
    || die "nvidia-smi could not query a GPU; check the NVIDIA driver and that a GPU is visible"
  [[ -n "$GPUS" ]] || die "no NVIDIA GPU is visible"
  DRIVER="$(head -n 1 <<<"$GPUS" | cut -d, -f2 | tr -d ' ')"
  [[ "${DRIVER%%.*}" =~ ^[0-9]+$ ]] || die "could not read the NVIDIA driver version ($DRIVER)"
  ((${DRIVER%%.*} >= MIN_DRIVER_MAJOR)) \
    || die "NVIDIA driver $DRIVER is too old; CUDA 13 needs $MIN_DRIVER_MAJOR or newer"
  log "GPUs (name, driver, compute capability):"
  while IFS= read -r gpu; do printf '    %s\n' "$gpu"; done <<<"$GPUS"

  git -C "$REPO_ROOT" rev-parse HEAD >/dev/null 2>&1 \
    || die "$REPO_ROOT is not a git checkout; clone the repository instead of copying it"

  # setup_env.py needs Python >= 3.11 with packaging; Conda's own Python has both.
  PYTHON=""
  for candidate in "${PYTHON_FOR_SETUP:-}" "$(dirname "$CONDA")/python" python3 python; do
    if [[ -z "$candidate" ]] || ! command -v "$candidate" >/dev/null; then continue; fi
    if "$candidate" -c 'import sys, packaging; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
      PYTHON="$candidate"
      break
    fi
  done
  [[ -n "$PYTHON" ]] \
    || die "no Python >= 3.11 with 'packaging' found; run: python3 -m pip install packaging (or set PYTHON_FOR_SETUP)"
fi

# --- the pinned cti-kernels wheel ----------------------------------------------------
VERSION="$(sed -nE 's/.*"cti-kernels\[[^]]*\]==([^"]+)".*/\1/p' "$REPO_ROOT/pyproject.toml")"
[[ -n "$VERSION" ]] || die "pyproject.toml has no exact cti-kernels requirement"
EXPECTED_SHA256="${KERNEL_WHEEL_SHA256:-}"
if [[ -z "$EXPECTED_SHA256" ]]; then
  [[ "$VERSION" == "$PINNED_KERNEL_VERSION" ]] \
    || die "pyproject.toml pins cti-kernels $VERSION but this script knows the hash of $PINNED_KERNEL_VERSION; set KERNEL_WHEEL_SHA256"
  EXPECTED_SHA256="$PINNED_KERNEL_SHA256"
fi
WHEEL_NAME="cti_kernels-$VERSION-py3-none-any.whl"

check_sha256() { # path expected -> fails unless the file's SHA256 is the expected one
  local actual
  actual="$(sha256 "$1")"
  [[ "$actual" == "$2" ]] || {
    printf 'error: %s has SHA256 %s, expected %s\n' "$1" "$actual" "$2" >&2
    return 1
  }
}

fetch_asset() { # tag name destination [url] -> downloads a KERNEL_REPO release asset, or fails
  local tag="$1" name="$2" destination="$3" custom_url="${4:-}"
  local url="${custom_url:-https://github.com/$KERNEL_REPO/releases/download/$tag/$name}"
  local part="$destination.part"
  rm -f "$part"
  log "Downloading $name from $url"
  if command -v curl >/dev/null && curl -fL --retry 3 --proto '=https' -o "$part" "$url"; then
    mv "$part" "$destination"
    return
  fi
  rm -f "$part"
  # `gh auth status` also fails for a working token that lacks an optional scope, so try
  # the download itself whenever gh is installed.
  if [[ -z "$custom_url" ]] && command -v gh >/dev/null; then
    log "Retrying with gh release download (GitHub CLI)"
    local staging
    staging="$(mktemp -d "$CACHE_DIR/gh.XXXXXX")" || return 1
    if gh release download "$tag" --repo "$KERNEL_REPO" --pattern "$name" --dir "$staging"; then
      mv "$staging/$name" "$destination"
      rmdir "$staging"
      return
    fi
    rm -rf "$staging"
  fi
  return 1
}

if [[ -n "$KERNEL_WHEEL" ]]; then
  [[ -f "$KERNEL_WHEEL" ]] || die "kernel wheel not found: $KERNEL_WHEEL"
  WHEEL="$(cd "$(dirname "$KERNEL_WHEEL")" && pwd)/$(basename "$KERNEL_WHEEL")"
  check_sha256 "$WHEEL" "$EXPECTED_SHA256" || die "refusing a wheel that does not match the pinned hash"
else
  mkdir -p "$CACHE_DIR/wheels"
  WHEEL="$CACHE_DIR/wheels/$WHEEL_NAME"
  if [[ -f "$WHEEL" ]] && check_sha256 "$WHEEL" "$EXPECTED_SHA256" 2>/dev/null; then
    log "Using cached $WHEEL"
  else
    rm -f "$WHEEL"
    fetch_asset "v$VERSION" "$WHEEL_NAME" "$WHEEL" "$KERNEL_WHEEL_URL" \
      || die "could not download $WHEEL_NAME from ${KERNEL_WHEEL_URL:-$KERNEL_REPO}; download the release asset yourself and pass --kernel-wheel PATH or place it in $CACHE_DIR/wheels"
    check_sha256 "$WHEEL" "$EXPECTED_SHA256" \
      || { rm -f "$WHEEL"; die "downloaded wheel failed the SHA256 check"; }
  fi
fi
log "cti-kernels $VERSION wheel verified (SHA256 $EXPECTED_SHA256)"

# --- flash-attn ----------------------------------------------------------------------
sm120_family_only() { # every GPU nvidia-smi reports has compute capability 12.0 or 12.1
  local caps cap
  caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null)" || return 1
  [[ -n "$caps" ]] || return 1
  while IFS= read -r cap; do
    cap="${cap//[[:space:]]/}"
    [[ "$cap" == 12.0 || "$cap" == 12.1 ]] || return 1
  done <<<"$caps"
}

fetch_prebuilt_flash_attn() { # sets FLASH_ATTN_WHEEL to this machine's verified prebuilt wheel, or fails
  local machine expected name wheel
  machine="$(uname -m)"
  case "$machine" in
    x86_64) expected="$PINNED_FLASH_ATTN_SHA256_X86_64" ;;
    aarch64) expected="$PINNED_FLASH_ATTN_SHA256_AARCH64" ;;
    *) return 1 ;;
  esac
  name="flash_attn-$PINNED_FLASH_ATTN_VERSION-cp312-cp312-linux_$machine.whl"
  mkdir -p "$CACHE_DIR/wheels"
  wheel="$CACHE_DIR/wheels/$name"
  if [[ -f "$wheel" ]] && check_sha256 "$wheel" "$expected" 2>/dev/null; then
    log "Using cached $wheel"
  else
    rm -f "$wheel"
    if ! { fetch_asset "v$VERSION" "$name" "$wheel" \
      && check_sha256 "$wheel" "$expected"; }; then
      rm -f "$wheel"
      return 1
    fi
  fi
  log "Prebuilt $name verified (SHA256 $expected)"
  FLASH_ATTN_WHEEL="$wheel"
}

choose_flash_attn() { # sets FLASH_ATTN_WHEEL to a prebuilt wheel, or leaves it empty to build
  if [[ -n "$FLASH_ATTN_WHEEL" ]]; then
    [[ -f "$FLASH_ATTN_WHEEL" ]] || die "flash-attn wheel not found: $FLASH_ATTN_WHEEL"
    FLASH_ATTN_WHEEL="$(cd "$(dirname "$FLASH_ATTN_WHEEL")" && pwd)/$(basename "$FLASH_ATTN_WHEEL")"
    log "Using flash-attn wheel $FLASH_ATTN_WHEEL"
    return
  fi
  if [[ "$BUILD_FLASH_ATTN" == 1 ]]; then
    log "flash-attn will be built from source (--build-flash-attn; about 10-20 minutes)"
    return
  fi
  if ! sm120_family_only; then
    log "flash-attn will be built from source for these GPUs (about 10-20 minutes)"
    return
  fi
  fetch_prebuilt_flash_attn \
    || warn "no verified prebuilt flash-attn wheel; flash-attn will be built from source (about 10-20 minutes)"
}

# --fetch-only stops here, with both verified wheels copied into its directory.
if [[ -n "$FETCH_ONLY" ]]; then
  fetch_prebuilt_flash_attn \
    || die "could not fetch a verified prebuilt flash-attn wheel from $KERNEL_REPO; place the release asset in $CACHE_DIR/wheels"
  mkdir -p "$FETCH_ONLY"
  for wheel in "$WHEEL" "$FLASH_ATTN_WHEEL"; do
    [[ "$wheel" -ef "$FETCH_ONLY/${wheel##*/}" ]] || cp "$wheel" "$FETCH_ONLY/"
  done
  log "Copied the verified wheels into $FETCH_ONLY"
  exit 0
fi

# --- the environment -----------------------------------------------------------------
env_exists() {
  "$CONDA" env list --json | "$PYTHON" -c '
import json, os, sys
envs = json.load(sys.stdin)["envs"]
sys.exit(0 if any(os.path.basename(p) == sys.argv[1] for p in envs) else 1)' "$1"
}
in_env() { "$CONDA" run --no-capture-output -n "$ENV_NAME" "$@"; }

SETUP_ARGS=(--kernel-wheel "$WHEEL")
[[ -n "$QUALIFICATION" ]] && SETUP_ARGS+=(--qualification "$QUALIFICATION")

if env_exists "$ENV_NAME" && ((RECREATE)); then
  log "Removing existing environment $ENV_NAME (--recreate)"
  "$CONDA" env remove -n "$ENV_NAME" --yes
fi
if env_exists "$ENV_NAME"; then
  log "Environment $ENV_NAME exists; verifying it (use --recreate to rebuild)"
  in_env python "$REPO_ROOT/scripts/setup_env.py" verify "${SETUP_ARGS[@]}" >/dev/null \
    || die "$ENV_NAME failed verification; rerun with --recreate"
else
  choose_flash_attn
  CREATE_ARGS=("${SETUP_ARGS[@]}")
  [[ -n "$FLASH_ATTN_WHEEL" ]] && CREATE_ARGS+=(--flash-attn-wheel "$FLASH_ATTN_WHEEL")
  log "Creating $ENV_NAME (CUDA, PyTorch, FlashAttention, cti-kernels, cti)"
  "$PYTHON" "$REPO_ROOT/scripts/setup_env.py" create --conda "$CONDA" --name "$ENV_NAME" "${CREATE_ARGS[@]}"
fi

# --- smoke check ---------------------------------------------------------------------
SMOKE_ARGS=()
[[ -n "$MODEL" ]] && SMOKE_ARGS+=(--model "$MODEL")
if ((SMOKE)) || [[ -n "$MODEL" ]]; then
  log "Smoke check${MODEL:+ (and 16 tokens generated with $MODEL)}"
  in_env python "$REPO_ROOT/scripts/smoke_check.py" "${SMOKE_ARGS[@]}"
fi

log "Done. Activate the environment with: conda activate $ENV_NAME"
