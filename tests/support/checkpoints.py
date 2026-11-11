"""Every checkpoint the tests pin, declared once, and the one way to read one from the local
Hugging Face cache.

No other test module constructs a ``Checkpoint`` or calls ``snapshot_download``.
tests/gpu/checkpoint_manifest.json lists the checkpoints GPU tests read, with their cached
files and sizes and, per qualification and shard, which ones each shard needs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

RTX5090 = "rtx5090"
RTX_PRO_6000_PAIR = "rtx-pro-6000-pair"
_EVERY_RUNNER = (RTX5090, RTX_PRO_6000_PAIR)
_PRO_ONLY = (RTX_PRO_6000_PAIR,)
# Configs, tokenizer files and weights; a repo's README, images and remote code need not be cached.
CACHED_FILES = ("*.json", "*.safetensors")


@dataclass(frozen=True)
class Checkpoint:
    """A checkpoint at a pinned revision. Two pins are equal when repo and revision are."""

    repo_id: str
    revision: str
    # The GPU qualifications whose devices hold its BF16 weights.
    fits: tuple[str, ...] = field(default=_EVERY_RUNNER, compare=False)
    # A Qwen3.5-MoE checkpoint, judged by the MoE consistency rules (tests/support/moe.py).
    moe: bool = field(default=False, compare=False)
    # The pin whose parity prompt ids it reads, which its tokenizer and chat template render too.
    prompt_ids_of: Checkpoint | None = field(default=None, compare=False)

    @property
    def key(self) -> str:
        """``REPO@REVISION``: the parity prompts' token-id set of this checkpoint."""

        return f"{self.repo_id}@{self.revision}"

    @property
    def slug(self) -> str:
        return self.repo_id.replace("/", "--")

    def snapshot(self, cache_dir: Path | None = None) -> Path:
        """The checkpoint's folder in the local Hugging Face cache."""

        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            folder = snapshot_download(
                repo_id=self.repo_id,
                revision=self.revision,
                cache_dir=cache_dir,
                allow_patterns=list(CACHED_FILES),
                local_files_only=True,
            )
        except LocalEntryNotFoundError as error:
            raise FileNotFoundError(
                f"{self.key} is not in the local Hugging Face cache; provision it there "
                f"before running the GPU tests"
            ) from error
        return Path(folder)


QWEN3_0_6B = Checkpoint("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca")
QWEN3_1_7B = Checkpoint("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
QWEN3_4B = Checkpoint("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c")
QWEN3_8B = Checkpoint("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")
QWEN3_8B_DFLASH_B16 = Checkpoint(
    "z-lab/Qwen3-8B-DFlash-b16", "9b41424b7109f9c5413454f481b09a82b85333f4"
)

QWEN3_30B_A3B = Checkpoint(
    "Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", _PRO_ONLY
)
QWEN3_30B_A3B_INSTRUCT_2507 = Checkpoint(
    "Qwen/Qwen3-30B-A3B-Instruct-2507", "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe", _PRO_ONLY
)
QWEN3_30B_A3B_THINKING_2507 = Checkpoint(
    "Qwen/Qwen3-30B-A3B-Thinking-2507", "144afc2f379b542fdd4e85a1fcd5e1f79112d95d", _PRO_ONLY
)
QWEN3_30B_A3B_BASE = Checkpoint(
    "Qwen/Qwen3-30B-A3B-Base", "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9", _PRO_ONLY
)
QWEN3_CODER_30B_A3B_INSTRUCT = Checkpoint(
    "Qwen/Qwen3-Coder-30B-A3B-Instruct", "b2cff646eb4bb1d68355c01b18ae02e7cf42d120", _PRO_ONLY
)
# No qualification holds the 235B-A22B weights; the tests read only their configs.
QWEN3_235B_A22B = Checkpoint("Qwen/Qwen3-235B-A22B", "8efa61729e24bd65b1d152b5ab5409052aa80e65", ())
QWEN3_235B_A22B_INSTRUCT_2507 = Checkpoint(
    "Qwen/Qwen3-235B-A22B-Instruct-2507", "ac9c66cc9b46af7306746a9250f23d47083d689e", ()
)

QWEN3_5_9B = Checkpoint("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
# The small sizes render the parity prompts as Qwen3.5-9B does, so they read its ids.
QWEN3_5_0_8B = Checkpoint(
    "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", prompt_ids_of=QWEN3_5_9B
)
QWEN3_5_2B = Checkpoint(
    "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", prompt_ids_of=QWEN3_5_9B
)
QWEN3_5_4B = Checkpoint(
    "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", prompt_ids_of=QWEN3_5_9B
)
QWEN3_5_9B_DFLASH = Checkpoint(
    "z-lab/Qwen3.5-9B-DFlash", "5fc3b3d474760f18c516db87d84c37edbfd3ede6"
)
QWEN3_8_27B = Checkpoint("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", _PRO_ONLY)
QWEN3_8_27B_DFLASH2 = Checkpoint(
    "z-lab/Qwen3.8-27B-DFlash2", "50307d4c4cde6860d4eee73e2547cd786fe8e8a4", _PRO_ONLY
)
QWEN3_5_35B_A3B = Checkpoint(
    "Qwen/Qwen3.5-35B-A3B", "59d61f3ce65a6d9863b86d2e96597125219dc754", _PRO_ONLY, moe=True
)
# The quantization gates' checkpoints (tests/hf_parity/test_quant_gates.py); each judged against
# its BF16 sibling above.
QWEN3_8B_FP8 = Checkpoint("Qwen/Qwen3-8B-FP8", "220b46e3b2180893580a4454f21f22d3ebb187d3")
QWEN3_8_27B_NVFP4 = Checkpoint(
    "nvidia/Qwen3.8-27B-NVFP4", "482ca0f3832238542f8f5295dde86b5f22711d80"
)
QWEN3_8_27B_FP8 = Checkpoint(
    "Qwen/Qwen3.8-27B-FP8", "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a", _PRO_ONLY
)
QWEN3_5_35B_A3B_FP8 = Checkpoint(
    "Qwen/Qwen3.5-35B-A3B-FP8", "9d1823d2dee688a6b25e77009dc727688c44936e", _PRO_ONLY, moe=True
)
QWEN3_5_35B_A3B_DFLASH = Checkpoint(
    "z-lab/Qwen3.5-35B-A3B-DFlash", "52cb554b4995dede3e2e1bdb129cdb1f3529332b", _PRO_ONLY
)

PINS = tuple(value for value in tuple(globals().values()) if isinstance(value, Checkpoint))


def pinned(key: str) -> Checkpoint:
    """The pin whose ``key`` is ``REPO@REVISION``."""

    for checkpoint in PINS:
        if checkpoint.key == key:
            return checkpoint
    raise KeyError(f"{key} is not a pinned checkpoint (tests/support/checkpoints.py)")


def checkpoints_of(item: pytest.Item) -> frozenset[Checkpoint]:
    """The checkpoints a collected test reads: those in its parameters (inside dataclasses and
    containers too) and those its ``checkpoints`` markers declare."""

    params = getattr(getattr(item, "callspec", None), "params", {})
    found = set(_checkpoints_in(params))
    for marker in item.iter_markers("checkpoints"):
        found.update(marker.args)
    return frozenset(found)


def _checkpoints_in(value: Any) -> Iterator[Checkpoint]:
    if isinstance(value, Checkpoint):
        yield value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for item in dataclasses.fields(value):
            yield from _checkpoints_in(getattr(value, item.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _checkpoints_in(item)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            yield from _checkpoints_in(item)
