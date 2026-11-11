"""``CheckpointFacts`` of a configuration alone, for models built without checkpoint files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cti.models import CheckpointFacts, CheckpointSource

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from cti.multimodal import ImagePreprocessing


def checkpoint_facts(
    hf_config: PretrainedConfig, *, image_preprocessing: ImagePreprocessing | None = None
) -> CheckpointFacts:
    """The facts of a checkpoint that is only ``hf_config`` (and ``image_preprocessing``): no
    files or stop set."""

    return CheckpointFacts(
        source=CheckpointSource("unused"),
        hf_config=hf_config,
        document={},
        stop_token_ids=(),
        image_preprocessing=image_preprocessing,
    )
