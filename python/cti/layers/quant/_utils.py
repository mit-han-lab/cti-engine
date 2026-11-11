"""Helpers the format modules share."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..placement import WeightShard


def block_windows(
    windows: tuple[WeightShard, ...], blocks: dict[int, int]
) -> tuple[WeightShard, ...]:
    """The windows over a tensor that holds one value per ``blocks[axis]`` elements of the
    weight's axis, for the weight's element ``windows``; each window starts and stops on a
    block, or this raises."""

    scaled = []
    for window in windows:
        block = blocks.get(window.axis, 1)
        start, stop = window.start, window.stop
        if start % block or stop % block:
            raise ValueError(
                f"the window [{start}, {stop}) on axis {window.axis} cuts its "
                f"{block}-element blocks"
            )
        if window.padded_size is not None:
            raise ValueError("a padded window has no block scales")
        scaled.append(
            WeightShard(
                window.axis,
                start // block,
                stop // block,
                replicas=window.replicas,
                replica=window.replica,
            )
        )
    return tuple(scaled)


def provider_row(providers: Mapping[str, Any], device: torch.device) -> str:
    """The name of ``device``'s row of a format's provider table; a device without one is
    refused."""

    if device.type not in providers:
        raise ValueError(f"no provider computes this format on {device.type} ({sorted(providers)})")
    return providers[device.type].__name__
