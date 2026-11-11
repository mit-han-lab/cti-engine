"""Vision graphs: a vision tower's blocks and merger replayed at token budgets.

A step's images are sorted by tokens and packed greedily, each pack replaying the smallest
budget that holds it. The budget's graph attends each image as one sequence of the tower's
variable-length attention and its padding as one more, so padding never attends with an
image; an image past the largest budget runs eagerly alone. Patch and position embeddings
and the rotary tables stay eager (``VisionTower.embed``), as in vLLM and SGLang.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING

import torch

from cti.forward import ImageEncoding

from ._graphs import capture_graphs

if TYPE_CHECKING:
    from cti.forward import ImageBatch, VisionTower
    from cti.inference._step import GraphCapture
    from cti.memory import GraphPool


@dataclass(frozen=True, slots=True)
class VisionPack:
    """Images one tower call encodes, in this order: through ``budget``'s graph, or eagerly
    without one."""

    budget: int | None
    images: tuple[int, ...]


@dataclass(frozen=True)
class VisionBudgets:
    """The ascending token budgets vision graphs are captured at. A budget's graph encodes
    at most ``max_images``: as many as it holds of the smallest budget, the smallest image's
    tokens."""

    sizes: tuple[int, ...]

    def max_images(self, budget: int) -> int:
        return budget // self.sizes[0]

    def select(self, tokens: int, images: int) -> int | None:
        """The smallest budget that holds ``images`` images of ``tokens`` tokens, or None."""

        fits = (b for b in self.sizes if b >= tokens and images <= self.max_images(b))
        return next(fits, None)

    def pack(self, tokens: Sequence[int]) -> tuple[VisionPack, ...]:
        """Packs of the images of ``tokens`` tokens each: by ascending tokens, an image joins
        the open pack while a budget holds them all, else starts the next; then each image
        past the largest budget alone, eagerly."""

        order = sorted(range(len(tokens)), key=tokens.__getitem__)
        fitting = [image for image in order if tokens[image] <= self.sizes[-1]]
        packs: list[VisionPack] = []
        images: list[int] = []
        total = 0
        for image in fitting:
            if images and self.select(total + tokens[image], len(images) + 1) is None:
                packs.append(VisionPack(self.select(total, len(images)), tuple(images)))
                images, total = [], 0
            images.append(image)
            total += tokens[image]
        if images:
            packs.append(VisionPack(self.select(total, len(images)), tuple(images)))
        return (*packs, *(VisionPack(None, (image,)) for image in order[len(fitting) :]))

    def boundaries(self, budget: int, patches: Sequence[int], patches_per_token: int) -> list[int]:
        """``budget``'s sequence bounds for a pack of images of ``patches`` patches, in pack
        order: each image, then the padding up to the budget's patches as one sequence,
        then empty ones up to the graph's fixed count, ``max_images + 1``."""

        padded = budget * patches_per_token
        empty = self.max_images(budget) + 1 - len(patches)
        return [0, *accumulate(patches), *(padded,) * empty]


class VisionGraph(ImageEncoding):
    """A vision tower's blocks and merger replayed from one graph per token budget, in a
    pool of their own; an image no budget holds runs eagerly. Replays read static inputs
    sized for the largest budget, and each copies its tokens out before the next."""

    def __init__(
        self,
        *,
        tower: VisionTower,
        budgets: VisionBudgets,
        pool: GraphPool,
        stream: torch.cuda.Stream,
        device: torch.device,
    ) -> None:
        self.budgets = budgets
        self._per_token = tower.merge_size**2
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._outputs: dict[int, torch.Tensor] = {}
        # Images encoded by a graph replay or eagerly since capture.
        self.replayed = self.eager = 0

        def capture() -> None:
            largest = budgets.sizes[-1]
            with torch.inference_mode(), torch.cuda.stream(stream):
                blank = tower.largest_image_batch(
                    max_tokens=largest, max_image_tokens=largest, device=device
                )
                # Static outside the pool: the largest budget's inputs and sequence bounds.
                self._inputs = tower.embed(blank)
                self._bounds = torch.zeros(
                    budgets.max_images(largest) + 2, dtype=torch.int32, device=device
                )
                # Largest first: smaller graphs reuse its blocks.
                for budget in reversed(budgets.sizes):
                    self._capture(tower, budget, pool, stream)

        self.captured: GraphCapture = capture_graphs(
            "vision",
            budgets.sizes,
            capture,
            pool=pool,
            device=device,
            max_rows=budgets.sizes[-1] * self._per_token,
        )

    def encode(self, tower: VisionTower, images: ImageBatch) -> torch.Tensor:
        inputs = tower.embed(images)
        patches = [frames * height * width for frames, height, width in images.grids]
        starts = [0, *accumulate(patches)]
        tokens = [count // self._per_token for count in patches]
        token_starts = [0, *accumulate(tokens)]
        output = None
        for pack in self.budgets.pack(tokens):
            spans = [(starts[image], patches[image]) for image in pack.images]
            if pack.budget is None:
                self.eager += 1
                start, count = spans[0]
                host = torch.tensor([0, count], dtype=torch.int32, pin_memory=True)
                rows = tuple(tensor[start : start + count] for tensor in inputs)
                encoded = tower.encode(rows, host.to(rows[0].device, non_blocking=True), count)
            else:
                self.replayed += len(pack.images)
                encoded = self._replay(pack.budget, inputs, spans)
            if output is None:
                output = encoded.new_empty((token_starts[-1], encoded.shape[1]))
            offset = 0
            for image in pack.images:
                count = tokens[image]
                output[token_starts[image] : token_starts[image] + count].copy_(
                    encoded[offset : offset + count]
                )
                offset += count
        return output

    def close(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        gc.collect()

    def _replay(
        self, budget: int, inputs: tuple[torch.Tensor, ...], spans: list[tuple[int, int]]
    ) -> torch.Tensor:
        """``budget``'s tokens of the images at ``spans`` (first patch, patches) of
        ``inputs``, packed in order; valid until the next replay."""

        offset = 0
        for start, count in spans:
            for static, tensor in zip(self._inputs, inputs, strict=True):
                static[offset : offset + count].copy_(tensor[start : start + count])
            offset += count
        bounds = self.budgets.boundaries(budget, [count for _, count in spans], self._per_token)
        host = torch.tensor(bounds, dtype=torch.int32, pin_memory=True)
        self._bounds[: len(bounds)].copy_(host, non_blocking=True)
        self._graphs[budget].replay()
        return self._outputs[budget]

    def _capture(
        self, tower: VisionTower, budget: int, pool: GraphPool, stream: torch.cuda.Stream
    ) -> None:
        """On ``stream``, warm up, then capture ``budget``'s graph over one image filling it;
        its output stays in the pool."""

        patches = budget * self._per_token
        inputs = tuple(tensor[:patches] for tensor in self._inputs)
        bounds = self.budgets.boundaries(budget, [patches], self._per_token)
        cu_patches = self._bounds[: len(bounds)]
        cu_patches.copy_(torch.tensor(bounds, dtype=torch.int32))
        tower.encode(inputs, cu_patches, patches)
        self._graphs[budget] = graph = torch.cuda.CUDAGraph()
        with pool.capture(graph, stream):
            self._outputs[budget] = tower.encode(inputs, cu_patches, patches)


__all__ = ["VisionBudgets", "VisionGraph", "VisionPack"]
