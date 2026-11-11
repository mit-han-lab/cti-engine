"""LoRA serving through the public ``LLM``: the greedy token ids and logprobs of fixed
prompts, their comparison between two engines, an adapter's expert factor shapes and the
receipt a test child prints; the merge of an adapter into the weights it adapts, which
layer tests compare an adapted layer against; and a projection's delta, one ``addmm`` per
slice, which its in-place serving delta on the CPU must equal."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from cti.layers import Adapter
    from cti.layers.lora import LoRAProjection
    from cti.llm import LLM


PROMPTS = ("The capital of France is", "def fibonacci(n):\n    ")
MAX_TOKENS = 24
# The dense LoRA test's graph bound: the router routes a token alike at every row count.
GRAPH_LOGPROB_TOLERANCE = 0.1
# Before two engines' greedy paths part, their logprobs are compared over at least this
# fraction of the steps.
MIN_COMPARED_FRACTION = 0.5
RECEIPT_PREFIX = "CTI_MOE_LORA="


def greedy(llm: LLM) -> list[dict[str, list]]:
    """Each prompt's greedy ``token_ids`` and their ``logprobs`` at ``llm``'s revision."""

    from cti.protocol import SamplingParams

    params = SamplingParams(
        temperature=0, ignore_eos=True, max_new_tokens=MAX_TOKENS, logprobs=True
    )
    runs = [
        {"token_ids": list(sample.sampled_token_ids), "logprobs": list(sample.sampling_logprobs)}
        for sample in llm.sample(list(PROMPTS), params).samples
    ]
    assert all(len(run["token_ids"]) == MAX_TOKENS for run in runs)
    return runs


def compare_runs(reference: list[dict], candidate: list[dict]) -> list[dict[str, Any]]:
    """Per prompt: where the greedy paths part and the largest logprob gap before it."""

    result = []
    for expected, actual in zip(reference, candidate, strict=True):
        pairs = list(zip(expected["token_ids"], actual["token_ids"], strict=True))
        parted = next((index for index, (a, b) in enumerate(pairs) if a != b), None)
        compared = len(pairs) if parted is None else parted
        deltas = [
            abs(a - b)
            for a, b in zip(
                expected["logprobs"][:compared], actual["logprobs"][:compared], strict=True
            )
        ]
        result.append({"parted_at": parted, "max_logprob_delta": max(deltas, default=0.0)})
    return result


def factor_shapes(
    prefix: str, lead: tuple[int, ...], hidden: int, width: int, rank: int
) -> list[list]:
    """The sorted ``[name, shape]`` pairs of the expert factors under ``prefix`` (names from
    the MLP on), ``lead`` their stacking dimensions and ``width`` their intermediate size."""

    shapes = {
        "down_proj.full": (hidden, width),
        "gate_up_proj.gate": (width, hidden),
        "gate_up_proj.up": (width, hidden),
    }
    return [
        entry
        for projection, (output_size, input_size) in shapes.items()
        for entry in (
            [f"{prefix}.{projection}.lora_A", [*lead, rank, input_size]],
            [f"{prefix}.{projection}.lora_B", [*lead, output_size, rank]],
        )
    ]


def print_receipt(result: dict[str, Any]) -> None:
    """Print a child's receipt: ``result`` and the device it ran on."""

    import torch

    receipt = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "result": result,
    }
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True), flush=True)


def read_receipt(output: str) -> dict[str, Any]:
    """The result of the one receipt a child printed in ``output``."""

    from tests.support.gpu import assert_cuda_device_metadata, child_receipt

    receipt = child_receipt(output, RECEIPT_PREFIX)
    assert_cuda_device_metadata(receipt)
    return receipt["result"]


def merge_adapter(adapters: Adapter, base_state: dict[str, torch.Tensor]) -> None:
    """Merge the adapter into the weights it adapts, then zero every B: graphs read the same
    storage on replay. The adapter set exposes no per-projection view, so this reads its
    projections directly. Expert stacks merge a few experts at a time."""

    import torch

    from cti.layers.lora import ExpertLoRAFactors

    with torch.no_grad():
        for key, projection in adapters._projections.items():
            # The adapted op's weights: a sparse block's experts, or a linear's weight.
            prefix = f"{key}." if key else ""
            if isinstance(projection, ExpertLoRAFactors):
                gate_a, gate_b, up_a, up_b = projection.gate_up
                gate_up = base_state[f"{prefix}experts.gate_up_proj"]
                width = gate_b.shape[1]
                stacks = (
                    (gate_up[:, :width], gate_a, gate_b),
                    (gate_up[:, width:], up_a, up_b),
                    (base_state[f"{prefix}experts.down_proj"], *projection.down),
                )
            else:
                weight = base_state[f"{prefix}weight"]
                factors = projection.factors
                stacks = tuple(
                    (
                        weight.narrow(0, output_slice.offset, output_slice.size)[None],
                        factors[2 * index][None],
                        factors[2 * index + 1][None],
                    )
                    for index, output_slice in enumerate(projection.output_slices)
                )
            for rows, a, b in stacks:
                for part in zip(rows.split(16), a.split(16), b.split(16), strict=True):
                    weights, part_a, part_b = part
                    update = (part_b.float() @ part_a.float()) * projection.scaling
                    weights.copy_((weights.float() + update).to(weights.dtype))
        for name, tensor in adapters.named_parameters():
            if name.endswith(".lora_B"):
                tensor.zero_()


__all__ = [
    "compare_runs",
    "factor_shapes",
    "greedy",
    "merge_adapter",
    "print_receipt",
    "read_receipt",
]


def lora_delta(projection: LoRAProjection, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``y`` plus each slice's ``scaling * x A^T B^T``, one out-of-place ``addmm`` per slice."""

    x_2d, y_2d = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
    factors = projection.factors
    slices = [
        torch.addmm(
            y_2d.narrow(1, output_slice.offset, output_slice.size),
            F.linear(x_2d, a),
            b.t(),
            alpha=projection.scaling,
        )
        for output_slice, a, b in zip(
            projection.output_slices, factors[::2], factors[1::2], strict=True
        )
    ]
    return torch.cat(slices, -1).view(y.shape)
