"""The Gated DeltaNet GPU tests' layer and child processes: a layer at Qwen3.5-9B's
linear-attention heads over the recurrent-state pool, on the ``ctikernels.gdn`` operations
its ``LayerServices`` resolve or on the torch reference operations of
``tests/support/gated_delta_net.py``. Each test runs one child process per component, which
prints a receipt; the parent asserts on it. Torch loads only in the child.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tests.support.gpu import (
    assert_cuda_device_metadata,
    isolated_runtime_env,
    run_isolated_process,
)
from tests.support.paths import PYTHON_ROOT, REPO_ROOT

if TYPE_CHECKING:
    import torch

    from cti.kvcache import RecurrentStatePool
    from cti.layers.gated_delta_net import GatedDeltaNet
    from cti.recurrent import RecurrentBackend
    from tests.support.forward_rows import Entry

RECEIPT_PREFIX = "CTI_GATED_DELTA_NET="
NUM_KEY_HEADS, NUM_VALUE_HEADS = 16, 32  # Qwen3.5-9B's linear-attention heads
HEAD_DIM = 128
CONV_CHANNELS = (2 * NUM_KEY_HEADS + NUM_VALUE_HEADS) * HEAD_DIM
PROJECTED_WIDTH = CONV_CHANNELS + NUM_VALUE_HEADS * HEAD_DIM + 2 * NUM_VALUE_HEADS
NUM_SLOTS = 8


# ============================== Child process ==============================


def gdn_layer(device: torch.device, *, kernels: bool = True) -> GatedDeltaNet:
    """A layer with seeded checkpoint weights on ``device``, on ``ctikernels.gdn`` or the
    torch reference operations; its services are ``layer.test_services``."""

    import torch

    from cti.layers import LayerServices
    from cti.layers.gated_delta_net import GatedDeltaNet
    from tests.support.tp_shards import ReferenceLayerServices, default_dtype, site

    services = LayerServices() if kernels else ReferenceLayerServices()
    with default_dtype(torch.bfloat16):
        layer = GatedDeltaNet(
            num_key_heads=NUM_KEY_HEADS,
            num_value_heads=NUM_VALUE_HEADS,
            key_head_dim=HEAD_DIM,
            value_head_dim=HEAD_DIM,
            conv_kernel_size=4,
            site=site(device=device, services=services),
        )
    layer.test_services = services
    generator = torch.Generator().manual_seed(0)
    weights = {
        "conv_weight": (0.5 * torch.randn(CONV_CHANNELS, 1, 4, generator=generator)).to(
            torch.bfloat16
        ),
        "A_log": torch.log(torch.rand(NUM_VALUE_HEADS, generator=generator) * 15 + 1),
        "dt_bias": torch.randn(NUM_VALUE_HEADS, generator=generator),
        "norm_weight": 1 + 0.1 * torch.randn(HEAD_DIM, generator=generator),
    }
    layer.load_state_dict({name: weight.to(device) for name, weight in weights.items()})
    return layer


def bind_layer(
    layer: GatedDeltaNet, device: torch.device
) -> tuple[RecurrentBackend, RecurrentStatePool]:
    """Bind ``layer`` to row 0 of a pool of deterministic garbage rows."""

    import torch

    from cti.kvcache import RecurrentStatePool
    from cti.recurrent import RecurrentBackend

    pool = RecurrentStatePool((layer.state_spec,), num_slots=NUM_SLOTS, device=device)
    generator = torch.Generator(device=device).manual_seed(123)
    state, conv_state = pool.layer_views(0)
    state.normal_(generator=generator)
    conv_state.copy_(torch.randn(conv_state.shape, generator=generator, device=device))
    layer.bind_state({"recurrent": pool.layer_state(0)})
    return RecurrentBackend(pool, device=device), pool


def run_layer(
    layer: GatedDeltaNet,
    backend: RecurrentBackend,
    projected: torch.Tensor,
    *entries: Entry,
    padding: int = 0,
) -> torch.Tensor:
    import torch

    from cti.forward import ForwardInputs
    from tests.support.forward_rows import forward_rows

    rows = forward_rows(*entries, padding=padding, padding_slot=NUM_SLOTS)
    metadata = backend.prepare_metadata(rows)
    batch = ForwardInputs(
        None, None, None, rows.size, rows.uses_ragged_attention, recurrent_metadata=metadata
    )
    with torch.inference_mode():
        return layer.forward(projected, batch)


def projected_rows(num_tokens: int, seed: int, device: torch.device) -> torch.Tensor:
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)
    rows = torch.randn(num_tokens, PROJECTED_WIDTH, generator=generator, device=device)
    return rows.to(torch.bfloat16)


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.float(), expected.float()
    scale = expected.abs().max().clamp_min(1e-30)
    return ((actual - expected).abs().max() / scale).item()


def pools_equal(first: RecurrentStatePool, second: RecurrentStatePool) -> bool:
    import torch

    return all(
        torch.equal(mine, theirs)
        for mine, theirs in zip(first.layer_views(0), second.layer_views(0), strict=True)
    )


def run_component(
    component: str, components: Mapping[str, Callable[[torch.device], dict[str, Any]]]
) -> None:
    """Run ``components[component]`` on the current CUDA device and print its receipt."""

    import torch

    import cti
    import ctikernels.gdn

    device = torch.device("cuda", torch.cuda.current_device())
    receipt: dict[str, Any] = {
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "component": component,
        "engine_root": str(Path(cti.__file__).resolve().parent),
        "kernels": str(Path(ctikernels.gdn.__file__).resolve().parent),
    }
    receipt.update(components[component](device))
    print(RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True))


# ============================== Parent ==============================


def component_receipt(tmp_path: Path, module: str, component: str) -> dict[str, Any]:
    """Run ``module``'s ``component`` in a child process and return its receipt."""

    env, log_root = isolated_runtime_env(tmp_path, f"gated-delta-net-{component}")
    output = run_isolated_process(
        (sys.executable, "-m", module, "--component", component),
        log_root / f"{component}.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=600,
    )
    receipts = [
        json.loads(line.removeprefix(RECEIPT_PREFIX))
        for line in output.splitlines()
        if line.startswith(RECEIPT_PREFIX)
    ]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert_cuda_device_metadata(receipt)
    assert receipt["component"] == component
    assert Path(receipt["engine_root"]).is_relative_to(PYTHON_ROOT)
    return receipt
