"""Tensor-parallel training on spawned gloo ranks against the same computation on one rank:
the vocabulary-sharded head, LoRA gradients through a tiny Qwen3's entries and exits, and
per-expert LoRA on expert-parallel ranks."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import safetensors.torch
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, Qwen3Config

from cti.config import (
    AdamWConfig,
    BatchLayout,
    LoRAConfig,
    StaticBatching,
    TrainablePlan,
    TrainingBatching,
    TrainingConfig,
)
from cti.distributed.runtime import ParallelGroup
from cti.layers import Adapter, ExpertParallel, ExpertPlacement, ExpertTensorParallel
from cti.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from cti.models import get_model_class
from cti.training import SFTDatum, TrainingBatch, TrainingCheckpoint
from tests.support.gloo_ranks import gloo_ranks
from tests.support.moe import ReferenceMoe
from tests.support.qwen3_5 import TINY_QWEN3_5_CONFIG, load_hf_config, randomize_hf_weights
from tests.support.qwen3_5_engine import bind_reference_kernels, load_model
from tests.support.qwen3_5_moe import TINY_QWEN3_5_MOE_CONFIG, text_only_config
from tests.support.qwen3_moe import save_tiny_checkpoint
from tests.support.tp_shards import load_sharded, qwen3_checkpoint_shapes, site
from tests.support.training import AdaptedTrainables, forward_backward, host_runner, optim_step
from tests.support.training_ops import (
    bind_training_references,
)

# Gloo ranks are processes of their own: one such test at a time keeps them from starving.
pytestmark = pytest.mark.xdist_group("gloo")

# 259 IDs leave a short last shard at TP2 and TP4.
_VOCAB, _HIDDEN, _ROWS = 259, 16, 300
# The sharded head and the reductions run in FP32, like the whole-vocabulary kernels.
_FP32 = {"atol": 1e-5, "rtol": 1e-4}
# The training tests' AdamW. Its eps serves this equivalence only, to make the bound below
# provable; no default changes. At the default eps an element whose gradient is near zero can
# flip its update's sign between reduction orders, a difference of lr.
_LR, _EPS, _MAX_GRAD_NORM, _STEPS = 1e-2, 1e-2, 0.5, 3
# After _STEPS steps: the gradients obey _FP32 (test_lora_gradients_match_one_rank), and
# clipping bounds each element by _MAX_GRAD_NORM, so |dg| <= 1e-5 + 1e-4 * 0.5 = 6e-5. With
# eps >> dg the update m/(sqrt(v) + eps) moves at most |dg| / eps a step, so the parameters
# differ by at most 2 * steps * lr * dg / eps (2 covers the moments' history and the steps'
# own parameter differences): 3.6e-4. The moments differ by about dg.
_AFTER_ADAMW = {"atol": 2 * _STEPS * _LR * 6e-5 / _EPS, "rtol": 0.0}


def _arrays(tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Rank results cross the Pipe as arrays: a tensor would travel as shared memory."""

    return {name: tensor.detach().numpy().copy() for name, tensor in tensors.items()}


def _tensors(arrays: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {name: torch.from_numpy(array) for name, array in arrays.items()}


def _head_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "weight": torch.randn(_VOCAB, _HIDDEN, generator=generator, dtype=torch.float64),
        "hidden": torch.randn(_ROWS, _HIDDEN, generator=generator, dtype=torch.float64),
        "targets": torch.randint(0, _VOCAB, (_ROWS,), generator=generator),
        "ids": torch.randint(0, _VOCAB, (7,), generator=generator),
        "upstream": torch.randn(_ROWS, generator=generator, dtype=torch.float64),
    }


def _host() -> str:
    """The CPU model, its vector flags and the ATen path and threads a rank computes with."""

    info = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    model = next(
        (
            line.split(":", 1)[1].strip()
            for line in info.splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    flags = next(
        (line.split(":", 1)[1].split() for line in info.splitlines() if line.startswith("flags")),
        [],
    )
    vector = sorted(flag for flag in flags if flag.startswith(("avx", "amx", "fma")))
    capability = torch.backends.cpu.get_cpu_capability()
    return f"{model}; {' '.join(vector)}; ATen {capability}; {torch.get_num_threads()} threads"


def _sharded_head(group: ParallelGroup, temperature: float) -> dict[str, Any]:
    inputs = _head_inputs()
    # The first gather is the log-probabilities' (max, sum of exps, target) per rank and row.
    gathered: list[torch.Tensor] = []
    all_gather = group.all_gather

    def recorded(tensor: torch.Tensor) -> torch.Tensor:
        result = all_gather(tensor)
        gathered.append(result.detach().clone())
        return result

    group.all_gather = recorded
    head = ParallelLMHead(_VOCAB, _HIDDEN, site=site(group=group, training=True))
    start, count = head.vocab_range
    head.weight = inputs["weight"].new_zeros(head.num_embeddings_tp, _HIDDEN)
    head.weight[:count] = inputs["weight"][start : start + count]
    hidden = inputs["hidden"].clone().requires_grad_(True)
    logprobs = head.forward_training_logprobs(hidden, inputs["targets"], temperature=temperature)
    (logprobs * inputs["upstream"]).sum().backward()
    loss_hidden = inputs["hidden"].clone().requires_grad_(True)
    loss = head.forward_training_loss(loss_hidden, inputs["targets"])
    loss.backward()
    logits_hidden = inputs["hidden"].clone().requires_grad_(True)
    logits = head.forward_training(logits_hidden)
    (logits * inputs["weight"][:, 0]).sum().backward()
    embedding = VocabParallelEmbedding(_VOCAB, _HIDDEN, site=site(group=group))
    embedding.weight = head.weight
    arrays = _arrays(
        {
            "triples": gathered[0].view(group.size, _ROWS, 3),
            "logprobs": logprobs,
            "hidden_grad": hidden.grad,
            "loss": loss,
            "loss_hidden_grad": loss_hidden.grad,
            "logits": logits,
            "logits_hidden_grad": logits_hidden.grad,
            "lookup": embedding.forward_training(inputs["ids"]),
        }
    )
    return {"arrays": arrays, "host": _host()}


def _triple_report(
    inputs: dict[str, torch.Tensor],
    ranks: list[dict[str, torch.Tensor]],
    row: int,
    size: int,
    temperature: float,
) -> str:
    """Row ``row``'s gathered (max, sum of exps, target) per shard against FP64, in FP32 ulps
    of each value, as every rank received them."""

    unit = 2.0**-24
    shard = -(-_VOCAB // size)
    logits = inputs["hidden"][row] @ inputs["weight"].T / temperature
    target = int(inputs["targets"][row])
    report = []
    for owner in range(size):
        cut = logits[owner * shard : min((owner + 1) * shard, _VOCAB)]
        maximum = cut.max()
        exact = (
            maximum,
            (cut - maximum).exp().sum(),
            logits[target]
            if owner * shard <= target < (owner + 1) * shard
            else torch.zeros((), dtype=torch.float64),
        )
        seen = [tuple(float(rank["triples"][owner, row, k]) for k in range(3)) for rank in ranks]
        drift = [
            round(abs(seen[0][k] - float(value)) / (unit * max(abs(float(value)), 1e-30)), 1)
            for k, value in enumerate(exact)
        ]
        report.append(f"shard {owner}: drift/u {drift}, ranks agree {len(set(seen)) == 1}")
    return "; triples " + " | ".join(report)


def _sharded_head_bounds(
    inputs: dict[str, torch.Tensor], size: int, temperature: float
) -> dict[str, torch.Tensor]:
    """Elementwise bounds on how far ``size`` ranks' sharded head (ShardedLogprobs) may lie
    from the FP64 reference, along its FP32 computation, with u FP32's unit roundoff.

    Each logit z_j comes in FP64 and is rounded to FP32 and divided by the temperature: off
    by 2u|z_j|. The log-normalizer L = M + log(sum of exps) is off by its largest logit's
    error 2u|M|, its own rounding u|L|, the shard's sum of exps and the ranks' combine
    (shard + size)u, and the roundings of the max, the combine's exps and the log, 8u. A
    softmax term exp(z_j - L), whose argument rounds once more (u|z_j - L|) and whose exp
    rounds once (u), is then off by p_j * eps_j, eps_j the sum of these. The logit gradient
    onehot_j - p_j adds the target's one in FP32 and takes g / T, itself rounded twice: a
    further 4u|onehot_j - p_j|. The input gradient sums these times w_j in FP64, so element k
    is off by at most sum_j (p_j eps_j + 4u|onehot_j - p_j|) |g| / T |w_jk|. A
    log-probability z_t - L is off by 2u|z_t|, L's error and its rounding; the loss sums the
    rows' FP32 values, adding (rows - 1)u times their magnitudes.
    """

    unit = 2.0**-24
    shard = -(-_VOCAB // size)
    weight, hidden, targets = inputs["weight"], inputs["hidden"], inputs["targets"]

    def bounds(temperature: float, upstream: torch.Tensor) -> tuple[torch.Tensor, ...]:
        logits = hidden @ weight.T / temperature
        normalizer = torch.logsumexp(logits, -1, keepdim=True)
        normalizer_error = unit * (
            2 * logits.amax(-1, keepdim=True).abs() + normalizer.abs() + shard + size + 8
        )
        softmax = (logits - normalizer).exp()
        onehot = F.one_hot(targets, _VOCAB).double()
        term = (
            softmax
            * (unit * (2 * logits.abs() + (logits - normalizer).abs() + 1) + normalizer_error)
            + 4 * unit * (onehot - softmax).abs()
        )
        gradient = (term * (upstream.abs() / temperature).unsqueeze(-1)) @ weight.abs()
        logprobs = (logits - normalizer).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        target = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        logprob = 2 * unit * target.abs() + normalizer_error.squeeze(-1) + unit * logprobs.abs()
        return logprob, gradient, logprobs

    logprobs, hidden_grad, _ = bounds(temperature, inputs["upstream"])
    loss_logprobs, loss_hidden_grad, values = bounds(1.0, torch.ones(_ROWS, dtype=torch.float64))
    loss = loss_logprobs.sum() + (_ROWS - 1) * unit * values.abs().sum()
    return {
        "logprobs": logprobs,
        "hidden_grad": hidden_grad,
        "loss": loss,
        "loss_hidden_grad": loss_hidden_grad,
    }


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_sharded_head_matches_one_rank(size: int, temperature: float, tmp_path: Path) -> None:
    inputs = _head_inputs()
    weight = inputs["weight"]
    hidden = inputs["hidden"].clone().requires_grad_(True)
    log_softmax = F.log_softmax(hidden @ weight.T / temperature, dim=-1)
    expected = log_softmax.gather(-1, inputs["targets"].unsqueeze(-1)).squeeze(-1)
    (expected * inputs["upstream"]).sum().backward()
    loss_hidden = inputs["hidden"].clone().requires_grad_(True)
    expected_loss = F.cross_entropy(loss_hidden @ weight.T, inputs["targets"], reduction="sum")
    expected_loss.backward()
    logits_hidden = inputs["hidden"].clone().requires_grad_(True)
    expected_logits = logits_hidden @ weight.T
    (expected_logits * weight[:, 0]).sum().backward()

    bounds = _sharded_head_bounds(inputs, size, temperature)

    with gloo_ranks(_sharded_head, size, temperature, tmp_path) as results:
        ranks = [_tensors(result["arrays"]) for result in results]
        hosts = sorted({result["host"] for result in results})
        for result in ranks:
            for name, value in (
                ("logprobs", expected),
                ("hidden_grad", hidden.grad),
                ("loss", expected_loss),
                ("loss_hidden_grad", loss_hidden.grad),
            ):
                error = (result[name].double() - value.detach()).abs()
                ratio = (error / bounds[name]).flatten()
                worst = int(ratio.argmax())
                # Enough to tell a loose bound from a perturbed row: how many elements
                # fail, the worst one, and every rank's value of it.
                assert (error <= bounds[name]).all(), (
                    f"{name}: {int((ratio > 1).sum())} of {ratio.numel()} elements over "
                    f"their bound; worst {worst}: error {float(error.flatten()[worst]):.3e}, "
                    f"bound {float(bounds[name].flatten()[worst]):.3e}, "
                    f"expected {float(value.flatten()[worst]):.9g}, ranks "
                    f"{[float(rank[name].flatten()[worst]) for rank in ranks]}"
                    + (
                        _triple_report(inputs, ranks, worst, size, temperature)
                        if name == "logprobs"
                        else ""
                    )
                    + f"; hosts {hosts}"
                )
            torch.testing.assert_close(result["logits"], expected_logits.detach())
            torch.testing.assert_close(result["logits_hidden_grad"], logits_hidden.grad)
            torch.testing.assert_close(result["lookup"], weight[inputs["ids"]])


# Four query and two KV heads: TP4 holds each KV head on two ranks.
_QWEN3 = Qwen3Config(
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    hidden_size=64,
    vocab_size=67,
    intermediate_size=64,
    rms_norm_eps=1e-6,
    max_position_embeddings=64,
    rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
    tie_word_embeddings=False,
    architectures=["Qwen3ForCausalLM"],
)
_QWEN3_5 = copy.deepcopy(dict(TINY_QWEN3_5_CONFIG))
# Both mixers; each of two ranks owns one Gated DeltaNet key head and one KV head.
_QWEN3_5["text_config"].update(
    vocab_size=67,
    num_hidden_layers=2,
    layer_types=["linear_attention", "full_attention"],
    num_attention_heads=4,
    num_key_value_heads=2,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
)


def _write_qwen3_5(folder: Path) -> None:
    reference = transformers.Qwen3_5ForConditionalGeneration(load_hf_config(_QWEN3_5, folder))
    randomize_hf_weights(reference, conv_dtype=torch.bfloat16)
    reference.save_pretrained(folder)


def _write_qwen3(folder: Path) -> None:
    generator = torch.Generator().manual_seed(0)
    tensors = {}
    for name, shape in qwen3_checkpoint_shapes(_QWEN3).items():
        noise = torch.randn(shape, generator=generator)
        if name.endswith("norm.weight"):
            tensors[name] = 1 + 0.1 * noise
        elif name == "model.embed_tokens.weight":
            tensors[name] = noise
        else:
            tensors[name] = noise * shape[-1] ** -0.5
    safetensors.torch.save_file(tensors, folder / "model.safetensors")
    _QWEN3.save_pretrained(folder)


def _write_qwen3_moe(folder: Path) -> None:
    save_tiny_checkpoint(folder)


def _write_qwen3_5_moe(folder: Path) -> None:
    """Text-only Qwen3.5-MoE whose heads split over two ranks; its shared expert's LoRA
    gradients are rank partials."""

    config = copy.deepcopy(dict(TINY_QWEN3_5_MOE_CONFIG))
    config["text_config"].update(
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )
    reference = transformers.AutoModelForCausalLM.from_config(
        load_hf_config(text_only_config(config), folder), experts_implementation="eager"
    )
    randomize_hf_weights(reference, conv_dtype=torch.bfloat16)
    reference.save_pretrained(folder)


def _training_model(
    folder: Path,
    adapters: Adapter,
    group: ParallelGroup | None,
    experts: ExpertPlacement | None,
):
    """The model at ``folder`` for training on ``group``'s rank, a MoE model's experts placed
    by ``experts``."""

    if experts is None:
        return load_model(folder, lora_adapters=adapters, group=group, training=True)
    return load_sharded(
        AutoConfig.from_pretrained(folder),
        folder,
        group=group,
        lora_adapters=adapters,
        training=True,
        moe_backend=ReferenceMoe(),
        experts=experts,
    )[0]


def _window(tensor: torch.Tensor, window: Any) -> torch.Tensor:
    if window is None:
        return tensor
    return tensor.narrow(window.axis, window.start, window.stop - window.start)


def _lora_step(
    group: ParallelGroup | None, argument: tuple[Path, dict[str, int] | None]
) -> dict[str, Any]:
    """One rank's summed loss and its LoRA factors and gradients, with each factor's window
    and partiality. Every B starts from one nonzero logical value of ``rows`` rows (a single
    rank's own), so each gradient is nonzero."""

    with pytest.MonkeyPatch.context() as monkeypatch:
        bind_reference_kernels(monkeypatch)
        bind_training_references(monkeypatch)
        return _bound_lora_step(group, *argument)


def _bound_lora_step(
    group: ParallelGroup | None, folder: Path, rows: dict[str, int] | None
) -> dict[str, Any]:
    targets = get_model_class(AutoConfig.from_pretrained(folder).architectures[0])
    adapters = Adapter(LoRAConfig(targets=targets.LORA_TARGETS, rank=4, alpha=8.0, seed=5))
    model = load_model(folder, lora_adapters=adapters, group=group, training=True)
    adapters.materialize(device=torch.device("cpu"), dtype=torch.float32)
    places = {
        name: place
        for (name, _), place in zip(adapters.named_parameters(), adapters.placements(), strict=True)
    }
    with torch.no_grad():
        for name, factor in adapters.named_parameters():
            if name.endswith(".lora_B"):
                window = places[name].window
                logical_rows = factor.shape[0] if rows is None else rows[name]
                seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
                generator = torch.Generator().manual_seed(seed)
                logical = torch.randn(logical_rows, factor.shape[1], generator=generator)
                factor.copy_(_window(0.1 * logical, window))
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, model.lm_head.num_embeddings, (2, 9), generator=generator)
    hidden = model.training_hidden_states(input_ids[:, :-1], None)
    loss = model.training_loss(hidden, input_ids[:, 1:].flatten())
    loss.backward()
    tensors = {"loss": loss}
    for name, factor in adapters.named_parameters():
        tensors[f"value:{name}"] = factor
        tensors[f"grad:{name}"] = factor.grad
    meta = {name: (place.window, place.reduce) for name, place in places.items()}
    return {"arrays": _arrays(tensors), "meta": meta}


@pytest.mark.parametrize(
    ("write", "size"), [(_write_qwen3, 2), (_write_qwen3, 4), (_write_qwen3_5, 2)]
)
def test_lora_gradients_match_one_rank(
    write: Callable[[Path], None], size: int, tmp_path: Path
) -> None:
    """Each rank's A starts as the single rank's cut to its window; complete gradients
    tile the logical one and partial gradients sum to it. Qwen3 at TP4 holds each KV head
    on two ranks; Qwen3.5 adds Gated DeltaNet and gated attention."""

    folder = tmp_path / "checkpoint"
    folder.mkdir()
    write(folder)
    expected = _tensors(_lora_step(None, (folder, None))["arrays"])
    rows = {
        name.removeprefix("value:"): tensor.shape[0]
        for name, tensor in expected.items()
        if name.endswith(".lora_B")
    }
    rendezvous = tmp_path / "rendezvous"
    rendezvous.mkdir()
    with gloo_ranks(_lora_step, size, (folder, rows), rendezvous) as results:
        ranks = [(_tensors(result["arrays"]), result["meta"]) for result in results]

    for arrays, _ in ranks:
        torch.testing.assert_close(arrays["loss"], expected["loss"], **_FP32)
    for name in ranks[0][1]:
        logical = torch.zeros_like(expected[f"grad:{name}"])
        for arrays, meta in ranks:
            window, partial = meta[name]
            if name.endswith(".lora_A"):
                torch.testing.assert_close(
                    arrays[f"value:{name}"], _window(expected[f"value:{name}"], window)
                )
            part = _window(logical, window)
            if partial:
                part += arrays[f"grad:{name}"]
            else:
                part.copy_(arrays[f"grad:{name}"])
        torch.testing.assert_close(
            logical,
            expected[f"grad:{name}"],
            **_FP32,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def _training_steps(group: ParallelGroup | None, argument: tuple[Any, ...]) -> dict[str, Any]:
    """SFT steps of the adapter through the runner: each step's loss and gradient norm, then
    the checkpoint's logical factors and moments. See ``_bound_training_steps``."""

    with pytest.MonkeyPatch.context() as monkeypatch:
        bind_reference_kernels(monkeypatch)
        bind_training_references(monkeypatch)
        return _bound_training_steps(group, *argument)


def _bound_training_steps(
    group: ParallelGroup | None,
    folder: Path,
    restore: Path | None,
    save: tuple[int, Path] | None,
    experts: ExpertPlacement | None,
) -> dict[str, Any]:
    """Run the steps after ``restore``'s, if any, to ``_STEPS``; ``save`` = (steps, path)
    publishes the checkpoint after that many steps from rank 0."""

    model_class = get_model_class(AutoConfig.from_pretrained(folder).architectures[0])
    adapters = Adapter(LoRAConfig(targets=model_class.LORA_TARGETS, rank=8, alpha=16.0, seed=5))
    model = _training_model(folder, adapters, group, experts)
    adapters.materialize(device=torch.device("cpu"), dtype=torch.float32)
    runner = host_runner(
        model,
        TrainingConfig(
            TrainablePlan.all_declared(),
            AdamWConfig(
                lr=_LR,
                betas=(0.9, 0.99),
                eps=_EPS,
                weight_decay=0.01,
                max_grad_norm=_MAX_GRAD_NORM,
            ),
            TrainingBatching(
                BatchLayout.PADDED, StaticBatching(microbatch_size=2, max_tokens_per_sample=16)
            ),
        ),
        trainables=AdaptedTrainables(model, adapters),
        ranks=group,
        placed=adapters,
    )
    result: dict[str, Any] = {}
    start = 0
    if restore is not None:
        checkpoint = TrainingCheckpoint.load(restore)
        runner.restore(checkpoint)
        start = checkpoint.num_optimizer_steps
        result["restored"] = _arrays(
            _state(runner.checkpoint(num_optimizer_steps=start, model_revision=0))
        )
    generator = torch.Generator().manual_seed(7)
    vocab = model.lm_head.num_embeddings
    trace = []
    for step in range(_STEPS):
        data = [
            SFTDatum(
                token_ids=tuple(torch.randint(1, vocab, (length,), generator=generator).tolist()),
                target_mask=(False, *([True] * (length - 1))),
            )
            for length in (9, 6, 7)
        ]
        if step < start:
            continue
        metrics = forward_backward(
            runner, TrainingBatch.sft(data, BatchLayout.PADDED, pad_token_id=0)
        )
        trace.append([float(metrics.loss_sum), optim_step(runner, metrics.normalization_count)])
        if save is not None and step + 1 == save[0]:
            runner.save(save[1], num_optimizer_steps=step + 1, model_revision=0)
    result.update(
        trace=trace,
        arrays=_arrays(_state(runner.checkpoint(num_optimizer_steps=_STEPS, model_revision=0))),
    )
    return result


def _state(checkpoint: TrainingCheckpoint) -> dict[str, torch.Tensor]:
    """The checkpoint's logical parameters and AdamW state, by parameter name."""

    optimizer = checkpoint.optimizer
    tensors = {}
    for index, name in enumerate(optimizer.names):
        tensors[f"value:{name}"] = optimizer.master[index]
        tensors[f"published:{name}"] = optimizer.live[index]
        for key, value in optimizer.adamw["state"][index].items():
            tensors[f"{key}:{name}"] = value
    return tensors


def _ranks(size: int, argument: tuple[Any, ...], rendezvous: Path) -> list[dict[str, Any]]:
    if size == 1:
        return [_training_steps(None, argument)]
    rendezvous.mkdir()
    with gloo_ranks(_training_steps, size, argument, rendezvous) as results:
        return list(results)


def _assert_states_close(actual: dict[str, Any], expected: dict[str, Any], **tolerance) -> None:
    actual, expected = _tensors(actual), _tensors(expected)
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(
            actual[name], tensor, **tolerance, msg=lambda message, name=name: f"{name}: {message}"
        )


@pytest.mark.parametrize(
    ("write", "size", "experts"),
    [
        (_write_qwen3, 2, None),
        (_write_qwen3, 4, None),
        (_write_qwen3_5, 2, None),
        (_write_qwen3_moe, 2, ExpertParallel()),
        (_write_qwen3_moe, 2, ExpertTensorParallel()),
        (_write_qwen3_5_moe, 2, ExpertParallel()),
    ],
    ids=["qwen3-2", "qwen3-4", "qwen3_5-2", "qwen3_moe-ep2", "qwen3_moe-etp2", "qwen3_5_moe-ep2"],
)
def test_training_steps_match_one_rank(
    write: Callable[[Path], None], size: int, experts: ExpertPlacement | None, tmp_path: Path
) -> None:
    """Per-step loss and clipped gradient norm, and the checkpoint's logical adapter and
    AdamW moments after the last step, match one rank's: every expert's factors, whether
    each rank holds half of the experts or half of every expert's width."""

    folder = tmp_path / "checkpoint"
    folder.mkdir()
    write(folder)
    single = None if experts is None else ExpertParallel()
    expected = _training_steps(None, (folder, None, None, single))
    for result in _ranks(size, (folder, None, None, experts), tmp_path / "rendezvous"):
        torch.testing.assert_close(
            torch.tensor(result["trace"]), torch.tensor(expected["trace"]), **_FP32
        )
        _assert_states_close(result["arrays"], expected["arrays"], **_AFTER_ADAMW)


@pytest.mark.parametrize(
    ("write", "saved", "loaded", "experts"),
    [
        (_write_qwen3_5, 2, 1, None),
        (_write_qwen3_5, 1, 2, None),
        (_write_qwen3_moe, 2, 1, ExpertParallel()),
    ],
    ids=["qwen3_5-2to1", "qwen3_5-1to2", "qwen3_moe-ep2to1"],
)
def test_checkpoint_resumes_across_rank_counts(
    write: Callable[[Path], None],
    saved: int,
    loaded: int,
    experts: ExpertPlacement | None,
    tmp_path: Path,
) -> None:
    """A checkpoint saved after two steps on ``saved`` ranks restores on ``loaded`` ranks to
    the same logical tensors bit for bit, and the next step matches the saver's own."""

    folder = tmp_path / "model"
    folder.mkdir()
    write(folder)
    path = tmp_path / "training.ckpt"
    source = _ranks(saved, (folder, None, (_STEPS - 1, path), experts), tmp_path / "saved")[0]
    written = _arrays(_state(TrainingCheckpoint.load(path)))
    for result in _ranks(loaded, (folder, path, None, experts), tmp_path / "loaded"):
        _assert_states_close(result["restored"], written, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.tensor(result["trace"]), torch.tensor(source["trace"][-1:]), **_FP32
        )
        _assert_states_close(result["arrays"], source["arrays"], **_AFTER_ADAMW)
