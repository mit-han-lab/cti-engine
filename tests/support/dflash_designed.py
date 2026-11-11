"""A designed DFlash pair for GPU tests.

The target's layers add nothing to the residual stream and its LM head maps each token's
embedding to the next token id, so greedy decoding continues a prompt's ``t -> t + 1``
chain with a wide margin; the draft's forward is scripted (``scripted_draft``). The target
and draft KV of every committed position is a function of its token and position, which
``CommittedKV`` checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cti.inference import InferenceEngine

VOCAB = HIDDEN = 256
Q = 16
EOS = 200
MAX_SEQ_LEN = 512


def chain(start: int, length: int) -> list[int]:
    return [(start + offset) % VOCAB for offset in range(length)]


def expected_output(start: int, length: int, max_tokens: int, ignore_eos: bool) -> list[int]:
    output = chain(start + length, max_tokens)
    if not ignore_eos and EOS in output:
        output = output[: output.index(EOS) + 1]
    return output


def write_checkpoints(root: Path) -> tuple[str, str, dict, dict]:
    import torch
    from safetensors.torch import save_file

    geometry = {
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "max_position_embeddings": MAX_SEQ_LEN,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "model_type": "qwen3",
    }
    generator = torch.Generator().manual_seed(0)

    def random(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator) * 0.02).to(torch.bfloat16)

    def layer(prefix: str) -> dict[str, torch.Tensor]:
        return {
            f"{prefix}.input_layernorm.weight": torch.ones(HIDDEN, dtype=torch.bfloat16),
            f"{prefix}.post_attention_layernorm.weight": torch.ones(HIDDEN, dtype=torch.bfloat16),
            f"{prefix}.self_attn.q_proj.weight": random(256, HIDDEN),
            f"{prefix}.self_attn.k_proj.weight": random(128, HIDDEN),
            f"{prefix}.self_attn.v_proj.weight": random(128, HIDDEN),
            f"{prefix}.self_attn.q_norm.weight": torch.ones(128, dtype=torch.bfloat16),
            f"{prefix}.self_attn.k_norm.weight": torch.ones(128, dtype=torch.bfloat16),
            f"{prefix}.self_attn.o_proj.weight": random(HIDDEN, 256),
            f"{prefix}.mlp.gate_proj.weight": random(64, HIDDEN),
            f"{prefix}.mlp.up_proj.weight": random(64, HIDDEN),
            f"{prefix}.mlp.down_proj.weight": random(HIDDEN, 64),
        }

    # The target's branches write zeros, so its final norm sees 16 * e_t and the head,
    # a permutation, scores token t + 1 at 16 and every other token at 0.
    target = {"model.embed_tokens.weight": torch.eye(VOCAB, dtype=torch.bfloat16)}
    target["model.norm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)
    target["lm_head.weight"] = torch.eye(VOCAB, dtype=torch.bfloat16).roll(1, dims=0)
    for index in range(3):
        weights = layer(f"model.layers.{index}")
        for name in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
            weights[f"model.layers.{index}.{name}"].zero_()
        target.update(weights)
    draft = {"fc.weight": random(HIDDEN, 2 * HIDDEN)}
    draft["hidden_norm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)
    draft["norm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)
    for index in range(2):
        draft.update(layer(f"layers.{index}"))

    folders = []
    for name, weights, config in (
        (
            "target",
            target,
            {**geometry, "architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 3},
        ),
        (
            "draft",
            draft,
            {
                **geometry,
                "architectures": ["DFlashDraftModel"],
                "num_hidden_layers": 2,
                "layer_types": ["full_attention"] * 2,
                "block_size": Q,
                "num_target_layers": 3,
                "dflash_config": {"mask_token_id": VOCAB - 1, "target_layer_ids": [0, 1]},
            },
        ),
    ):
        folder = root / name
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps({**config, "eos_token_id": EOS}))
        save_file(weights, str(folder / "model.safetensors"))
        folders.append(str(folder))
    return folders[0], folders[1], target, draft


def engine_over(rank, *, overlap: bool) -> tuple[InferenceEngine, list]:
    """The single rank's ``InferenceEngine`` over ``rank``'s executor, built as
    ``EngineCore`` builds it, with checks, and the list it publishes its replies to."""

    from cti.inference import InferenceEngine

    replies: list = []
    executor = rank.executor
    engine = InferenceEngine(
        executor.limits,
        executor,
        publish=replies.extend,
        agree_on_step=None,
        text_decoder=None,
        overlap=overlap,
        checks=True,
    )
    return engine, replies


def admit(engine: InferenceEngine, replies: list, message) -> None:
    """Hand ``message`` to ``engine``, which must admit it without replying."""

    published = len(replies)
    engine.handle([message], admitting=True)
    assert len(replies) == published, replies[published:]


def scripted_draft(self, batches, embeddings):
    """Hidden rows whose target LM-head argmax proposes the chain through the scripted
    acceptance length, then wrong tokens."""

    import torch

    block = batches[0].input_ids.view(-1, Q).long()
    steps = torch.arange(Q, device=block.device)
    proposals = block[:, :1] + steps
    wrong = steps > (block[:, :1] * 7) % Q
    proposals = torch.where(wrong, proposals + 17, proposals) % VOCAB
    # The head scores token t + 1 from e_t; scatter, unlike one_hot, is capture-safe.
    rows = torch.zeros(proposals.numel(), HIDDEN, dtype=embeddings.dtype, device=block.device)
    return rows.scatter_(1, ((proposals - 1) % VOCAB).view(-1, 1), 16.0)


def _rope(x, positions):
    import torch

    half = x.shape[-1] // 2
    inv_freq = 1.0 / 1_000_000 ** (torch.arange(half, device=x.device) / half)
    angles = positions[:, None].float() * inv_freq
    first, second = x[..., :half], x[..., half:]
    return torch.cat(
        (
            first * angles.cos() - second * angles.sin(),
            second * angles.cos() + first * angles.sin(),
        ),
        -1,
    )


def _rms(x):
    return x * (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()


def _expected_kv(weights: dict, prefix: str, rows, positions):
    """K and V a layer stores for normed input ``rows`` at ``positions``, in FP32."""

    weight = {
        name: weights[f"{prefix}.self_attn.{name}_proj.weight"].cuda().float() for name in "kv"
    }
    k = _rope(_rms(rows @ weight["k"].T), positions)
    return k, rows @ weight["v"].T


class CommittedKV:
    """Checks that the target and draft KV of committed positions match their tokens and
    positions, keeping each relative error in ``errors``."""

    def __init__(self, model_runner, target_weights: dict, draft_weights: dict) -> None:
        self._runner = model_runner
        self._target_weights = target_weights
        self._draft_weights = draft_weights
        self.errors: list[float] = []

    def __call__(self, slot: int, tokens: list[int]) -> None:
        import torch

        runner = self._runner
        positions = torch.arange(len(tokens), device="cuda")
        cache_rows = runner.page_table.tokens[slot, : len(tokens)].long()
        embeddings = torch.eye(VOCAB, device="cuda")[torch.tensor(tokens, device="cuda")]
        # Target branches add nothing, so every layer reads the normed embedding, and
        # every tap is the embedding itself.
        taps = torch.cat((embeddings, embeddings), -1)
        fc = self._draft_weights["fc.weight"].cuda().float()
        # The rounds' backends attend over the target's group and the draft's.
        target_pages = runner._participants[0].kv
        draft_pages = runner._speculative._draft_backend.kv
        for pages, weights, prefix, count, rows in (
            (target_pages, self._target_weights, "model.layers", 3, _rms(embeddings)),
            (draft_pages, self._draft_weights, "layers", 2, _rms(taps @ fc.T)),
        ):
            for layer in range(count):
                k, v = _expected_kv(weights, f"{prefix}.{layer}", rows, positions)
                for cache, expected in zip(pages[layer], (k, v), strict=True):
                    actual = cache.flatten(0, 1)[cache_rows].flatten(1).float()
                    error = (actual - expected).abs().max() / expected.abs().max()
                    self.errors.append(error.item())
