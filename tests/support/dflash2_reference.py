"""An independent CPU reference of the DFlash2 draft (docs/dflash.md), written out from the
contract in FP64 with explicit loops: the block forward with its grouped convolutions and
non-causal sliding attention, the candidate lattice, the greedy chain and the conditional q.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


def rms(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def rotate(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    """NeoX rotary embedding of ``x`` ``[T, heads, D]`` at ``positions``."""

    half = x.shape[-1] // 2
    inv_freq = 1.0 / base ** (torch.arange(half, dtype=torch.float64) / half)
    angles = positions.double()[:, None, None] * inv_freq
    first, second = x[..., :half], x[..., half:]
    return torch.cat(
        (
            first * angles.cos() - second * angles.sin(),
            second * angles.cos() + first * angles.sin(),
        ),
        -1,
    )


def grouped_conv(
    x: torch.Tensor, deltas: torch.Tensor, base: torch.Tensor, block_size: int
) -> torch.Tensor:
    """``out[n, ch] = sum_t (base[t, ch] + deltas[n, t, ch // group]) * x[n - t, ch]`` over the
    taps t that stay inside row n's block."""

    rows, channels = x.shape
    taps, groups = deltas.shape[1], deltas.shape[2]
    group_of = torch.arange(channels) // (channels // groups)
    out = torch.zeros_like(x)
    for n in range(rows):
        for t in range(taps):
            if n % block_size >= t:
                out[n] += (base[t] + deltas[n, t, group_of]) * x[n - t]
    return out


def _conv_pair(
    weights: Mapping[str, torch.Tensor], name: str, x: torch.Tensor, taps: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """A branch's convolved input and its output side's deltas."""

    projection = weights[f"{name}.kernel_projection.weight"].double()
    deltas = (x @ projection.T).view(len(x), 2, taps, -1)
    base = weights[f"{name}.base_kernel"].double()
    return grouped_conv(x, deltas[:, 0], base[0], block_size), deltas[:, 1]


def draft_block(
    weights: Mapping[str, torch.Tensor],
    taps: torch.Tensor,
    embeddings: torch.Tensor,
    *,
    heads: int,
    kv_heads: int,
    head_dim: int,
    window: int,
    rope_base: float,
    conv_taps: int,
) -> torch.Tensor:
    """The final-normed hidden rows of one block after ``taps``' context rows. Context rows
    sit at positions 0..C-1 and the block at C..C+Q-1; a block query at p sees every block
    key and the context keys k with p - k < ``window``."""

    w = {name: tensor.double() for name, tensor in weights.items()}
    num_context, q = taps.shape[0], embeddings.shape[0]
    context = rms(taps.double() @ w["fc.weight"].T, w["hidden_norm.weight"])
    positions = torch.arange(num_context + q)
    keys_at, queries_at = positions, positions[num_context:]
    visible = (keys_at[None] >= num_context) | (queries_at[:, None] - keys_at[None] < window)
    h, layer = embeddings.double(), 0
    while f"layers.{layer}.input_layernorm.weight" in w:
        p = f"layers.{layer}."
        q_w, k_w, v_w = w[p + "self_attn.qkv_proj.weight"].split(
            [heads * head_dim, kv_heads * head_dim, kv_heads * head_dim]
        )
        x, deltas = _conv_pair(
            w, p + "attention_conv", rms(h, w[p + "input_layernorm.weight"]), conv_taps, q
        )
        query = rms((x @ q_w.T).view(q, heads, head_dim), w[p + "self_attn.q_norm.weight"])
        sources = torch.cat((context, x))
        key = rms((sources @ k_w.T).view(-1, kv_heads, head_dim), w[p + "self_attn.k_norm.weight"])
        value = (sources @ v_w.T).view(-1, kv_heads, head_dim)
        query, key = rotate(query, queries_at, rope_base), rotate(key, keys_at, rope_base)
        out = torch.empty(q, heads, head_dim, dtype=torch.float64)
        for head in range(heads):
            kv = head // (heads // kv_heads)
            logits = query[:, head] @ key[:, kv].T / head_dim**0.5
            logits = logits.masked_fill(~visible, float("-inf"))
            out[:, head] = logits.softmax(-1) @ value[:, kv]
        attended = out.reshape(q, -1) @ w[p + "self_attn.o_proj.weight"].T
        h = h + grouped_conv(attended, deltas, w[p + "attention_conv.base_kernel"][1], q)
        x, deltas = _conv_pair(
            w, p + "mlp_conv", rms(h, w[p + "post_attention_layernorm.weight"]), conv_taps, q
        )
        gate, up = (x @ w[p + "mlp.gate_up_proj.weight"].T).chunk(2, -1)
        mlp = (torch.nn.functional.silu(gate) * up) @ w[p + "mlp.down_proj.weight"].T
        h = h + grouped_conv(mlp, deltas, w[p + "mlp_conv.base_kernel"][1], q)
        layer += 1
    return rms(h, w["norm.weight"])


def top_candidates(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's k largest logits (the smaller ID among equal ones) by ascending ID."""

    ids, values = [], []
    for row in logits.tolist():
        best = sorted(sorted(range(len(row)), key=lambda i: (-row[i], i))[:k])
        ids.append(best)
        values.append([row[i] for i in best])
    return torch.tensor(ids), torch.tensor(values, dtype=torch.float64)


def lattice(
    selector: Mapping[str, torch.Tensor],
    hidden: torch.Tensor,
    ids: torch.Tensor,
    unary: torch.Tensor,
    anchor: int,
) -> torch.Tensor:
    """``scores[e, p, c] = unary[e, c] + sum_r A[pred]_r (P h_e)_r B[ids[e, c]]_r`` for one
    row's G positions; position 0's predecessor is the anchor for every p."""

    a = selector["predecessor_codebook"].double()
    b = selector["successor_codebook"].double()
    projected = hidden.double() @ selector["hidden_projection.weight"].double().T
    positions, k = ids.shape
    scores = torch.empty(positions, k, k, dtype=torch.float64)
    for e in range(positions):
        for p in range(k):
            predecessor = anchor if e == 0 else int(ids[e - 1, p])
            for c in range(k):
                state = a[predecessor] * projected[e] * b[int(ids[e, c])]
                scores[e, p, c] = unary[e, c] + state.sum()
    return scores


def greedy_chain(scores: torch.Tensor) -> list[int]:
    """Candidate indices of the argmax walk from the anchor (smaller index on ties)."""

    chain = [int(scores[0, 0].argmax())]
    for e in range(1, scores.shape[0]):
        chain.append(int(scores[e, chain[-1]].argmax()))
    return chain


def chain_q(scores: torch.Tensor, chain: list[int], temperature: float) -> torch.Tensor:
    """Each position's temperature softmax over its candidates after the chosen predecessor."""

    rows = [scores[0, 0]] + [scores[e, chain[e - 1]] for e in range(1, scores.shape[0])]
    return (torch.stack(rows) / temperature).softmax(-1)
