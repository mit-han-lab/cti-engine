"""DFlash rounds across tensor-parallel ranks, as spawned processes over gloo on CPU.

A tiny Qwen3 target (four query and two KV heads) and a two-layer draft (four query heads
and one KV head, which both ranks keep): a v1 draft (a sliding layer, then a full one) or
a DFlash2 draft (two non-causal sliding layers with convolutions and a candidate
selector). They prefill two prompts, then run greedy, sampled and filtered rounds in FP32
with the CPU references at every kernel seam, under the engine's checks. Both final norms
read one channel, so the target and the draft choose between the same two tokens and
rounds accept long prefixes. TP2 ranks record the same rounds and token tables as each
other and as the single rank; a v1 peer that holds other seeds still commits the primary's
sampled verdicts, while DFlash2 ranks draw the same chains from the same seeds.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import safetensors.torch
import torch
import torch.nn.functional as F
from transformers import AutoConfig, Qwen3Config

from cti import kvcache
from cti.attention.fixed_query import FixedQueryMetadata, RoundAddresses
from cti.attention.paged import store_kv_pair
from cti.config import Checkpoint as CheckpointOption
from cti.distributed.runtime import ParallelGroup
from cti.forward import ForwardInputs, RoundShape
from cti.inference import Round
from cti.inference._execution._graphs import forward_rows
from cti.inference._execution._sampling import LogitProcessing, SampledLogprobs
from cti.inference._execution._speculative import dflash as dflash_module
from cti.inference._execution._speculative.dflash import (
    RoundBackends,
    RoundInput,
    SpeculativeRuntime,
    TargetForward,
)
from cti.inference._step import QueryKind, RequestSpec, Row
from cti.kvcache import GroupRole, PageGroup
from cti.layers import LayerServices
from cti.models import CheckpointSource, get_model_class, open_weights
from cti.models import dflash as drafts
from cti.models.dflash import DFlashConfig
from cti.sampling import LogprobMode, TokenSampling
from tests.support import reference_ops
from tests.support.gloo_ranks import gloo_ranks
from tests.support.paged_attention import (
    ReferencePagedAttentionBackend,
    bind_model_state,
    kv_group,
)
from tests.support.qwen3_5_engine import bind_reference_kernels
from tests.support.tp_shards import (
    CPU,
    default_dtype,
    load_sharded,
    qwen3_checkpoint_shapes,
    site,
)

# Gloo ranks are processes of their own: one such test at a time keeps them from starving.
pytestmark = pytest.mark.xdist_group("gloo")


@dataclass
class _ReferenceRows:
    """What the reference reads of a fixed-query batch when it attends."""

    request_indices: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    causal: bool
    window: int | None

    def prepare(self) -> None:
        """The reference reads the borrowed metadata when it attends."""


class _ReferenceRoundAttentionBackend(ReferencePagedAttentionBackend):
    """The reference backend, which also serves DFlash rounds: row ``b`` of a fixed-query
    batch attends with its Q queries to the cache rows ``kv_indices[kv_indptr[b]:
    kv_indptr[b + 1]]``, its queries' keys last; a causal query sees the keys up to its own,
    and a ``window`` W only the last W of those; a non-causal one with a window W sees its
    block's keys and the keys less than W before its own."""

    # DFlash's startup log names the providers of its round backends.
    providers = "reference"

    addresses: RoundAddresses

    def create_fixed_query(
        self,
        request_indices: torch.Tensor,
        positions: torch.Tensor,
        *,
        causal: bool,
        window: int | None = None,
    ) -> FixedQueryMetadata:
        size, addresses = request_indices.numel(), self.addresses
        kv_indptr = addresses.indptr[: size + 1]
        rows = _ReferenceRows(request_indices, kv_indptr, addresses.indices, causal, window)
        write = addresses.write[: size * addresses.width]
        return FixedQueryMetadata(write, rows, kv_indptr, torch.empty(0))

    def prepare_round(
        self, request_indices: torch.Tensor, frontiers: torch.Tensor, done: torch.Tensor
    ) -> None:
        self.addresses.prepare(request_indices, frontiers, done)

    def round_write_rows(self, tokens: int) -> torch.Tensor:
        return self.addresses.write[:tokens]

    def map_warm_up(self, rows: int) -> None:
        self.addresses.map_warm_up(rows)

    def write_kv(
        self, key: torch.Tensor, value: torch.Tensor, layer_id: int, batch: ForwardInputs
    ) -> None:
        rows = batch.attn_metadata.cache_write_indices[: key.shape[0]]
        store_kv_pair(self._kv[layer_id], key, value, rows)

    def _attend(self, q: torch.Tensor, layer_id: int, batch: ForwardInputs) -> torch.Tensor:
        if not isinstance(batch.attn_metadata, FixedQueryMetadata):
            return super()._attend(q, layer_id, batch)
        metadata = batch.attn_metadata.operation
        k_cache, v_cache = self._token_rows(layer_id)
        size = metadata.request_indices.numel()
        q = q.view(size, -1, *q.shape[1:])
        bounds = metadata.kv_indptr.tolist()
        outputs = []
        for row in range(size):
            keys = metadata.kv_indices[bounds[row] : bounds[row + 1]].long()
            length, queries = keys.numel(), q.shape[1]
            # Query i's own key is key length - Q + i.
            own = length - queries + torch.arange(queries)[:, None]
            key_positions = torch.arange(length)[None]
            if metadata.causal:
                visible = key_positions <= own
            else:
                visible = (key_positions >= 0).expand(queries, length)
            if metadata.window is not None:
                near = key_positions > own - metadata.window
                # A non-causal sliding query also sees every key of its block.
                block = key_positions >= length - queries
                visible = visible & (near if metadata.causal else near | block)
            output = F.scaled_dot_product_attention(
                q[row].transpose(0, 1),
                k_cache[keys].transpose(0, 1),
                v_cache[keys].transpose(0, 1),
                attn_mask=visible.expand(queries, length),
                enable_gqa=True,
            )
            outputs.append(output.transpose(0, 1))
        return torch.cat(outputs)


_SIZE = 2
_MAX_SEQ_LEN = 96
_EOS = 3
_PROMPTS = ((5, 9, 13, 17, 21), (8, 6, 4, 2, 10, 12, 14))
# Rounds of up to 64 tokens in all outrun this budget, so rows finish within them.
_BUDGET = 40
_ROUND_TOKENS = 64
_SEED = 1234
# Each scenario's law. Logprobs, and a steered law's min-p, penalties, logit bias and
# min_tokens (masking EOS), make every round a full one.
_SCENARIOS = (
    ("greedy", TokenSampling(0.0)),
    ("sampled", TokenSampling(1.0)),
    ("filtered", TokenSampling(0.8, 8, 0.9)),
    ("logprobs", TokenSampling(1.0, logprobs=LogprobMode.PROCESSED, top_logprobs=2)),
    (
        "steered",
        TokenSampling(
            1.0,
            min_p=0.05,
            presence_penalty=0.5,
            frequency_penalty=0.2,
            repetition_penalty=1.1,
            logit_bias=((5, 2.0),),
            min_tokens=3,
            logprobs=LogprobMode.RAW,
            top_logprobs=2,
        ),
    ),
)
_TARGET = Qwen3Config(
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    hidden_size=64,
    vocab_size=67,
    intermediate_size=128,
    rms_norm_eps=1e-6,
    max_position_embeddings=128,
    rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
    hidden_act="silu",
    tie_word_embeddings=False,
    architectures=["Qwen3ForCausalLM"],
)
_DRAFT = {
    "architectures": ["DFlashDraftModel"],
    "model_type": "qwen3",
    "vocab_size": 67,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 64,
    "hidden_act": "silu",
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10_000.0,
    "tie_word_embeddings": False,
    "layer_types": ["sliding_attention", "full_attention"],
    "sliding_window": 24,
    "block_size": 16,
    "num_target_layers": 4,
    "dflash_config": {"mask_token_id": 66, "target_layer_ids": [1, 2]},
}
_DRAFTS = {
    "dflash": _DRAFT,
    "dflash2": {
        **{key: value for key, value in _DRAFT.items() if key != "block_size"},
        "architectures": ["DFlash2DraftModel"],
        "is_causal": False,
        "layer_types": ["sliding_attention"] * 2,
        "dflash_config": {
            "block_size": 8,
            "mask_token_id": 66,
            "target_layer_ids": [1, 2],
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "selector_rank": 8,
            "selector_top_k": 16,
        },
    },
}


def _draft_shapes(kind: str) -> dict[str, tuple[int, ...]]:
    hidden, head_dim = _DRAFT["hidden_size"], _DRAFT["head_dim"]
    query, kv, mlp = 4 * head_dim, 1 * head_dim, _DRAFT["intermediate_size"]
    shapes = {
        "fc.weight": (hidden, 2 * hidden),
        "hidden_norm.weight": (hidden,),
        "norm.weight": (hidden,),
    }
    if kind == "dflash2":
        rank, vocab = 8, _DRAFT["vocab_size"]
        shapes |= {
            "candidate_selector.hidden_projection.weight": (rank, hidden),
            "candidate_selector.predecessor_codebook": (vocab, rank),
            "candidate_selector.successor_codebook": (vocab, rank),
        }
    for layer in range(_DRAFT["num_hidden_layers"]):
        prefix = f"layers.{layer}"
        shapes |= {
            f"{prefix}.input_layernorm.weight": (hidden,),
            f"{prefix}.post_attention_layernorm.weight": (hidden,),
            f"{prefix}.self_attn.q_proj.weight": (query, hidden),
            f"{prefix}.self_attn.k_proj.weight": (kv, hidden),
            f"{prefix}.self_attn.v_proj.weight": (kv, hidden),
            f"{prefix}.self_attn.o_proj.weight": (hidden, query),
            f"{prefix}.self_attn.q_norm.weight": (head_dim,),
            f"{prefix}.self_attn.k_norm.weight": (head_dim,),
            f"{prefix}.mlp.gate_proj.weight": (mlp, hidden),
            f"{prefix}.mlp.up_proj.weight": (mlp, hidden),
            f"{prefix}.mlp.down_proj.weight": (hidden, mlp),
        }
        if kind == "dflash2":
            for conv in ("attention_conv", "mlp_conv"):
                shapes |= {
                    f"{prefix}.{conv}.base_kernel": (2, 2, hidden),
                    # Two sides, two taps, four groups of 16 channels.
                    f"{prefix}.{conv}.kernel_projection.weight": (16, hidden),
                }
    return shapes


def _random_tensors(
    shapes: dict[str, tuple[int, ...]], final_norm: str, final_scale: float, seed: int
) -> dict:
    """Weights near unit scale; the final norm keeps hidden channel 0 alone, scaled."""

    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, shape in shapes.items():
        noise = torch.randn(shape, generator=generator)
        if name.endswith("norm.weight"):
            tensors[name] = 1 + 0.1 * noise
        elif name.endswith("base_kernel"):
            # Near the identity: tap 0 passes each row on, tap 1 mixes in a little.
            tensors[name] = 0.01 * noise
            tensors[name][:, 0] += 1
        elif "codebook" in name or "kernel_projection" in name:
            # A selector and convolutions that keep the draft's choice of token.
            tensors[name] = 0.1 * noise * shape[-1] ** -0.5
        elif name == "model.embed_tokens.weight":
            tensors[name] = noise
        else:
            tensors[name] = noise * shape[-1] ** -0.5
    tensors[final_norm] = torch.zeros(shapes[final_norm])
    tensors[final_norm][0] = final_scale
    return tensors


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("dflash-tp")
    (folder / "target").mkdir()
    # Opposite signs: with these seeds the draft's channel 0 keeps the sign the target's
    # has, so both pick the same of their two tokens; the target's sharp p rejects a few.
    target = _random_tensors(qwen3_checkpoint_shapes(_TARGET), "model.norm.weight", 96.0, 0)
    safetensors.torch.save_file(target, folder / "target" / "model.safetensors")
    _TARGET.save_pretrained(folder / "target")
    for kind, config in _DRAFTS.items():
        (folder / kind).mkdir()
        draft = _random_tensors(_draft_shapes(kind), "norm.weight", -32.0, 1)
        safetensors.torch.save_file(draft, folder / kind / "model.safetensors")
        (folder / kind / "config.json").write_text(json.dumps(config))
    return folder


def _bind_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    bind_reference_kernels(monkeypatch)

    def rotate_key(key, cos_sin_cache, positions, *, head_dim):
        return reference_ops.apply_rotary_(
            key.clone(), key, cos_sin_cache, positions, head_dim=head_dim
        )[1]

    rotary = ModuleType("ctikernels.rotary.inference")
    rotary.apply_rotary_ = reference_ops.apply_rotary_  # type: ignore[attr-defined]
    rotary.apply_rotary_key_ = rotate_key  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, rotary.__name__, rotary)


def _pool(layers, q: int, role: GroupRole, index: int) -> PageGroup:
    layer = next(iter(layers))
    num_pages = len(_PROMPTS) * _MAX_SEQ_LEN + 2 * q
    return kv_group(
        len(tuple(layers)),
        layer.num_kv_heads,
        layer.head_dim,
        num_pages,
        device=CPU,
        role=role,
        index=index,
    )


def _scenario(
    folder: Path,
    kind: str,
    group: ParallelGroup | None,
    law: TokenSampling,
    seed: int,
) -> dict:
    """Prefill both prompts, then run the rounds; return each round's record, the token
    table, frontiers and the draft pool's KV heads."""

    target_config = AutoConfig.from_pretrained(folder / "target")
    target_site = site(group=group)
    target = load_sharded(
        target_config, folder / "target", group=group, services=target_site.services
    )[0]
    model_config = get_model_class(target_config.architectures[0]).engine_config(target_config)
    draft_path = str(folder / kind)
    document = json.loads((folder / kind / "config.json").read_text(encoding="utf-8"))
    dflash = DFlashConfig.from_hf(document, AutoConfig.from_pretrained(draft_path), model_config)
    q = dflash.block_size
    plan = SimpleNamespace(
        max_batch_tokens=_MAX_SEQ_LEN,
        dtype=torch.float32,
        weights=CheckpointOption(),
        rounds=RoundShape(q, 2, min(4, len(_PROMPTS), _MAX_SEQ_LEN // q)),
        attention=SimpleNamespace(page_size=1),
        max_seq_len=_MAX_SEQ_LEN,
    )
    checkpoint = SimpleNamespace(
        model_config=model_config,
        facts=SimpleNamespace(stop_token_ids=(_EOS,), source=CheckpointSource(str(folder))),
    )
    # The draft at the target's site with services of its own, as the model host builds it.
    draft_site = replace(target_site, training=False, services=LayerServices())
    with torch.device("meta"), default_dtype(plan.dtype):
        draft = getattr(drafts, document["architectures"][0])(
            dflash, round_width=dflash.block_size, site=draft_site
        )
    phases: list[str] = []

    def loading(phase: str) -> None:
        phases.append(f"draft {phase}")

    draft.load_state_dict(
        open_weights(CheckpointSource(draft_path), plan.weights, loading).state_dict(
            draft, CPU, show_progress=False, progress=loading
        )
    )
    models = SimpleNamespace(
        checkpoint=checkpoint,
        device=CPU,
        model=target,
        services=target_site.services,
        draft=draft,
    )
    round_models, scratch = dflash_module.prepare_rounds(dflash, plan, models)  # type: ignore[arg-type]

    rows = len(_PROMPTS) * _MAX_SEQ_LEN
    cache_index_table = torch.zeros((len(_PROMPTS) + 1, _MAX_SEQ_LEN), dtype=torch.int32)
    cache_index_table[: len(_PROMPTS)] = torch.arange(rows, dtype=torch.int32).view(
        -1, _MAX_SEQ_LEN
    )
    token_id_table = torch.zeros((len(_PROMPTS) + 1, _MAX_SEQ_LEN + q), dtype=torch.int32)
    target_pool = _pool(tuple(target.iter_attention_layers()), q, GroupRole.TARGET, 0)
    draft_pool = _pool(tuple(draft.iter_attention_layers()), q, GroupRole.DRAFT, 1)
    target_backend = _ReferenceRoundAttentionBackend(target_pool, cache_index_table)
    # The prompts' rows, then the rounds' guard rows; no graph sink page of its own.
    capacity = kvcache.StateCapacity(
        page_size=1,
        num_slots=len(_PROMPTS),
        max_seq_len=_MAX_SEQ_LEN,
        table_width=_MAX_SEQ_LEN,
        num_pages=rows,
        snapshot_rows=range(0),
        sink_slot=rows,
        guard_slots=range(rows, rows + 2 * q),
        snapshot_grid=1,
        export_granularity=1,
    )
    # Rounds attend through the reference backends here, not cti-kernels' DraftAttention,
    # and run_round's records stay on the device.
    draft_backend = _ReferenceRoundAttentionBackend(draft_pool, cache_index_table)
    # Each model's layers on its KV group's backend, as the runner binds them.
    bind_model_state(target, target_backend)
    bind_model_state(draft, draft_backend)

    target_backend.addresses = draft_backend.addresses = RoundAddresses(
        cache_index_table,
        round_models.fixed_query_geometries()[0].shape,
        max_seq_len=_MAX_SEQ_LEN,
        guard_start=rows,
    )
    dflash_module.RoundRecords = lambda *_: None
    state = kvcache.InferenceState(capacity, (target_pool, draft_pool), None)
    runner = SpeculativeRuntime(
        round_models,
        scratch,
        state,
        recurrent_rounds=None,
        backends=RoundBackends(target_backend, draft_backend),
        token_id_table=token_id_table,
        processing=LogitProcessing(token_id_table, _DRAFT["vocab_size"]),
        logprobs=SampledLogprobs(token_id_table.device, _DRAFT["vocab_size"]),
        padding_rows=range(len(_PROMPTS), len(_PROMPTS) + 1),
        forward_hidden=target.forward_hidden,
        decides=group is None or group.is_primary(),
        publish=(lambda words: None) if group is None else group.broadcast_device,
        checks=True,
        differs=None if group is None else group.differs_device,
    )

    with torch.inference_mode():
        # The prompts' forward stores their draft context; its argmax is each first token.
        entries = tuple(
            Row(row, QueryKind.EXTEND, 0, len(prompt)) for row, prompt in enumerate(_PROMPTS)
        )
        rows = forward_rows(entries)
        requests = torch.cat([torch.full((len(p),), r) for r, p in enumerate(_PROMPTS)])
        positions = torch.cat([torch.arange(len(prompt)) for prompt in _PROMPTS])
        for row, prompt in enumerate(_PROMPTS):
            token_id_table[row, : len(prompt)] = torch.tensor(prompt)
        batch = ForwardInputs(
            token_id_table[requests, positions],
            positions,
            target_backend.prepare_metadata(rows),
            rows.size,
            rows.uses_ragged_attention,
            logit_rows=torch.tensor(rows.logit_rows),
        )
        first = target.forward(batch).argmax(-1)
        runner.after_forward(TargetForward(batch, rows, positions, requests))
        for row, prompt in enumerate(_PROMPTS):
            token_id_table[row, len(prompt)] = first[row]

        block = Round(tuple(range(len(_PROMPTS))))
        specs = [
            RequestSpec(row, len(prompt), len(prompt) + _BUDGET, law, seed=seed + row)
            for row, prompt in enumerate(_PROMPTS)
        ]
        records, faults, scores, results = [], [], [], scratch.results
        size = len(_PROMPTS)
        for _ in range(_ROUND_TOKENS // q):
            staged = RoundInput.stage(
                block,
                specs,
                capacity=runner.max_batch_size,
                vocab_size=runner.vocab_size,
                padding=range(len(_PROMPTS), len(_PROMPTS) + 1),
                pin_memory=False,
            )
            run = runner.run_round(staged)
            records.append(
                {
                    name: getattr(results, name)[:size].tolist()
                    for name in ("frontiers", "tokens", "accepted", "committed", "done")
                }
            )
            # The faults the engine's checks found in the round.
            faults += [message for message, fault in run.faults if bool(fault)]
            scores.append(
                {
                    name: getattr(scratch.logprobs, name)[:size].tolist()
                    for name in ("logprobs", "top_ids", "top_logprobs")
                }
            )
    layer = next(draft.iter_attention_layers())
    return {
        "phases": phases,
        "records": records,
        "faults": faults,
        # A round's logprobs, which only full rounds fill.
        "scores": scores if law.logprobs is not None else None,
        "token_table": token_id_table[:size].tolist(),
        "draft_heads": (layer.num_qo_heads, layer.num_kv_heads),
    }


def _run(folder: Path, kind: str, group: ParallelGroup | None, seed: int) -> dict[str, dict]:
    return {name: _scenario(folder, kind, group, law, seed) for name, law in _SCENARIOS}


def _rank(group: ParallelGroup, argument: tuple[Path, str]) -> dict[str, dict]:
    # The rank process exits after the scenario, so its bindings are never undone. A v1
    # peer holds other seeds: it must commit the primary's verdicts all the same. DFlash2
    # ranks draw their chains from the request's seed, which every rank shares.
    folder, kind = argument
    _bind_kernels(pytest.MonkeyPatch())
    peer_seed = _SEED + 1000 if kind == "dflash" else _SEED
    return _run(folder, kind, group, _SEED if group.is_primary() else peer_seed)


def _diverging_rank(group: ParallelGroup, folder: Path) -> list[str]:
    _bind_kernels(pytest.MonkeyPatch())
    seed = _SEED if group.is_primary() else _SEED + 1000
    return _scenario(folder, "dflash2", group, _SCENARIOS[1][1], seed)["faults"]


@pytest.fixture(scope="module", params=list(_DRAFTS))
def kind(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def single_rank(checkpoints: Path, kind: str) -> dict[str, dict]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        _bind_kernels(monkeypatch)
        return _run(checkpoints, kind, None, _SEED)


@pytest.fixture(scope="module")
def ranks(
    checkpoints: Path, kind: str, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[list[dict]]:
    rendezvous = tmp_path_factory.mktemp("rendezvous")
    with gloo_ranks(_rank, _SIZE, (checkpoints, kind), rendezvous) as results:
        yield results


@pytest.mark.parametrize("scenario", [name for name, _ in _SCENARIOS])
def test_every_rank_commits_the_single_rank_rounds(
    ranks: list[dict], single_rank: dict, scenario: str
) -> None:
    expected = single_rank[scenario]
    assert not expected["faults"]
    for rank, result in enumerate(ranks):
        assert not result[scenario]["faults"], f"rank {rank}"
        assert result[scenario]["records"] == ranks[0][scenario]["records"], f"rank {rank}"
        assert result[scenario]["token_table"] == ranks[0][scenario]["token_table"], f"rank {rank}"
    assert ranks[0][scenario]["records"] == expected["records"]
    assert ranks[0][scenario]["token_table"] == expected["token_table"]


def test_the_draft_load_reports_its_shard(single_rank: dict) -> None:
    for name, _ in _SCENARIOS:
        assert single_rank[name]["phases"] == [
            "draft loading weights: shard 1/1 (model.safetensors)"
        ], name


def test_rounds_accept_drafts_and_finish_rows(single_rank: dict) -> None:
    # Greedy rounds accept every draft until the budget finishes both rows; sampled rounds
    # accept some and reject others.
    greedy = single_rank["greedy"]["records"]
    q = len(greedy[0]["tokens"][0])
    assert greedy[0]["accepted"] == [q - 1] * len(_PROMPTS)
    assert greedy[-1]["done"] == [1] * len(_PROMPTS)
    # Steered rounds may reject every draft: drafts are not penalized.
    for scenario in ("sampled", "filtered", "logprobs"):
        accepted = [
            count for record in single_rank[scenario]["records"] for count in record["accepted"]
        ]
        assert 0 < max(accepted) and min(accepted) < q - 1, scenario


def test_other_seeds_would_draw_other_verdicts(
    checkpoints: Path, kind: str, single_rank: dict
) -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        _bind_kernels(monkeypatch)
        name, law = _SCENARIOS[1]
        peer = _scenario(checkpoints, kind, None, law, _SEED + 1000)
    assert peer["records"] != single_rank[name]["records"]


@pytest.mark.parametrize("scenario", ["logprobs", "steered"])
def test_every_rank_reports_the_deciding_ranks_logprobs(
    ranks: list[dict], single_rank: dict, scenario: str
) -> None:
    # Only the primary builds the p a processed logprob reads; the peers report its scores.
    for rank, result in enumerate(ranks):
        assert result[scenario]["scores"] == ranks[0][scenario]["scores"], f"rank {rank}"
    pairs = zip(ranks[0][scenario]["scores"], single_rank[scenario]["scores"], strict=True)
    for got, want in pairs:
        assert got["top_ids"] == want["top_ids"]
        # Sharded projections round differently from one rank's.
        torch.testing.assert_close(
            torch.tensor(got["logprobs"]), torch.tensor(want["logprobs"]), rtol=1e-5, atol=1e-4
        )


def test_each_rank_holds_its_share_of_the_draft_heads(ranks: list[dict], single_rank: dict) -> None:
    assert single_rank["greedy"]["draft_heads"] == (4, 1)
    # Two query heads per rank; both keep a copy of the one KV head.
    assert [result["greedy"]["draft_heads"] for result in ranks] == [(2, 1)] * _SIZE


def test_dflash2_ranks_drawing_other_chains_fail_the_checks(
    checkpoints: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # A peer with other seeds draws other sampled chains, which the checks catch.
    rendezvous = tmp_path_factory.mktemp("diverging")
    with gloo_ranks(_diverging_rank, _SIZE, checkpoints, rendezvous) as results:
        for faults in results:
            assert "replicas proposed different drafts" in faults
