# Single-node tensor-parallel inference

cti runs inference for dense `Qwen3ForCausalLM`, dense Qwen3.5 text
(`Qwen3_5ForConditionalGeneration`, `Qwen3_5ForCausalLM`), `Qwen3MoeForCausalLM` and
Qwen3.5-MoE (`Qwen3_5MoeForConditionalGeneration`, `Qwen3_5MoeForCausalLM`) across the GPUs of one node,
one engine process per GPU; see [Prerequisites](#prerequisites). Select the number of
GPUs with the `parallel` options:

```python
from cti.config import EngineOptions, ModelOptions, ParallelOptions
from cti.llm import LLM

llm = LLM(EngineOptions(model=ModelOptions(path="/path/to/Qwen3"), parallel=ParallelOptions(num_ranks=2)))
try:
    # llm.sample(...) and llm.generate(...) keep their ordinary contracts.
    pass
finally:
    llm.shutdown()
```

`OnlineServer` takes the same options. The CLI flag is `--parallel.num-ranks`, and YAML
uses `parallel.num_ranks`. Ranks use local GPUs
`0..N-1` of the visible devices. The model geometry must divide evenly: query heads,
the MLP intermediate size, and either the KV heads by the size or the size by the KV
heads (KV heads are then replicated). Qwen3.5 also needs the size to divide its Gated
DeltaNet key heads, and therefore its value heads; model construction rejects any
other size.

## Prerequisites

- PyNCCL, the default all-reduce provider, comes with the pinned cti-kernels release.

## Scope

- Bounded sampling, selected-token log-probabilities and online serving, with decode
  CUDA graphs and breakable graphs.
- FlashInfer attention only. `auto` selects it on every device of compute capability
  8.0 or newer; any other explicit backend fails during configuration resolution.
- A model supports tensor parallelism exactly when it declares
  `ModelCapability.TENSOR_PARALLEL`. Dense Qwen3 and Qwen3.5, Qwen3-MoE and Qwen3.5-MoE declare it; their sparse
  blocks place whole experts per rank (expert parallel, the default) or split every
  expert by intermediate width (expert tensor parallel, `parallel.enable_expert_parallel:
  false`); see [MoE](#moe).
- Prefix reuse (`cache.prefix: radix`) as on one GPU. Each rank keeps its own radix cache, a
  Qwen3.5 rank its snapshot rows over its own Gated DeltaNet heads, and every rank makes
  the same decisions: LRU follows a logical clock, and each rank holds the fewest KV pages
  and snapshot rows any rank planned.
- DFlash runs at every size; see `docs/dflash.md`.
- LoRA and training run at every size, a routed expert's LoRA factors with its expert.
  A training checkpoint holds the logical tensors, so it resumes at another size.
  `tests/integration/distributed/test_training_tp_gloo.py` matches one rank's training:

  | Test | Models and sizes |
  |---|---|
  | `test_lora_gradients_match_one_rank` | Qwen3 at 2 and 4, Qwen3.5 at 2 |
  | `test_training_steps_match_one_rank` | Qwen3 at 2 and 4, Qwen3.5 at 2, Qwen3-MoE at 2 with expert and expert tensor parallelism, Qwen3.5-MoE at 2 with expert parallelism |
  | `test_checkpoint_resumes_across_rank_counts` | Qwen3.5 2 to 1 and 1 to 2, Qwen3-MoE 2 to 1 with expert parallelism |

  Two directly cabled hosts run as one node ([Two hosts](#two-hosts)); general multi-node
  clusters, data, pipeline and context parallelism are out of scope.
- With `RandomWeights`, every rank draws the same random values for its local
  shards. A random model is therefore a different function at each size; compare
  sizes only with real checkpoints.

## MoE

Attention, Gated DeltaNet, embedding and LM head split as in the dense model. Each sparse
block keeps its router on every rank, so every rank routes every token, and its routed and
shared partial sums meet in one all-reduce at the block's exit. `parallel.enable_expert_parallel`
places the routed experts of `n` ranks:

- `true` (default, expert parallel): rank `r` holds experts `[r E/n, (r + 1) E/n)` whole.
  They take cti-kernels' expert map, so a route to another rank's expert takes no slot and
  adds nothing.
- `false` (expert tensor parallel): every rank holds every expert, cut to `I/n` gate and up
  rows and the down columns that read them. No route is remote; ranks hold equal work at
  any batch, which is faster at small batches on Qwen3-30B-A3B.

A shared expert (Qwen3.5-MoE has one, Qwen3-MoE none) is split by its intermediate width
either way.

## Process model

The parent spawns one engine process per GPU, joined through a private file rendezvous,
and talks to rank 0 alone. Rank 0 reads the parent's commands and the online requests
and relays each record, in order, to every peer over its own pipe, so every rank's
scheduler decides alike; no rank's per-token loop waits on the parent. The parent
watches every rank's process: the first rank failure, reported or not, stops every rank
and reaches the caller as an `EngineProcessError`.

Each rank owns its process groups and its shard of the model. Its NCCL communicator
connects at startup, before memory sizing. Layers take the TP group from their
constructor and declare each parameter's checkpoint window; the loader copies each
window once.

A Qwen3.5 Gated DeltaNet layer keeps whole key-head groups per rank (see `GatedDeltaNet`),
so its recurrent state is rank-local.

Every rank schedules locally and prepares its own batch metadata, such as positions,
rotary positions and cache rows, from the same scheduler state; no batch crosses
ranks. Development runs with `CTI_CHECKS=1` compare a digest of each batch across
ranks before running it, and every rank fails on a mismatch. Only rank 0 samples; it broadcasts the selected tokens and log-probabilities
before any rank writes them, so every scheduler sees the same tokens. Every rank loads
the sampling provider at startup, so rank 0 never builds it while its peers wait.
Online, only rank 0 reads requests. Each iteration it broadcasts a fixed header (epoch,
blocking mode, status, payload size) and the request payload, and every rank checks
the epoch and mode against its own scheduler. A stop or an ingress failure reaches
every rank in the same iteration.

Shutdown keeps the memory-saver contract of a single-GPU runtime: stop ingress, settle
execution and synchronize every rank, acknowledge, then terminate and join every rank.

## Two hosts

An engine's ranks may span hosts, an equal share on each (`parallel.num_nodes`): every host
runs the same command but for `parallel.node_rank`. Node 0 serves; every other node joins
it at `parallel.dist_init_addr` (`host:port`, node 0's address on the link) and runs its own
ranks, each on its host's GPUs from `cuda:0`. For two GB10 hosts, one rank on each:

```bash
export CTI_DIST_AUTHKEY=<the same secret on both hosts>
export NCCL_SOCKET_IFNAME=<link interface> GLOO_SOCKET_IFNAME=<link interface>
export NCCL_IB_HCA=<the link's RoCE devices> NCCL_IB_GID_INDEX=3
# host A
python -m cti --model.path <model> --parallel.num-ranks 2 --parallel.num-nodes 2 \
    --parallel.node-rank 0 --parallel.dist-init-addr <A's link address>:29500
# host B: the same, with --parallel.node-rank 1
```

- The nodes start in either order, within the startup stall window. Node 0 admits a node
  only if it holds `CTI_DIST_AUTHKEY` and resolves the same model, draft and
  parallel layout (but for `node_rank`), engine and cti-kernels versions and checkpoint; a
  local checkpoint folder is compared by its `config.json` and weight files, so each host
  keeps its own copy. Node 0 sends it the rest of its options: a training engine's node 1
  joins with the plain command line, which spells no training.
- Only rank 0 talks to the parent, the frontend and the tokenizers, all on node 0; a
  remote rank's relay and failure report are authenticated TCP connections, and the
  process group meets at a `tcp://` address on node 0.
- The interface and RoCE devices come from the environment: without
  `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` the rendezvous may take another interface.
- Either side's end ends the other: a remote rank that fails, or a node whose link
  closes, fails node 0's engine, which reclaims its ranks; a node whose link to node 0
  closes reclaims its own. TCP keepalive notices a lost link or host within about 11 s.

## Numerics

Tensor parallelism changes floating-point reduction order, so it is not bit-exact
with one GPU, and cti promises no batch invariance. A greedy continuation
follows the near-tie rule: it either equals the single-GPU continuation, or first
diverges where the two runs' own picks have log-probabilities within 0.25 nats of each
other and neither pick has a probability above 0.75. Over the common prefix, the mean
log-probability difference is at most 0.02 nats. A fresh single-GPU process must repeat
itself bit for bit. A greedy pick reports its log-probability under the unscaled model
distribution.
