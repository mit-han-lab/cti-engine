<!-- SPDX-License-Identifier: MIT -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors -->

# DFlash contract

The v1 draft lives in `python/cti/models/dflash/model.py` and DFlash2 in
`dflash/dflash2.py`; `dflash/__init__.py` exports both and `DFlashConfig`. The draft table
is the speculative seam's (`inference/_execution/_speculative/method.py`): `load_speculative` reads the
draft checkpoint, picks its class by architecture and checks it against the target, and the
composition root hands the method to `ResolvedModel.check` and `ResolvedModel.plan`.

These CPU oracles pin inputs, round counts and attention masks. They do not
implement execution or qualify checkpoint numerics. Only the target authorizes output.

One shared runtime serves two draft classes: `DFlash2DraftModel` extends
`DFlashDraftModel` with per-layer two-tap dynamic convolutions and a candidate
selector. Only the draft forward, the proposal step and sampled verification's q differ;
verification, sampling, commit and cache/graph lifecycles are shared. Upstream reference:
[z-lab/dflash@07ebd93db9f4 (`dflash/model.py`)](https://github.com/z-lab/dflash/blob/07ebd93db9f4/dflash/model.py).

## Pinned inputs

| Target @ revision | Draft @ revision |
| --- | --- |
| `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` | `z-lab/Qwen3-8B-DFlash-b16@9b41424b7109f9c5413454f481b09a82b85333f4` |
| `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` | `z-lab/Qwen3.5-9B-DFlash@5fc3b3d474760f18c516db87d84c37edbfd3ede6` |
| `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |

Run `python -m tests.support.dflash` with optional `--cache-dir`.
`CTI_DFLASH_TARGET_REPO` selects a pair (default Qwen3-8B);
`CTI_DFLASH_TARGET_REVISION`, `CTI_DFLASH_DRAFT_REPO` and
`CTI_DFLASH_DRAFT_REVISION` must match that pair's pins.

Preflight uses `local_files_only=True`. It checks snapshot repo/revision, both configs,
target tokenizer files, and every indexed shard or the single weight file. Drafts
share the target tokenizer. Missing snapshots/files, broken links, unresolved LFS
pointers, invalid indexes and wrong revisions **block qualification, never skip**.
An existing but invalid index cannot fall back to a single weight file.
Readability does not establish content hashes or numerical correctness; synthetic
CPU fixtures do not establish that real checkpoints are cached.

## Draft config

| Contract | DFlash v1 | DFlash2 |
| --- | --- | --- |
| `architectures` | `["DFlashDraftModel"]` | `["DFlash2DraftModel"]` |
| `is_causal` | Reject any key, including false/null | Require explicit boolean false |
| Trained block (`block_size`) | 16 | 8 |
| Layers | Qwen3: five full; Qwen3.5: five sliding + one full | Five sliding only |
| Sliding window W | `sliding_window`, per sliding layer (Qwen3.5: 4096) | 2048 |

Read the trained block from `dflash_config.block_size`, or top-level `block_size` for Qwen3
v1. A round's width Q is `speculative.num_speculative_tokens` + 1 (1–16 drafts), whatever
the trained block; unset, it is the trained block. Rows past the trained block are mask
rows, and DFlash2's convolution restarts at each Q-row round block, as SGLang and vLLM
run it.
DFlash2 requires `dflash_config` fields `conv_kernel_size=2`, `conv_group_size=16`,
`selector_rank=256`, `selector_top_k=16`; v1 rejects these extension fields.
The pinned pair determines the required draft architecture; overrides cannot
silently substitute another draft type. DFlash2 requires exactly five
`sliding_attention` entries and `num_hidden_layers == len(layer_types)`.

## K1 sampling

Phase 1 supports seeded and unseeded requests with any combination of
greedy/temperature/top-k/top-p, using the same p/q construction regardless of seed.
Target p is full-vocabulary **FP32 logits → temperature → top-k → top-p**.
For v1, q is a point mass on the draft argmax (smaller ID on ties): accept draft d with
probability p(d); at the first rejection sample p without d, renormalized (p itself at zero mass).
For DFlash2, q is FP32 over 16 selector candidates, conditioned on the chosen
predecessor (anchor at position 0), using temperature only; retain IDs and conditional q, zero outside.
DFlash2 uses `min(1, p(x)/q(x))` acceptance and samples normalized full-vocabulary `[p−q]+`
at the first rejection (p at zero mass). Both keep the longest accepted prefix;
if all proposals pass, sample the bonus from the final target row.
Normalize identity filters before dispatch; greedy is T=0 only, using the longest
exact prefix and target argmax correction/bonus. T>0 always uses the stochastic contract, even k=1.
Truncation consumes no randomness. Ties break by ascending global token ID:
top-k boundary ties keep smaller IDs; top-p sorts by probability descending then ID ascending,
retaining the token crossing the threshold. Top-16 and argmax ties also keep smaller IDs.
Candidate CDF order is ascending global token ID. Seeded reproducibility matches seeded
temperature-only at the same configuration; bitwise across batches only in the batch-invariant tier.
Unseeded requests receive an immutable engine-assigned seed at admission.
RNG follows the versioned K1 ABI: Philox-4x32-10, seed split into two 32-bit keys,
`(cursor_lo, cursor_hi, domain, i)` counter; proposal/accept/correction/bonus tags 1/2/3/4.
Use word0's high 24 bits mapped to [0,1); ordinary sampling has a disjoint namespace.
The device response cursor is the round start for proposal and verify; i is the
proposal/accept position, or 0 for correction/bonus. Streams are independent of
batch composition, padding, request reordering and slot reuse; TP shares request identity.
Model code owns the DFlash2 selector weights, its lattice and the serial walk; K1 supplies candidate
sampling and sparse verify arithmetic. The lattice is scored as in vLLM and SGLang: the hidden projection,
A[pred]·Ph and the 16×16 predecessor×candidate edges in the model dtype (FP32 accumulation), plus the
FP32 unary logits in FP32. The walk gathers each position's scores by the chosen predecessor's
candidate index. Cross-TP model bitwise parity is not required. Under TP, the LM head merges each shard's top-16
into the global top-16 (`gather_topk`); the selector is replicated, so every rank scores the same
lattice and walks the same chain, drawing with the request's replicated counter-based RNG, without
a broadcast. One sampling rank builds p from gathered global logits and broadcasts accepted IDs,
correction/bonus and a; all ranks commit m=a+1. Shard-local normalization is forbidden.

## Acceptance and visibility

A verify block contains one already-emitted, uncached anchor and Q−1 draft tokens.
For greedy verification, a is the longest prefix agreeing with target predictions,
0 ≤ a ≤ Q−1. Commit exactly a+1 inputs: anchor plus accepted drafts.
Emit a+1 new tokens: accepted drafts plus the target correction at the first
mismatch, or the final target row's bonus if all drafts agree.
The correction/bonus remains uncached and becomes the next round's anchor.
Acceptance rate is `sum(a) / sum(actual verified drafts)`, excluding anchors and padding.

EOS (inclusive), the remaining output budget and the sequence capacity cap the
committed length v ≤ a+1: the round commits and emits only those tokens; a stays the raw
acceptance. A private tail must not be published.

## K2 verify / accept / commit

Follow the `cti-kernels` GDN replay (delayed write-back) contract:
verify is read-only for all live state/history/ring/metadata; acceptance length stays
on device, with the committed length v in `int32[B]` driving immediate canonical
replay/commit on the same stream. No host acceptance readback or lazy commit; `-1` rows
(done requests) are no-ops and must not access state/history/RNG. Ordinary decode keeps its
existing default.

## Execution

`speculative.model_path` (`SpeculativeOptions.model_path`) enables DFlash
v1 or DFlash2 for a Qwen3, Qwen3.5 or Qwen3.8 target, dense or MoE. A LoRA target verifies with its
adapter, and an engine that trains serves its rollouts with DFlash (the draft stays frozen).
Rounds sample as ordinary steps do, through the same implementation (`inference/_execution/_sampling.py`):
selected and top logprobs (raw, or with `logprobs_processed`: the p a row's verdict draws from),
min-p, and presence, frequency and repetition penalties, which count at verify position i the
row's context plus the i drafts before it. Only verification applies a round's logits
processing (penalties, stop masks and logit biases); its drafts never follow it, as in vLLM
and SGLang, and K1 stays exact since q is the law the drafts were drawn from. A target reuses prefixes through the radix cache, which
`cache.prefix: auto` selects as without DFlash (naive only for a Qwen3.5 target whose prefill
exports no states): the draft context shares the target's pages, so a hit serves both, and a
Qwen3.5 hit restores its recurrent state, which the next round hands off. This is unqualified.

Temperature-0 requests are greedy and carry no filters (the sampling law refuses them).
Rounds verify greedily when every row has temperature 0; otherwise K1's `probabilities`
and, by the draft's q, `verify_point_mass` (v1) or `verify_sparse` (DFlash2) sample them,
temperature-0 rows still greedily. A request's seed
is its own; the primary rank's ingress stamps an unseeded one with a random nonce plus its
count, so peers replay the same seed. The seed draws the whole output: the first token in the
ordinary namespace at response position 0, then every round. A row's RNG cursor is its anchor
position minus the prompt length plus one, read from the device frontier. Ordinary steps draw
from the same K1-filtered target p.

A round drafts Q−1 tokens for up to four running requests (and `max_batch_tokens // Q`),
then verifies `[anchor, drafts]` on the target with causal fixed-query attention. Q is
`speculative.num_speculative_tokens` + 1, by default the draft's trained block (16 for v1,
8 for DFlash2), one Q per engine. The proposal step runs inside the
draft (and its captured graph): v1 takes the target head's argmax at each position.
DFlash2 runs its layers with each branch wrapped in a grouped convolution along the block.
It then takes each position's top-16 target-head candidates (in ascending ID order) and
walks one chain from the anchor, scoring each position's candidates after the token chosen
before it. A greedy row takes the best candidate; a sampled row draws from the candidates'
temperature softmax with K1's `sample_candidates`, which also records the FP32 q its verify
reads. Candidate IDs and q live at fixed addresses, `[B, Q−1, 16]` each. Target
and draft KV are separate pools that share the page mapping; draft KV below a request's
frontier holds the projected taps of its committed tokens, which every target forward
stores. Each request row keeps its frontier F (int64) and a done flag on the device. A
round commits v ≤ a + 1 tokens: those within the output budget and the sequence capacity,
through the first stop token; it writes them to the token table, advances F by v and marks
the row done when it finished. A done row runs as Q dummy queries on its execution row's
guard KV rows (taken from the pools' capacity) and commits nothing; positions past the
sequence capacity use guard rows too, so every request decodes in rounds to its end.

A request has up to two unsettled rounds. Each holds a capacity claim reaching Q past the
furthest anchor its predecessors may leave; claims settle oldest first with the committed
length, and pages past both the new frontier and every pending claim return. A round its
predecessors certainly finish is not submitted, and one they may finish only rides along
with other requests' rounds; a successor of a request that ended (EOS or abort) still
runs, and its settlement only retires its claim. Results leave the round
scratch for one of two device records, and reach pinned host memory on a copy stream, so
the next round is submitted before the host reads this one; settlement checks the record
(frontier, committed length, done) against the host's budget and stop rules. Prompts and
rounds mix without draining. As vLLM schedules running requests before waiting ones, each
step (`RoundSteps`) runs the round of every row that can run and, in the same target
forward, the waiting prompts' extensions that fit the rest of the token budget, in chunks
where it cuts them: one target forward runs the extensions, then the verify rows (attention
writes all K/V once and attends the two groups through FlashInfer and `DraftAttention`;
Gated DeltaNet layers prefill one group and verify the other), one LM head serves both,
and the step completes as one ticket. A captured round size replays its draft graph
before the target forward, which replays a breakable graph when its tokens fit a bucket. A prompt runs alone only when no row can run. A request whose prompt finished in a
step joins the next round before that step settles, since the device already holds its
anchor; if the first token stops the request, that round's row is discarded like an
aborted one. A request's own stops are checked on the host only: a round reaching one
publishes its tokens through it and is discarded the same way, committing no KV. Admission
reserves Q − 1 positions past the output budget, within the sequence capacity. Each decode
graph batch size (by default every round size, 1–4) captures two graphs, each keyed by
shape only: the draft's (block placement, draft forward and proposal) and the target's
forward over the block rows, one per status slot. The verdict (LM head, logits processing,
K1), commit and log-probabilities run eagerly after them, each only when a row needs it. A
round replays the smallest captured size that holds it, padded with done rows past the
request rows, and runs eagerly only past the largest. So that no round compiles on first use, startup runs one round of each kind (for a
Qwen3.5 target, one per kind and status slot, below) at every size whose warm-up blocks (Q KV
rows per request) fit the KV pools. A round's bookkeeping is two `ctikernels.speculative`
kernels: `prepare_block` writes the block inputs and the KV index CSR from the device
frontiers, `commit_block` accepts, caps and commits; both compile at bind. Verify and draft
attention run on `ctikernels`' `DraftAttention` (its CuTe provider for head dimension 128 on
RTX 5090, RTX PRO 6000 and GB10, otherwise FlashInfer): causal for the verify, and per draft
layer the mask its config declares, full for full layers and causal over the layer's window
for sliding ones (DFlash2: non-causal, every block key and the context in the window).
`max_seq_len` is bounded only by the draft's rotary cache and the KV capacity. Sliding layers
still keep their full KV. `attention.FixedQueryBackend` owns these operations, sized by the
configured sequence capacity, for both pools; the target's other batches keep its paged
backend, and the draft pool has none. Every operation borrows one float scratch, since rounds
run on one stream.

With `parallel.num_ranks` > 1, every rank runs every round. The draft splits its query and
KV heads (KV heads copied where ranks outnumber them) and its MLP channels as the target's
layers do, and its draft pool holds the rank's heads; every rank projects the whole taps, which
are the same on every rank. A greedy round's drafts and target tokens agree on every rank by
construction: each rank's argmax reduces the same gathered words, and a DFlash2 rank walks
the same chain, sampled or not, from the same global candidates and seeds. Under the
engine's checks, the ranks compare their drafts every round, and settlement fails a round
whose drafts differ. A sampled round's verdict, its
target tokens and accepted counts, is drawn by the primary rank alone from the gathered logits
and given to the peers on the device before any rank commits it, inside the captured round.
A Qwen3.5 target's K2 replay and its snapshot rows hold the rank's Gated DeltaNet heads.

A Qwen3.5 target verifies at M-RoPE positions (sequence position plus the request's rotary
offset, on every axis) and runs its Gated DeltaNet layers through K2's
`GatedDeltaNetReplay`. The recurrent pool stays the canonical state: verify only reads it,
commit replays the committed tokens into it, and a slot that an ordinary forward advanced
is handed off in place before its next round; a done row's slot is -1, untouched. Each round is one K2 invocation on
one of two status slots, with one captured target graph per round size and slot.
Its settlement checks every layer's status before the round's tokens are published and
fails the engine otherwise. Where K2's CuTe phases are qualified (RTX 5090, RTX PRO 6000,
GB10), verify and commit run trusted: they report no status, so settlement covers the
round's handoffs, and commit borrows its status slot's arrival counters. Startup runs a
captured size's round of each kind at each slot right before that slot's capture, and an
uncaptured size's as often.

## Draft attention masks

Draft context and query logical `positions` are `int64[T]`, distinct from target
rotary positions. Qwen3 uses 1D positions; Qwen3.5/3.8 M-RoPE uses
`rope_positions: int64[3,T]`, preserving image offsets. GDN consumed positions are
independent int64 values; never borrow a rotary axis. Widen before position arithmetic.

For existing context/block keys k, logical query position p and block start s:

| Layer | Visible keys |
| --- | --- |
| v1 full | All context and block keys |
| v1 sliding | `p−W < k ≤ p` |
| DFlash2 sliding | `k ≥ s` or `p−k < 2048` |

Each query has its own left boundary. Sliding masks exclude context at distances
W and W+1, include W−1 and self; only DFlash2 includes future block keys.
The v1 full layer also includes future block keys. These are draft masks;
target verification remains causal.
