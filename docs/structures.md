# cti Architecture

This page describes the implemented V0 ownership map and dependency rules. It
documents current code rather than a proposed package tree.

## Runtime topologies

Every engine runs one process per rank; the parent spawns them and talks to rank 0:

```text
caller
  -> cti.llm.LLM                    prompt encoding, its chats preprocessed concurrently
       -> EngineClient              rank processes, commands out, replies back by id;
                                    offline, sampling requests over the request link
  -> rank 0 (EngineCore)            reads commands and requests, relays them to its peers
       -> ModeSwitch                which side holds the device, and when it changes hands
            -> InferenceEngine     a rank's inference: steps launched and settled
            -> TrainingScheduler  session, FIFO, window, optimizer steps, revision
       -> Rank                     the parts EngineCore composes, closed in reverse
            -> GPUWorker           ranks, streams, device memory, hosted models
            -> InferenceRunner     KV storage, CUDA graphs, rounds on the worker's streams
            -> TrainingRunner      training execution on the training stream
  -> rank 1.. (EngineCore)          apply rank 0's records in the same order
```

Online inference puts separately owned frontend processes in front of the same ranks:

```text
OpenAI client
  -> API server
  -> tokenizer / detokenizer processes
  -> inference protocol and ZMQ transport
  -> rank 0's EngineCore, which orders requests for every rank
```

Bounded generation is a frontend like the online tokenizer: it sends the same requests
over the same link. Scheduling calls the worker in-process inside its rank process, so
the hot scheduler-to-device path has no batch IPC.

See [`tensor-parallel.md`](tensor-parallel.md) for its process model.

## Ownership boundaries

- `cti.llm.LLM` is the public bounded facade. It owns one `EngineClient`, which it
  samples through, the pool that preprocesses its chats, explicit training sessions and
  deterministic shutdown.
- `cti.runtime.EngineClient` is the parent's only process supervision: it spawns
  every rank (each leading its own process group, bound to the exact pre-spawn parent
  PID on Linux), numbers its commands to rank 0, hands each reply to the call waiting
  for its id, and reclaims every rank on the first failure, reported or not. An online
  engine's client (`EngineClient.online`) also runs the detokenizer and tokenizer
  workers, fails with them and stops them as it closes; the client hands out the
  `TrainingSession` that `LLM` and `OnlineServer` return.
  `cti.runtime._process` holds the process-supervision primitives and
  `cti.runtime._protocol` the closed parent/engine messages that both sides
  import.
- `cti.runtime._core.EngineCore` is a rank's entry point and composition root. Its
  control plane (`cti.runtime._control`) is rank 0's `Leader`, which reads the parent's
  commands and the online requests, seeds requests when the plan serves only seeded
  rows, and relays each record to its peers, or a peer's `Follower`.
- The embedded `LLM` does not install process-global signal handlers. CLI owners
  translate their own first termination request into bounded cleanup; a repeated
  request force-terminates only their currently owned child set.
- `cti.runtime._core.ModeSwitch` alone owns which side (inference or training) holds the
  device and the switch order: the holder drains and releases, the worker moves the
  shared memory, both sides hear where the device went (`switched`). Serving comes first.
- `cti.inference._control._engine.InferenceEngine` is the inference side: it launches the
  scheduler's steps on the runner two deep; requests wait, or are refused, while the
  switch admits none. Its scheduler and cache manager are private collaborators.
- `cti.inference._control._scheduler.InferenceScheduler` owns
  admission, planning, request-local state, termination policy, usage accounting,
  and settlement.
- `cti.inference._control._scheduling.rounds.RoundLedger` owns the scheduler's DFlash
  rounds: each request's unsettled rounds and output credit, their capacity claims,
  round admission, the check of each round against its device record, its settlement
  up to a request's own stop, and round totals.
- `cti.inference._control._scheduling.rows.RoundSteps` is the scheduler's decode strategy when
  requests decode in rounds: each step is the round its `RoundLedger` plans, beside the
  waiting prompts' extensions that fit the rest of the token budget.
- `cti.inference._control._cache.CacheManager` owns logical request slots, page
  reservations, cache-index mappings, and reusable-prefix visibility. It owns no
  physical K/V tensor and no scheduler request.
- `cti.worker.GPUWorker` owns the rank's device resources: its parallel group,
  streams, device memory and hosted models, and moves the shared memory between the
  modes. `runtime._core.Rank`, which `EngineCore._start` builds, composes the
  executor and, when training is enabled, the `TrainingRunner` on them; the worker
  owns no request scheduling, execution or public phase.
- `cti.inference._execution.runner.InferenceRunner` owns the model, physical inference state (paged KV
  and, for recurrent layers, one recurrent-state row per request), GPU mirrors,
  inference preparation/execution streams and the training stream, sampling metadata,
  and its CUDA graphs (`GraphedForward`: decode graphs and breakable graphs in one
  `memory.GraphPool`).
  It sizes the inference state around the training memory plan
  (`cti.training._memory_plan`): what training keeps on the device beside
  inference stays out of the KV budget, and once the inference state exists it checks
  that training fits (`cti.worker.DeviceMemory`).
- `cti.training.runner.TrainingRunner` runs one rank's training: it plans a batch,
  runs its microbatches, commits or discards gradients, captures or restores the
  optimizer and suspends or resumes training state, on the training stream ordered
  after inference execution, behind one failure latch. It keeps no submission ids.
- `cti.training.scheduler.TrainingScheduler` schedules training on a rank: the
  session, the accepted batch and its microbatches, the submission FIFO, the
  accumulation window, optimizer steps, the model revision, the deferred
  forward/backward and optimizer outcomes, and the parent's replies and rejections.
  It is one side of the mode switch: entry, close, abort and the parent's serving pause
  and resume ask for a mode, and the switch's `switched` call answers or refuses them;
  its own state is the session's (closed, open, paused for serving).
  A `TrainingBatch` is its tokens in one layout (`PaddedTokens` or `PackedTokens`), the
  target positions its loss reads, and its objective (`SFT` or `GRPO`), which owns
  admission, normalization and its `Metrics`; the layout and the objective are
  independent. A training model's layers take their loss, log-probability and packed
  attention kernels from its `LayerServices`, like every other kernel; `TrainingRunner`
  drives the model through its one `TrainingModel` contract.
- The mode switch orders every change of hands: the holder drains and releases, the
  worker moves the shared memory, both sides hear where the device went. `MemoryManager`
  owns only physical residency; a resume whose memory does not fit is refused before
  anything is mapped (`ResidencyRestoreError`), the holder keeps the device, and the
  parent sees the switch as rejected (`SwitchRejectedError`).

## Dependency direction

```text
public API / server
        |
runtime composition and inference control
        |
execution and training owners
        |
models -> layers -> operation/provider wrappers -> kernels
```

Models are peers. They share lower layers and operations, never another model's
implementation. Scheduling does not import tokenizers, Transformers model loaders,
training objectives, kernels, or CUDA-graph-private state. Cache policy and scheduler
policy collaborate through narrow lifecycle operations without becoming one owner.
Long-lived resources are instance-owned; per-forward semantics use explicit immutable
values rather than mutable global context.

Kernel modules consume resolved tensors, shapes, and primitive settings; they do not
read engine environment policy or use engine logging. Engine selects execution
backends, including prefill/decode composition, cache layout and graph strategy.
Kernels own implementation requirements and operation-local tuning, with their
benchmarks and qualification evidence. Engine integration constraints and explicit
overrides remain valid; provider-specific defaults need not all move into kernels.
Choices affecting layout, workspace or capture must be resolved before resource
allocation and graph capture, not silently changed during replay.

TP/EP topology, process-group lifecycle, weight/expert placement and distributed
execution belong to engine. Kernels or external communication libraries implement
reusable compute/communication operations and may borrow communicators; they do not
own engine topology or scheduling. The runtime memory layer owns TMS integration and
residency decisions; kernels neither select nor manage residency.

## Hugging Face integration

Transformers is an intentional runtime dependency, not only a correctness oracle.
Each engine version pins one qualified official stable Transformers release. An
upgrade is a separate dependency-update PR, including input semantics and GPU
validation; installing an untested newer version is not an implicit upgrade.

- `resolution.py` loads the official config and `tokenizer/text.py` the official
  tokenizer; `resolution.py` reads the checkpoint's `generation_config.json` and
  tokenizer files for the stop set.
  `ModelConfig.from_hf` normalizes configuration once; execution layers consume
  engine-owned values, not Transformers version branches. Legacy checkpoint
  fields are normalized by Transformers, not a second engine compatibility
  framework.
- `ResolvedModel.stop_token_ids` holds the EOS ids, read once. The model's come first,
  following HF `generate`: `generation_config.json` when it declares
  `eos_token_id`, otherwise the model config, read from the text config of a
  multimodal model (HF would use no EOS when `generation_config.json` omits it).
  The tokenizer's `eos_token` follows unless already listed, as vLLM and SGLang
  also stop at it. `resolution.py` reads it from the tokenizer's files without
  constructing a tokenizer: `tokenizer_config.json` names it, and its id comes
  from that file's `added_tokens_decoder`, else from `tokenizer.json`'s
  `added_tokens`, as in Transformers' `Qwen2Tokenizer`. An `eos_token` that is
  not an added token is an error. `ResolvedModel.load` reads the set, so a
  malformed EOS field fails before the weights load, and the rank keeps that
  set instead of reading the files again. The scheduler finishes a request with
  reason `stop` on any of them unless the request sets `ignore_eos`.
- A request's own stops, its `stop_token_ids` and `stop` strings, end it too, with
  or without `ignore_eos`, and the scheduler decides them where tokens settle. It
  matches stop strings incrementally in the output's final text, decoded with the
  request's `skip_special_tokens`: the text of a `TextStream` the runtime tokenizer's
  `TokenizerTextDecoder` opens, the one owner of when text is final. The token
  completing one ends the request and stays in its output ids, and the reply names the
  string, so the renderer cuts the text where the string begins: offline sampling, or
  the API server, which holds back streamed text while its tail may begin a stop
  string. The detokenizer streams exactly that final text, so a match is in what it
  renders however steps and batches group the tokens. Only byte-level BPE alone, every
  token holding a byte (Qwen's), is proven to keep a prefix's text, which lets a step
  decode at once: its text is final but for a trailing incomplete character, which 3
  more ids settle, and a trailing run of replacement characters shows once a complete
  character follows it. Ids that render nothing (skipped special tokens, ids the
  vocabulary does not assign) stay out of the decoding. There a request's match state
  and a step's decoding are O(Q), the step's tokens, not O(the output); another decoder
  goes token by token and commits at the latest after 8 pending ids.
- The tokenizer/frontend uses public tokenization, chat-template and decoding
  APIs. Preserve model control tokens and template semantics rather than manually
  reconstructing prompts. Text input retains tokenizer special-token defaults;
  rendered chat templates must not acquire a second set of special tokens.
  `cti.tokenizer.PromptEncoder` is the one owner of this encoding: the online
  tokenizer worker, offline sampling and `LLM.prompt_encoder()` all go through it.
- Training accepts token IDs and masks. User-side preparation, including cookbook
  examples, may use the same HF tokenizer; conversation rendering does not become
  part of the training execution loop.
- VLM input processors (`cti.multimodal`) compose the tokenizer with the
  model's Transformers image processor; video is unsupported. They prepare model
  inputs such as resized uint8 patch pixels, image-grid metadata and rotary position
  layouts; pixel normalization, the vision tower, projector/merger, multi-axis rotary embedding and
  language-model computation remain engine-owned, using shared lower layers and
  kernels. Preprocessing is not a responsibility of the text tokenizer or scheduler.
- Template support alone is not tool-calling support. Tools, tool-result messages
  and response parsing require an explicit end-to-end API contract. V0 currently
  supports text chat, not tools or multimodal requests. Future input options must
  be wired through deliberately, not hidden behind an encode/decode-only adapter.

Model discovery/downloads (`huggingface_hub`) and tensor files (`safetensors`) are
separate dependencies. Native execution does not use HF `generate()` or Trainer;
HF models remain independent correctness oracles in tests. Kernel code does not
depend on Transformers integration. No generic adapter registry or global patches
are needed for these boundaries.

For upgrades, hold model/tokenizer assets fixed and check config/EOS/RoPE/dtype,
exact input token IDs, chat controls and streaming decode before GPU acceptance.
Retain existing inference and training numerical thresholds, graph/switching
checks, and relevant end-to-end performance comparisons. Requalifying a golden's
environment metadata does not authorize changing its numerical expectations.

## Current package map

The two distributions have separate repositories. This repository owns
`python/cti`; `cti-kernel` owns `python/ctikernels` and its operation tests. Engine
and integration tests run against the released wheel selected by the exact kernel package
requirement; kernel tests run in the kernel repository without engine imports. Versions
and release schedules are independent; kernel upgrades enter engine through explicit
dependency-update PRs. There is no nested source copy or old-import compatibility layer.
Independent artifacts, paired validation and the two repositories follow the
[package release boundary](package-release.md).

The kernel wheel includes its JIT sources and upstream notices. Native C++
namespaces and FFI symbols are intentionally unchanged: they are internal ABI
identities, not dependencies on the Python engine package. TMS remains an engine
dependency because allocation lifetime and residency are engine responsibilities.

- `cti.inference`: inference policy/results, request loop, transport, logical
  cache, scheduling, and inference-only wire protocol; the runner (`runner.py`) and, private
  to it, the step it executes (`_step.py`), sampling, inference CUDA graph owners
  and the DFlash round runner (`_speculative/`: draft model, target taps, device frontiers
  and round scratch).
- `cti.forward`: the forward batch a runner hands its model, with its `ImageBatch`.
- `cti.plan`, `cti.resolution`: rank-local execution values and backend
  resolution; `kvcache.plan_capacity` plans the inference state's capacity, with the
  allocator model `memory.native_backing`. Configuration has three phases: the request (the
  public `EngineOptions` and the internal `EngineTuning`, both in `config.py`, which the
  parent sends unresolved, neither building the model nor probing CUDA nor importing
  `plan.py`), the `Plan` (`plan.py`, resolved values only,
  built only by `ResolvedModel.plan` in `resolution.py`; the composition root
  (`EngineCore._start`) loads the checkpoint and any speculative method
  (`inference.load_speculative`), runs `ResolvedModel.check` to reject what they cannot serve
  before any device state exists, then plans once per rank for its device and injects the plan
  and checkpoint into the worker), and
  the launch metadata a rank publishes (`inference/_step.py`), which carries the plan back.
- `cti.training`: trainable selection, padded/packed SFT and GRPO, batching,
  staging, optimizer, checkpoint, and training protocol.
- `cti.models`: independent dense Qwen3, dense Qwen3.5, Qwen3.5-MoE and Qwen3-MoE,
  plus the DFlash draft (`dflash`), which the speculative seam's draft table resolves by its
  own architecture and which borrows its target's embedding, LM head and hidden-state taps
  through model hooks. The `qwen3_5` peer package keeps its configuration, text model and frozen
  vision tower (`vision.py`, with the tower's own LayerNorm, 2-D rotary and
  bidirectional attention) together; it imports no other model, and only the registry
  names it. The `qwen3_5_moe` peer package therefore holds a verbatim copy of that vision
  tower and its config, kept identical by a test until a model-neutral module shares
  them. Each model class declares the optional execution features it supports and
  the LoRA targets it adapts (`LORA_TARGETS`), and rejects Hugging Face config
  semantics it does not implement; `GPUWorker` rejects a configuration that needs an
  undeclared feature or target, and an unsupported config, before any device state or
  model exists. `training_contract.py` holds the Protocols training needs from a model:
  `TrainingModel`, whose input hook `BaseLLMModel` defaults, and one per padded or packed
  path; a class that declares `TRAINING` implements all of them.
  Each model also declares how its checkpoint tensors become its parameters
  (`checkpoint_mapping`, a `CheckpointMapping` with model-private
  transforms; Qwen3.5 builds its own from its Gated DeltaNet geometry); the shared
  loader applies that declaration and gives each tensor the dtype its parameter
  declares.
  Layers are complete when constructed. Each model instance gets one `LayerServices`
  in its `Site` from its composition root (the model host; one of its own for the
  DFlash draft): a layer takes its kernels from it at construction, one per geometry and
  device, and reads the DFlash taps from it at forward. Inference state is bound per
  layer: the runner groups the models' state layers (`kvcache.group_layers`) and binds
  each (`bind_state`) to its KV group's backend and index, or to its recurrent row.
  A model chooses an attention layer's fused prologue at construction with
  `AttentionLayer(prologue=PrologueFactory(...))`; the layer checks its geometry (Q/K
  norms of its head width and one kind, positive finite matching epsilon, rotary of its
  head width) and takes the `ctikernels.attention.inference.Prologue` of it, which it
  hands, with its norm weights and rotary rows, to the attention backend's
  `forward_prologue` to run on the layer's cache. Qwen3.5 passes `required=True`: the
  layer always runs the operation (256-wide heads, NeoX rotary of the first 64
  channels, zero-centered norms) and a geometry it cannot run fails construction. Its
  CuTe provider computes in FP32 with one rounding per output; it need not match the old
  two-rounding composition bitwise. Other devices retain the composed provider. Qwen3
  and the DFlash draft pass the default, which takes the operation's 128-wide variant
  (standard Q/K RMSNorm, full NeoX rotary) only where cti-kernels runs it as one CuTe
  kernel; elsewhere, and on a backend without `SUPPORTS_KV_PROLOGUE`, they keep the
  separate norm, rotary and K/V write.
  A model declares how many rotary position rows each token reads
  (`rope_position_axes()`: one by default, three for Qwen3.5 M-RoPE); the runner
  gives a step one row that every axis reads unless a prompt holds images, whose rows
  it checks against that count, so decode graphs keep one row. A model that
  encodes images does so in its top-level forward: one encoder call per step, whose
  tokens replace the placeholders' embeddings; `warm_up_image_encoder()` lets the runner
  measure the encoder's largest activation peak, which shares the peak reserve with the
  text forward's: only what the reserve cannot hold comes out of the KV cache.
- `cti.layers`: model-facing attention, linear, normalization, rotary,
  embedding, activation, MoE and Gated DeltaNet composition. A tensor-parallel layer
  receives its group from its constructor and declares the checkpoint window of each
  parameter it owns; the loader reads only those windows. `MoELayer` owns a replicated
  router, its `Experts` and an optional sigmoid-gated `SharedExpert`. The site's expert
  placement (`ExpertParallel`: whole experts, an equal range per rank) gives the `Placement`,
  the one owner of a rank's experts, and the loader reads only those; the shared expert
  splits by its intermediate width. With a shared expert, inference runs the backend's
  whole-block op when it offers one (`BaseMoeBackend.sparse_block`; the fused backend's is
  cti-kernels' `MoEBlock`, which picks a fused kernel or a composition per call); otherwise
  the layer composes the block around `BaseMoeBackend.forward`. Either way the ranks'
  partial sums meet in one all-reduce at the layer's exit. Inference projections without
  bias or LoRA, the LM head and the DFlash draft's context K/V included, run on
  `ctikernels.linear.Linear` through one `InferenceGemm` per call site; training and
  biased projections run `F.linear`. Qwen3.5's `o_proj` takes the attention output gate:
  its GEMM fuses `x * sigmoid(gate)` (FP32, one rounding) where cti-kernels accepts the
  call (`gate_fused`, decode-sized rows of measured shapes); otherwise the gate runs
  first. `Linear` dispatches from the per-device table cti-kernels ships (cuBLAS outside
  it); nothing is timed on the machine. Before planning capacity, every call site runs
  one zero-row call per M bucket that its table gives a CuTe kernel, up to the most rows a
  forward runs, so no forward or graph capture compiles one. The runner then binds one
  split-K workspace, sized for the largest call of any route, to all of them, so the KV
  budget counts it (startup logs the routes, the time and the workspace). The
  `gated_delta_net`
  family is a layer plus its `ctikernels.gdn` decode and prefill operations, which the
  layer takes from its model's services at construction, one set per head geometry,
  before the runner allocates any inference state. After the GEMMs, startup
  compiles the fused prologues and the CuTe recurrent prefill for every forward the
  runner can run (`warm_up_inference_ops`: its most rows, sequences and exported states),
  and the radix cache loads its key comparison, so a first request compiles none of
  them; other prefill providers still compile per shape on first use. The
  layer reads its rows of the runner's recurrent-state pool; under tensor parallelism it holds whole
  key-head groups, so that state is rank-local. A training site also gives it the shared
  training operation, which never reads the pool. For DFlash rounds each
  layer's `create_replay` gives its `LayerReplay` (verify, handoff, commit) in the model's
  `GatedDeltaNetK2`, the family's `RoundTransaction` over the same pool
  (`device_bytes` plans its one allocation); `cti.recurrent.RecurrentRounds` owns
  the round's transaction over the replays. Training selects the model-agnostic training
  operations (loss, log-probabilities and packed attention).
- `cti.attention`: inference attention metadata, backend selection, KV writes,
  and capture/replay preparation behind the layer boundary; layers and models never
  hold the KV pool or its write indices. A backend that declares `SUPPORTS_KV_PROLOGUE`
  also runs a layer's K/V-writing prologue on its cache (`forward_prologue`). Native
  provider calls live in `ctikernels.attention.inference` and receive tensors and numerical
  settings, never scheduler batches or cache owners.
- `cti.recurrent`: the runtime's view of recurrent layers (`RecurrentLayer`),
  per-batch slot addressing (`RecurrentMetadata`) and the decode-graph slot buffer
  (`RecurrentBackend`). It names no layer family. The backend schedules recurrent
  snapshot restores on the execution stream selected by `InferenceRunner`, addresses the
  captures a layer's prefill exports, and copies a decode step's exported row after the
  forward.
- `cti.kvcache`: KV groups (`PageGroup`: a tensor per layer in its spec's
  `RowLayout`, which an attention family implements and its backend views), the
  recurrent-state pool (one row per request slot and recurrent layer), the `KVLayout`
  that sizes them before allocation, `InferenceState.allocate`, the one place that
  builds them, and naive and radix cache mechanisms.
- `ctikernels` (separate repository): operation-local inference/training wrappers
  and numerical source.
- `cti.server`, `cti.tokenizer`, and `cti.protocol`:
  online inference frontend and wire ownership. At run time the server imports only
  the engine client, the options, the protocol (which owns the request types
  `SamplingParams` and `AdmissionPriority`) and utils. The tokenizer package root
  exports `PromptEncoder`, `IncrementalDetokenizer`, `load_text_decoder` and, loaded on
  first use, `tokenize_worker`, which the engine client starts. Each process builds one
  text decoder: a rank through `load_text_decoder`, the tokenizer workers and offline
  sampling through their `PromptEncoder`; `LLM.tokenizer` is the offline encoder's.
- `cti.multimodal`: frontend inputs of vision-language models. The package
  root holds the model-neutral contracts: image input values (`inputs`), the
  per-request rotary position rule (`rope.RopePositionLayout`) and the input
  processor with its architecture table (`processor`). Each supported model
  implements them in its own peer subpackage (`qwen3_5/`); there is no family base
  class. It imports no model, layer or kernel code. A request carries its
  `ImageInputs` and layout through `UserMsg`; the scheduler freezes each step's rotary
  positions from the layouts and packs the step's images into one `ImageBatch`: the
  images it encodes whole, and the encoder rows each prompt extension loads, an image
  split across prefill steps loading its part in each. The runner copies the batch to the
  device on its preparation stream. A request's pixels are released once its last
  image is submitted.
- `cti.distributed`: the instance-owned parallel runtime of one rank, with its
  CPU and device groups and their collectives. Tensor-parallel inference, with MoE
  experts placed over the same ranks, uses it on one node; SP, EP training, DP, PP and
  multi-node execution are not implemented.

Runtime imports among `engine`, `models`, `layers`, `attention`, `recurrent`,
`kvcache` and `moe` follow one acyclic allowlist: engine imports attention, kvcache,
models, moe and recurrent; models import layers; layers import kvcache. Below the
engine, engine and other runtime owners are imported for annotations only. Runtime
files (engine, memory, residency, worker, kvcache, inference, training, recurrent,
the model base and the layer package root) name no mixer family.
`tests/quality/architecture` enforces both rules.

Attention workspaces and graph-buffer lifetimes remain controlled by the engine.
Each backend declares the workspace it allocates (`BaseAttnBackend.workspace_bytes`);
InferenceRunner reserves it before sizing the KV pool.
The FlashInfer operation retains its native wrapper across layers and graph replay;
its boundary does not copy tensors, synchronize streams, or select providers per
forward. Varlen attention (packed training and bidirectional vision attention) is a separate leaf, so importing
inference does not load its provider.

Layers break breakable graphs only through `batch.graph_breaks.eager` (attention and the
Gated DeltaNet core, which read the caches); a break calls its backend like an eager
forward, so no backend or model declares graph support. The MoE block runs inside a
segment over the whole row bucket, its padding rows' outputs dropped, as vLLM and SGLang
capture it.

The repository-root `cookbook/chat.py` is a source-only HTTP client, not a package or
server mode. It owns local conversation history and presentation but no engine,
frontend, tokenizer, or child-process lifecycle.
