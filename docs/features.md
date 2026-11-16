# cti V0 Feature Contract

This page describes the supported V0 surface. Source-level seams retained for future
models or parallel execution are not release claims.

| Capability | V0 status |
|---|---|
| Dense Qwen3 inference on one GPU | Implemented |
| Dense Qwen3 (TP2/TP4) and Qwen3.5 text (TP2) tensor-parallel inference on one node | Implemented but unqualified; see [prerequisites](tensor-parallel.md#prerequisites) |
| OpenAI-compatible online serving and bounded typed generation | Implemented |
| Online serving and local typed training sharing one supervised backend | Implemented |
| Strict startup YAML configuration for Python and CLI entrypoints | Implemented |
| Right-padded and native packed SFT | Implemented |
| Reference-free padded and packed GRPO | Implemented |
| Full-tail inference/training memory switching | Implemented |
| Per-layer KV ownership and contiguous-tail mechanism | Maintained seam; selection policy deferred |
| Qwen3-MoE BF16 inference with prefix reuse and breakable CUDA graphs, on one GPU or tensor-parallel attention with expert- or expert-tensor-parallel MoE on one node; attention and per-expert LoRA and LoRA training | Implemented but unqualified |
| Dense Qwen3.5 inference with offline and online image input, LoRA, text LoRA training, and prefix reuse of text and images (an encoder-output cache reuses repeated images) | Implemented but unqualified |
| Qwen3.5-MoE BF16 inference: one GPU with offline and online image input, prefix reuse of text and images (an encoder-output cache reuses repeated images), DFlash v1, mixer and per-expert LoRA and text LoRA training, or tensor-parallel attention with expert- or expert-tensor-parallel MoE on one node | Implemented but unqualified |
| Training compile or CUDA graphs | Deferred |
| DFlash v1 / DFlash2 for Qwen3 / Qwen3.5 / Qwen3.8 targets | [Contract](dflash.md); v1 for Qwen3 and Qwen3.5 targets, dense or MoE, and DFlash2 for Qwen3.8, with prefix reuse, tensor parallelism, prompts mixed into rounds, LoRA targets, training rollouts, logprobs and steering, implemented but unqualified (DFlash2 CPU-tested only) |
| Tensor-parallel LoRA and training on one node (MoE: expert- or expert-tensor-parallel sparse blocks) | Implemented but unqualified |
| Reference KL and SP | Deferred |
| DP, PP, CP, multi-node execution, and HTTP training | Out of scope |

## Online Serving

cti supports online inference with an OpenAI-compatible API server. It provides
`GET /v1/models` and streaming or non-streaming `POST /v1/completions` and
`POST /v1/chat/completions`, including usage accounting and cancellation. The model's id
is its `model.path` as configured. For command-line options, run
`python -m cti --help`.

Both endpoints take OpenAI's `n`, `temperature`, `top_p`, `presence_penalty`,
`frequency_penalty`, `seed` and `stop`, and
vLLM's extensions `top_k`, `min_p`, `repetition_penalty`, `ignore_eos`, `stop_token_ids`,
`include_stop_str_in_output`, `skip_special_tokens` and `return_token_ids` (the prompt's
and the output's token ids). Each prompt gets `n` choices, choice `j` of prompt `i` at
index `i * n + j`, seeded `seed + j` when the request has a seed, as vLLM does; usage
counts each prompt once and every choice's output. Chat takes `logprobs` and
`top_logprobs` (0 to 20, which needs `logprobs`) and `/v1/completions` the legacy
integer `logprobs` (0 to 20), with `echo` the prompt's too, its first token's null;
every output token reported in the token ids has its entry, a stop token's too, and a
token's text is the token decoded alone. A prompt whose logprobs are asked for is computed
in full, never restored from the prefix cache. The endpoints' logprobs read the model's
distribution. Offline, `SamplingParams(logprobs=True, logprobs_processed=True)` reads the one
sampled from instead, after penalties, temperature, top-k, top-p and min-p, as RL needs.
`/v1/completions` takes `max_tokens` (default 16), text, token ids, or a batch of either,
and `echo` (with `max_tokens: 0`, the prompt alone; such a request still reserves one
completion token of context). Chat takes `max_completion_tokens` or the deprecated
`max_tokens`, not both; with neither it is 16 for now. The engine ends a request at the
token completing a stop string or at a stop token id; the text ends where the stop string
begins, or after it with `include_stop_str_in_output`, which a stop token id's text never
shows; the token ids run through the completing token, a stop token's too (offline,
`output_token_ids` leave out a terminal stop token). A request without a `seed` samples
from the shared generator; one with a `seed` samples from its own stream, so it repeats
its output at the same configuration and batch, and across batches only as far as the
numerics are batch invariant, which cti does not promise; seeded and unseeded
requests share batches, online and offline. DFlash seeds every request. Unknown fields
are rejected, and known fields the engine does not honor are rejected by name with code
`unsupported_parameter` unless their value asks for nothing: `best_of`,
`suffix` and chat's `response_format` are not supported.

Chat takes OpenAI's `tools`, `tool_choice` and `parallel_tool_calls`, with assistant
messages' `tool_calls` and `tool` messages' `tool_call_id` in the history. The chat
template renders the tools: all of them for `auto` (the default with tools) and
`required`, only the named one for a named choice, none for `none`. The model's calls are
parsed out of its text into `message.tool_calls` (streamed as `delta.tool_calls`, each
call's `id`, `type` and `name` on its first chunk), and a choice that called a tool and
stopped ends with `finish_reason` `tool_calls`. The format follows the chat template:
`hermes` for Qwen3 and Qwen3-MoE, `qwen3_coder` for Qwen3.5 and Qwen3.5-MoE;
`--server.tool-call-parser` overrides it. A model whose template takes no tools answers a
request for tools with a 400. A call of a tool the request did not offer, a call cut off
by `max_tokens`, or a `<tool_call>` mentioned mid-line in a `<think>` block stays text.
A hermes call's arguments stream as the model writes them, so one whose JSON turns out
malformed still closes as a call with those raw arguments, streamed or not. An empty
`tools` list is a 400, as in OpenAI and vLLM.
Calls are parsed, not constrained: `required`, a named choice, `parallel_tool_calls` and
a function's `strict` do not force the model's output.
Chat also takes vLLM's `chat_template_kwargs`, the chat template's own variables such as
`{"enable_thinking": false}`; the template decides what an unknown key means, and a key
that is a renderer parameter (`tokenize`, `chat_template`, ...) is a 400.
Both endpoints also take OpenAI's `logit_bias` (token id to a bias in [-100, 100], added
to the logits) and vLLM's `min_tokens` (until that many tokens are out, every stop token
is masked); the extension `priority` (`high`, `normal` or `low`) orders waiting requests'
admission. Offline, `SamplingParams` takes the same `logit_bias` and `min_tokens`; it
requires `max_new_tokens`, its `temperature` defaults to 1.0, and `top_k` None keeps the
whole vocabulary (the endpoints' -1). Temperature 0 is greedy and refuses `top_k` and
`top_p`, which the endpoints drop at temperature 0.

A request whose prompt plus `max_tokens` exceeds the model's maximum sequence length is
rejected, as OpenAI does: HTTP 400 with code `context_length_exceeded`, streaming or not,
and `RequestRejectedError` from the Python API, whose `kind` is the same
`RejectionKind` the server maps to its status and code. A streaming response therefore starts
with the first token. `LLM.generate`/`sample` returns every result or raises
`RequestRejectedError` naming the first rejected prompt: a token-id or processed prompt
that breaks the rule rejects the command before any prompt runs; a text prompt is
checked when it is tokenized, as the command's prompts are fed in, and then the prompts
after it never run and the ones already running are aborted, as a client abort would.

The repository-root `cookbook/chat.py` preserves terminal chat as a source-only
client of those endpoints. It owns local history and presentation only; it never
imports, starts, or shuts down the engine and is not included in the release wheel.

Only serving is online. V0 training remains a local, typed, single-operator
`TrainingSession`; asynchronous training Futures are not a network service. The V0
target attaches the online inference frontend and that local training control to one
supervised backend and one model revision. It does not add an HTTP training route,
training frontend, or multi-tenant training admission.

Applications that need both surfaces use `cti.server.OnlineServer` as the owner:
OpenAI-compatible clients send inference requests to its HTTP endpoint, while the
single local operator uses `server.training_session()` on the engine that same server
owns, not a second runtime. See
`cookbook/online_training.py` for the complete composition.

For online GRPO, `server.sample(prompts, params, priority=...)` takes `LLM.sample`'s
arguments and returns its `SamplingResult`, ready for `TrainingBatch.grpo`. Its requests
travel the server's own path and are admitted with HTTP requests; `priority=LOW` lets
those go first. LOW gets no minimum share and can wait indefinitely while `NORMAL` traffic
saturates the engine; use `NORMAL` when the rollout must not wait behind HTTP
traffic. A seeded prompt may sample differently across calls when one hits the prefix cache,
as the prefix-hit note under Radix Cache says. It
takes text and token-id prompts only, and raises when a publish lands while it samples,
as its samples would straddle two model revisions.

The local operator may keep the same session alive across an explicit
`pause_for_serving()` / `resume_training()` interval. Async applications use the
event-loop-safe `pause_for_serving_async()` / `resume_training_async()` wrappers;
both surfaces execute the same operation. Automatic inference-arrival handling and
explicit control enter one coordinator-owned serving-demand transition and a configured
`TrainingPauseBoundary`: `MICROBATCH`, the default, yields after the current microbatch
obtains CUDA completion evidence, while `LOGICAL_BATCH` completes the accepted logical batch.
The request remains cancellable while held outside scheduler admission. Serving then keeps
the device and admits every new request until none runs or waits; only then does
automatic handling return the device to training, however long training's work waits.
A session's close or abort is the exception: it takes the device back, serving draining
for it, until it has run. Neither transition commits an optimizer step implicitly, and
every return to inference invalidates prior logical KV state.

While serving holds the device, `TrainingStateResidency` decides what training's state
leaves it with. `RESIDENT` keeps everything. `OFFLOAD_OPTIMIZER`, the default on a discrete
GPU, releases the AdamW moments. `OFFLOAD_ALL` also releases the FP32 masters and an open
accumulation window's gradients. Released state lives in pinned host memory: the masters
and moments are mirrored asynchronously after every optimizer step, so the switch to
serving copies at most the gradients. On the return, the copies back overlap training up
to the first read. The KV pool is sized with the bytes released. Unified-memory devices
(GB10) default to `RESIDENT`.

## PEFT Adapters

`lora.adapter` names a PEFT adapter directory (`adapter_config.json`,
`adapter_model.safetensors`) that the engine loads at launch, on one rank or tensor-parallel,
into the same LoRA factors it trains. The adapter's `target_modules` (a list of module names),
`r` and `lora_alpha` set the LoRA targets, rank and scaling (`lora_alpha / r`); a per-expert
rank pads with zeros to a multiple of 8. Each model maps PEFT's per-module factors through its
checkpoint mapping: `q_proj`, `k_proj`, `v_proj` onto `qkv_proj` slices, Qwen3.5's `q_proj`
(query and gate rows interleaved per head) onto two slices under one A, `in_proj_qkv` onto the
Gated DeltaNet's q, k and v under one A. Routed experts come in two formats: Transformers 5's
fused `mlp.experts.gate_up_proj` and `down_proj` adapted through `target_parameters` (one A
for each expert's gate and up, `lora.expert_layout` `shared_a`, which training uses), and
Transformers 4's per-expert `mlp.experts.{e}.{gate,up,down}_proj` (an A for each,
`per_projection`, served only). A slice the adapter leaves out (a module or an expert) is
zero, and one log line names them. The engine refuses `use_dora`, `use_rslora`, `modules_to_save`,
`rank_pattern`, `alpha_pattern`, `layers_to_transform`, `fan_in_fan_out`, a `bias` other than
`none`, a regex `target_modules`, and modules no LoRA target holds (embeddings, the LM head,
the vision tower, the router). The adapter loads only at launch, one at a time; there is no
runtime load or export yet.

## Usage and Metrics

A chat completion's `usage` holds `prompt_tokens`, `completion_tokens` (a terminal stop
token included), `total_tokens`, and `prompt_tokens_details.cached_tokens`, the prompt
tokens served from the prefix cache. A stream sends it in a final chunk when the request
sets `stream_options.include_usage`. Offline, `SampledSequence.cached_prompt_tokens`
holds the same count.

The engine logs totals at INFO every 10 s while it works and once when it falls idle:
the requests finished since startup (aborted ones are not counted), how many hit the
prefix cache, their prompt, cached and generated tokens, and the KV pages and snapshot
rows in use by requests, cached for reuse (evictable), and in total. With DFlash it adds
the settled rounds of live requests, their accepted drafts, and the mean acceptance
length (accepted drafts plus one per round, before EOS and budget clipping).

## Single-GPU V0

The V0 release qualification is single-GPU. Qwen3, Qwen3.5, Qwen3-MoE and Qwen3.5-MoE
also have an unqualified tensor-parallel path across the GPUs of one node, inference,
LoRA and training alike, as each model's capabilities declare; see
[`tensor-parallel.md`](tensor-parallel.md) for its scope and prerequisites. SP training
remains follow-up work. DP, PP, CP, multi-node execution, and HTTP training are not V0
goals.

The V0 release qualification targets one RTX 5090 (SM120, x86-64), one RTX PRO 6000
(SM120, x86-64) and one DGX Spark GB10 (SM121, Arm64). A named qualification validates
the exact device identity and native provider images; it does not generalize to another
GPU. DGX Spark uses unified memory, so a physical-memory measurement is reported as
unavailable when the platform cannot expose it rather than being inferred from
process-local allocator telemetry.

V0 configuration supports the typed Python constructors, the inference CLI, and a
strict YAML entrypoint shared by `LLM.from_yaml`, `OnlineServer.from_yaml`, and
`python -m cti --config`. YAML constructs the same typed policies as Python;
it does not add a second runtime or a network training surface. See
[`configuration.md`](configuration.md) for the schema, precedence, and examples.

## Supported Models

The maintained V0 model profile is dense Qwen3 on one GPU, each size gated against
Transformers (`tests/hf_parity`): Qwen3-0.6B, 1.7B and 4B on an RTX 5090, Qwen3-8B on an
RTX 5090 and on RTX PRO 6000s (TP1 and TP2), with unqualified
[tensor-parallel inference](tensor-parallel.md) on one node. Unregistered
architectures fail during model resolution rather than falling through to an
unqualified compatibility implementation.

Checkpoint semantics the model classes do not implement also fail before any
weight loads: sliding-window and other layer types a model does not implement,
attention projection bias, logit soft-capping, Qwen3-MoE dense MLP layers or non-SiLU
experts, a quantization format no recipe claims, and RoPE types other than `default`,
`llama3` and `yarn` (whose cos/sin caches match Hugging Face). So do a `model.max_seq_len`
above the model's `max_position_embeddings` and a non-`auto` `moe_backend` for a
dense model. With YaRN, `max_position_embeddings` sizes the RoPE cache, so set it
to the extended length, for example 131072 for Qwen3's factor-4 recipe.

Two quantized checkpoint formats load: block FP8 (HF `quant_method: fp8`, E4M3 weights
with one scale per 128x128 block, activations quantized at run time) and ModelOpt
`MIXED_PRECISION` with per-module FP8 and NVFP4 linears; NVFP4 routed experts are not
implemented. Block FP8 checkpoints serve and train LoRA adapters. ModelOpt checkpoints serve,
and serve LoRA adapters, but do not train: a training engine refuses them, by recipe, before
the model is built.

Qwen3-MoE resolves from `Qwen3MoeForCausalLM` checkpoints such as Qwen3-30B-A3B, its
Instruct-2507, Thinking-2507 and Base variants, Qwen3-Coder-30B-A3B-Instruct and the
235B-A22B geometry. It runs BF16 inference with text prefix reuse and breakable CUDA
graphs, on one GPU or [tensor-parallel](tensor-parallel.md#moe) with whole experts per rank (expert parallel, the default) or every expert split by intermediate width (expert tensor parallel, `parallel.enable_expert_parallel: false`); it implements attention projection bias and both
`norm_topk_prob` routings. LoRA adapts the attention projections (`attention`)
and every routed expert's projections by per-expert factors (`experts`, unmerged, one A for
each expert's gate and up as PEFT adapts a fused `gate_up_proj`; the router stays frozen), and
trains both (SFT and GRPO, padded and packed) through the frozen sparse
blocks, whose routes take the inference top-k's FP32 weights. The `experts` target takes
a rank that is a multiple of 8 from 8 to 128 (`LoRAConfig`). A [PEFT adapter](#peft-adapters)
adapts the fused expert parameters (`target_parameters`) or names an expert's projections apart
(`experts.{e}.{gate,up,down}_proj`, as vLLM 0.31.0's and SGLang 0.5.22's adapter loaders read
them; served only). Its config check refuses,
by field, dense MLP layers (`mlp_only_layers`, `decoder_sparse_step`), non-SiLU experts,
sliding windows, the dual-chunk attention of Qwen's `config_1m.json` and logit
soft-capping (`models/qwen3_moe/config.py`); inference reads nothing from the
training-only `attention_dropout` and `output_router_logits`, and training refuses a
non-zero dropout and the router's auxiliary loss. It is registered but unqualified
until it has its own hardware, correctness, and performance receipts.

Dense Qwen3.5 models resolve from image-text
(`Qwen3_5ForConditionalGeneration`) and text-only (`Qwen3_5ForCausalLM`) checkpoints,
gated against Transformers as Qwen3.5-0.8B, 2B and 4B on an RTX 5090, Qwen3.5-9B on one
RTX PRO 6000 and Qwen3.8-27B on two (the truth over both). An image-text checkpoint loads its frozen
vision tower, which every tensor-parallel rank holds whole; the
multi-token-prediction head is not loaded. The model declares LoRA, training, and
text prefix reuse. LoRA targets `attention` (the full-attention
projections), `linear_attention` (the Gated DeltaNet projections), and `mlp`; the
vision tower is never adapted, and a [PEFT adapter](#peft-adapters) loads at launch. It runs
[tensor-parallel](tensor-parallel.md) inference, LoRA and training when the size divides its
Gated DeltaNet key heads. Its config check also rejects what the model would run
differently from transformers, such as RoPE other than interleaved default M-RoPE or
head geometry other than the published one.

Qwen3.5-MoE models resolve from image-text (`Qwen3_5MoeForConditionalGeneration`) and
text-only (`Qwen3_5MoeForCausalLM`) checkpoints, such as Qwen3.5-35B-A3B. The engine runs
them on one GPU, an image-text checkpoint with its frozen vision tower, which is dense
Qwen3.5's and takes images the same way (below); the multi-token-prediction head is not
loaded. Each decoder layer's sparse block routes a token to its top experts by renormalized
router probabilities and adds a sigmoid-gated shared expert; the routed experts load from
the published stacked tensors or from one tensor per expert. Inference runs each block
through cti-kernels' whole-block `MoEBlock`, a fused kernel for few rows and a composition
otherwise, except a block with expert adapters, which composes itself. A composed block routes
by cti-kernels' row-count-invariant router (`ctikernels.moe.route`), the arithmetic of the
whole-block op, so a token with the same hidden state takes the same experts on every path and
in every batch, Qwen3-MoE's blocks too; the paths still round the experts' sums apart, which can
move a later layer's near-tie route. Like dense Qwen3.5 it reuses
text prefixes on one GPU and serves as a DFlash target (z-lab/Qwen3.5-35B-A3B-DFlash).
Across the GPUs of one node it places whole experts per rank (expert parallel, the default) or every expert split by intermediate width (expert tensor parallel, `parallel.enable_expert_parallel: false`) ([tensor-parallel inference](tensor-parallel.md#moe)). LoRA targets `attention`,
`linear_attention` and `experts`: the last adapts every expert of each sparse block,
unmerged, with its own A and B on each routed expert's gate, up and down projection,
stacked by the rank's experts, and a LoRA on each shared-expert projection; the router and
the shared-expert gate stay frozen. Its rank is a multiple of 8 from 8 to 128. The factors
are large: on the 35B-A3B, 79M parameters per unit of rank (1.3 GB in BF16 at rank 8),
whose FP32 training state stays on the GPU across switches, so training needs a lower
`memory.inference_ratio` than mixer adapters do. Serving adds the routed-expert deltas inside
cti-kernels' fused experts (`fused_experts_lora`), into the grouped GEMMs' intermediate
caches, on BF16 and FP8 expert stacks alike. Text LoRA
training, per-expert adapters included, runs on one GPU or tensor-parallel: each sparse
block trains through the same fused op, whose backward gives the input, routing-weight and
factor gradients.

Offline sampling (`LLM.generate` and `LLM.sample`) accepts image prompts as the
`ProcessedPrompt` that `llm.prompt_encoder().chat(messages)` builds from a chat with image
content parts; images arrive as base64 `data:` URLs or PIL images. Every prompt, text and
token ids included, passes the model's prompt encoder, which the engine builds with the
first sampling command: an image placeholder without its image, or a processed prompt whose
placeholders do not match its images, is rejected for that request alone when serving,
and fails an offline command the same way as an over-long prompt. A `model.max_seq_len`
below the smallest image still serves text and rejects each image prompt; an image-text
checkpoint without `preprocessor_config.json` fails to load its prompt encoder, naming the
file. The tower's output rows stay cached by image content, least recently used first out,
in rows that come out of the KV cache and go with it to training: the same image in a later
prompt, or twice in one step, is encoded once.
A prefill chunk may end inside an image: the first step that reaches an image not cached
encodes it whole, with the step's other images, in one call of the vision tower, and
each chunk loads its own tokens' rows; the image stays cached until its last chunk is
planned. A step whose images' rows would not fit beside those of the split images ends
its chunk where the next image starts. A cached prefix may also end inside an image.
An image may fill as many tokens as the checkpoint's image preprocessing allows
(`longest_edge` pixels over (patch × merge)², 16384 for Qwen3.5), within `model.max_seq_len`;
larger images are scaled down to it. The encoder rows hold
max(`scheduling.max_batch_tokens`, that image), as vLLM sizes its encoder budget. The runner
measures the vision tower's activation peak encoding that many rows, the largest image among
them, in one call; the tower ends before the text model
starts, so the peak shares the peak reserve with the text forward's, and only what the
reserve cannot hold comes out of the KV cache. Decode steps stay text-only, on the
decode CUDA graphs. Training takes text only. With `model.text_only` an image-text
checkpoint loads as its text-only export: the vision tower is neither built nor loaded
(0.85 GiB of Qwen3.5-9B's weights), and a prompt with images is rejected as for a text
checkpoint.

Online, `/v1/chat/completions` takes OpenAI content parts: `text` in any message, and
`image_url` (`{url, detail}`; `detail` is validated and ignored) in user messages; other
part types are refused. A URL is a base64 `data:image/...` URL, an `http(s)` URL, or, only
under `server.allowed_local_media_path`, a `file://` URL; `server.allowed_media_domains`
restricts the hosts, redirects included. The API server fetches a request's images at
once, up to 256 MB each, and the tokenizer workers decode them with the offline encoder,
so an image gives the same tokens either way. A fetch follows vLLM's retries:
`server.media_fetch_max_retries` attempts (default 3) on timeouts, connection errors and
5xx, the timeout growing from 5 s fourfold per attempt with a 1 s, 4 s, ... pause between
them; a client that disconnects cancels them. More images than a request may carry are
refused before any fetch. Errors are OpenAI error objects with `param: image_url`, naming
only the URL's host: 400 for a malformed URL or a scheme, host or path not admitted (and,
from the tokenizer, an undecodable image), 422 for another 4xx or an image too large, 502
for 429 or when every attempt failed on a 5xx or connection error, and 504 for 408 or
when the last attempt timed out.

Every norm of the residual stream runs on cti-kernels' residual-stream kernel, in
inference and training alike. Each add is in FP32 and stores the sum in the stream's
dtype. Inference normalizes the FP32 sum before that rounding, as FlashInfer's fused
add+norm does; training normalizes the stored sum, which its backward and
activation-checkpoint recompute rebuild exactly. The residual stream between decoder layers
is in the model's dtype, so with BF16 a training forward differs from the inference forward
by that rounding at every norm after an add.

Training covers the whole SFT and GRPO contract in padded and packed layouts, with
activation checkpointing, and updates only the adapters; base weights stay frozen. It
runs on one GPU or tensor-parallel. Training inputs are text: an image-text checkpoint's image and video
placeholder tokens are rejected in SFT data and GRPO prompts until image training lands.
Like Qwen3-MoE, this profile is registered but unqualified until it has its own
hardware, correctness, and performance receipts.

Qwen3.5 residency tests enforce public-session median switch-latency bounds for both
checkpoint modes: entry/exit limits are 50/10 ms on RTX 5090 and 150/160 ms on
RTX PRO 6000 Blackwell Server Edition. Other devices report latency as explicitly
ungated; gate status, measured medians, and bounds are reported in the test output.
Residency accounting for fla-core temporaries retained by captured decode graphs
remains part of qualification.

## Chunked Prefill

Chunked Prefill, a technique introduced by [Sarathi-Serve](https://arxiv.org/abs/2403.02310), is enabled by default. This feature splits long prompts across inference submissions, reducing peak memory usage and preventing Out-Of-Memory (OOM) errors in long-context serving. The aggregate query-token budget shared by prompt extensions and decode is `scheduling.max_batch_tokens` (`--scheduling.max-batch-tokens n`). Within it, prompt extensions take at most `scheduling.max_prompt_tokens` per step, after the decodes or rounds: on an online server by default, by the device's total memory as SGLang's `chunked_prefill_size` (2048 below 20 GiB, 4096 below 60 GiB, 8192 below 160 GiB, else 16384, within `max_batch_tokens`), and `max_batch_tokens` offline. Setting `n` too small (for example, 128) may significantly degrade performance.

## Attention Backends

The V0 environment supports FlashInfer inference and native FlashAttention
variable-length packed SFT. The inference resolver also contains `fa` and `trtllm`
providers; availability depends on the installed provider, GPU architecture, model,
and execution mode. Each backend also lists the head dimensions that lack an accepted
operation test on it: resolution never selects a backend automatically for a listed
head dimension and rejects an explicit request for one. `fa` and `trtllm` list head
dimension 256.

The engine selects the backend for the device and model. Backend, page size and graph
sizes are internal tuning (`cti.config.EngineTuning`, which tests and benchmarks
pass from Python), not public options. A tuning may name a prefill and a decode backend
(`"fa,fi"`); some backends override the page size, for example `trtllm` supports 16,
32 and 64.

## CUDA Graph

To minimize CPU launch overhead, cti captures and replays CUDA graphs, by default.
Every kind follows one rule: a batch replays the smallest captured size that holds it,
padded with dummy rows (tokens of a breakable graph, done requests of a decode graph or a
DFlash round; none on an exact fit), and only a batch past the largest size runs eagerly.
Which sizes are captured trades speed against capture time and memory, never correctness.

Decode graphs run a whole forward of uniform decodes. By default every batch size up to 4
is captured, then multiples of 8 up to the scheduling bounds and free memory (internal
tuning may set them, `()` disabling decode graphs). DFlash rounds capture the same sizes.

Prompt steps and batches of prompts with decodes, without images, replay a breakable
graph: captured segments of row-wise work over a row bucket, around breaks that run eagerly
on the real rows. The breaks are attention (with its prologue and K/V write), the Gated
DeltaNet core and the MoE block, so padding rows never reach the KV, recurrent state or
routing. By default buckets run every 4 rows up to 32, every 16 up to 256 and every 32 up
to 512, so padding stays small; the internal tuning's `graph_row_buckets` overrides them,
and `()` captures none. Breakable graphs are captured before the KV is
sized and share one pool with decode and DFlash graphs, which training's KV suspension
releases and resumption restores without capturing again.

Vision graphs, opt-in through the internal tuning's `vision_graphs`, replay the Qwen3.5
vision tower's blocks and merger at token budgets: the smallest image's tokens doubling
below `max_batch_tokens`, then `max_batch_tokens`. A step's images, sorted by tokens, pack
greedily into the smallest budget that holds them; the padding attends as one sequence of
its own, never with an image, and an image past the largest budget runs eagerly. Patch and
position embeddings stay eager. They capture before the KV is sized, into a pool of their
own, so the capture report shows their memory apart.

## Radix Cache

Adopting the original design from
[SGLang](https://github.com/sgl-project/sglang.git), cti implements a Radix Cache
to manage KV storage. It reuses cache state for shared prefixes. The default
`cache.prefix: auto` enables it for models that support prefix reuse and selects the
naive cache manager for the others; startup logs the choice. Explicitly requesting
`radix` for a model without prefix reuse fails at startup. DFlash changes neither.
Qwen3.5 text uses a joint attention/GDN cache: a hit restores FP32 recurrent state
and convolution history from the same prefix frontier. `cache.prefix: naive` selects the naive cache manager.
Greedy output may differ at a near tie between a prefix hit and a miss, which the default
precision tier allows; for run-to-run reproducibility use `cache.prefix: naive` or a fixed batch.

Singleton hits must bitwise match a naive-cache run (no prefix reuse) split at the same boundary, both teacher-forced with the same truth (full logits and logsumexp), when the seed's forward ends at that boundary; a snapshot exported inside a longer forward follows GEMMs over more rows, so its hit meets the default tier.
Concurrent and edge-budget hits retain the A/A mean, p99, decided-disagreement and greedy near-tie gates.
The single-point probability-weighted maximum is recorded only.
Greedy text is not guaranteed identical between a prefix-cache hit and a cold run: a hit
prefills only the uncached suffix, so its chunks have other shapes and their GEMMs and
attention reduce in another order, and a near tie can flip (seen at TP4 and TP8). Such a
continuation is judged by the near-tie rule of
[tensor-parallel.md](tensor-parallel.md#numerics), not by equality.

Qwen3.5 snapshots share the recurrent pool's allocations and residency regions. The
provisional default reserves 25% of
memory remaining after request state and copy workspace for snapshots; the rest
holds attention KV.
Naive caching and attention-only models allocate no snapshot rows.
With `scheduling.max_running_requests` unset, startup takes the most request slots, up to
256, whose recurrent rows and snapshots leave KV pages for one `max_seq_len` sequence, or
for as many bytes as those rows when fewer, and logs the choice. A set value whose rows do
not fit fails at startup with the bytes needed and available.

Snapshots lie on one grid, `A = lcm(page_size, 64)`. A prompt exports at two points: the
deepest multiple of `A` a hit may restore, and the admission's KV match rounded down to `A`,
where it leaves a cached prefix that holds no state. A Gated DeltaNet prefill call captures at
most once per sequence, after its most whole multiples of 64 tokens, as SGLang does, so every
prompt forward starts on the grid: a chunk short of the prompt's end ends on it (a prompt
whose budget reaches no grid point waits a step; `max_prompt_tokens` must reach `A`), and a
forward that would cross the KV-match point without capturing it stops there, at most one
extra forward per prompt, without ending the step's admission. The deepest point is exported
only where the final forward captures, so, as in SGLang, a prompt whose length is a multiple
of 64 leaves it unexported (with `A = 128`, about half of all prompts do) and exports its end
instead: no repeat restores there, but extensions do, and the KV it publishes shows a repeat
the branch to stop at, so repeats hit from the third. The prefill
exports states where cti-kernels reports `supports_state_capture` (every provider but
fla-core); elsewhere `auto` selects the naive cache for Qwen3.5 and an
explicit `radix` fails at startup. An export takes a free row, else the least recently used
unlocked snapshot's, else is skipped; nothing waits for a row. Rows pass to the radix when
their forward settles. Batch selection holds a waiting prompt back, for at most four plans,
while a forward planned before it or still unsettled exports a snapshot of a prefix they
share and restoring there would save at least 256 tokens of recompute. A hit restores the
deepest snapshot short of the final prompt token, whose logits are always recomputed. A
decode step reaching a grid point copies its request's state row into one row leased for the
request; each later crossing overwrites it, and the last write passes to the radix when the
request ends, unless that write's step was released. A DFlash round writes the same row
inside K2's commit, at the first grid point past its frontier that it commits; a round the
host discards voids a row it wrote. Lookup falls back to a shallower snapshot when recurrent state was evicted. Every return from training residency invalidates both caches
before admission resumes, including after real LoRA updates and no-op training sessions.

This implementation targets text and images, on one GPU or tensor-parallel. CPU reference tests cover exact state copies
and hit/cold logits; GPU numerical, eviction, residency and memory gates are written
but remain unqualified. Copies use the approved grouped Torch fallback with a
budgeted temporary.
The fused cross-layer copy awaits a public ctikernels API; native providers stay
outside the engine. The radix keys each image's placeholders by its content: eight
negative int32 words from sha256 of its pixels' digest and patch grid, repeated over the
span. Different images diverge within their first eight tokens, and only the same image
at the same offset shares a prefix. Key ids never reach the model, which reads 0 there.
TP qualification, session hints, declared shared prefixes and decode captures follow.

![radix](https://lmsys.org/images/blog/sglang/radix_attn.jpg)
*Illustration of Radix Attention from [LMSYS Blog](https://lmsys.org/blog/2024-01-17-sglang/).*

## Overlap Scheduling

To further reduce CPU overhead, cti employs overlap scheduling, a technique proposed in [NanoFlow](https://arxiv.org/abs/2408.12757). This approach overlaps the CPU scheduling overhead with GPU computation, improving overall system throughput.

![overlap](https://lmsys.org/images/blog/sglang_v0_4/scheduler.jpg)
*Illustration of Overlap Scheduling from [LMSYS Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/).*

## SFT and Switching

The typed Python API supports right-padded and native packed SFT over a fixed set of
model-declared non-base tensors while base weights remain frozen. Repeated
`forward_backward` calls accumulate gradients; `optim_step` updates the trained adapter
after normalizing by the total target-token count. Training runs on FP32 masters of the
selected tensors, which the training kernels cast at each use, so every gradient
accumulates in FP32; inference reads the adapter last published, and only
`session.publish()` copies the trained adapter there, as exactly one new model revision.
A recipe that samples or serves after each step publishes after each step. Training
layout and batching policy are explicit and independent: padded and packed layouts
support a fixed sample-count policy, while packed layout additionally supports a
dynamic total-token budget. One logical submission may contain multiple internal
microbatches without changing the explicit optimizer boundary. Layout, shape-profile
and target-token validation run on the CPU before a submission is acknowledged: a
batch they reject raises from `forward_backward` itself and leaves earlier submissions
and the accumulation window intact. Training
sessions are exclusive. Entry quiesces inference and suspends releasable KV storage;
exit invalidates prior logical KV state before storage reuse and inference resume,
without committing implicitly.

Eager activation checkpointing supports construction-bound `none`, `full` and
`full_offload` modes. `full` recomputes each decoder layer in backward, one layer per
checkpoint, from the first layer that holds a selected parameter. The frozen layers
before it run without autograd and are neither saved nor recomputed, so `full`
differentiates exactly what `none` does.
`full_offload` is `full` with each decoder layer split into two checkpoints, its token
mixer and then its MLP or sparse block, so a segment's backward holds no other segment's
saves, and with each checkpoint's input, except the last, held in pinned host memory
between forward and backward: a copy stream moves it out byte for byte, in the residual
stream's dtype, and fetches it back one segment ahead of backward, so a step trains bit
for bit as `full` does. The forward keeps at most two segment inputs on the device at a
time, besides the embedding output, which the model holds until its forward returns.
A dense MLP segment trains in tiles of 4096 rows: its checkpoint keeps only its input, and
backward recomputes and differentiates one tile at a time, so its backward holds one
tile's activations whatever the microbatch's length.
Training construction pins one arena of exactly
(2 x checkpointed layers - 1) x microbatch token bound x hidden size x element size bytes,
where the bound is `max_tokens_per_microbatch` for dynamic batching and
`microbatch_size` x `max_tokens_per_sample` for static batching. It fails before
allocating when the arenas of the process, the new one included, would exceed 70% of
the host's memory, or when the new arena exceeds the host's available memory.
A padded batch whose microbatches, padding included, exceed the bound is refused before
it runs. The arena stays pinned for the life of the training runner; switching to
inference neither copies nor releases it. `full_offload` needs a CUDA device; with the
default `training_pause_boundary: microbatch`, serving waits for one microbatch rather
than a whole logical batch.

The arena uses an anonymous mapping registered with CUDA to avoid the pinned allocator's
power-of-two rounding. Copies use reusable ready/done events per slot; each stream wait
captures the preceding record before the next transfer reuses the event. Device tensors
stay owned until the training stream waits for their copies, and abort synchronizes the
copy stream before releasing them. Prior art: verl's saved-tensor hooks
(`verl/utils/activation_offload.py` at `8718ca3`) and DeepSpeed's last-checkpoint retention
(`runtime/activation_checkpointing/offload_activations.py` at `a77aeb6`).

Eager execution is the only maintained V0 SFT/GRPO path; there is no training
compiler option or placeholder graph runtime. Exact-capacity partial-KV release and
fixed-shape training graph execution remain separately measured future work.

The physical foundation already has one TMS-owned region per KV layer and a
contiguous-tail suspension mechanism. Models with recurrent layers place one
recurrent-state region per such layer first in the same allocation, so the whole
inference state suspends, loses its contents and resumes together, and a tail
release frees KV layers first. This is not a production layerwise release policy: V0
safely suspends the complete inference state because eager training has no exact
transient-capacity owner. Selecting the minimum sufficient suffix remains deferred.

Training memory has two TMS regions, split by lifetime. The FP32 master gradients
live in one reusable region that suspends with the training window. Activations,
recomputation, backward temporaries and everything else a window allocates on its
training thread share one fresh allocator pool per window: backward runs on the
calling thread, so the step's peak is the larger of its forward and backward
high-water marks, not their sum. Leaving training frees that pool, after the worker
moves the gradients and evidence it keeps to the host, so a later window maps only
what its own steps need, never the largest step so far.

Startup plans training memory before it sizes the inference state
(`cti.training._memory_plan`): the FP32 master weights, the AdamW moments, the FP32
gradient accumulators and the optimizer step's temporaries follow from the selected
parameters, and a fixed SFT shape profile's staging buffers from the profile, each
packed as the caching allocator places it; the activation peak follows from the model's
account of its training ops at the largest microbatch the batching policy allows
(tokens, and sequences: `microbatch_size` or `max_sequences_per_microbatch`) under the
checkpointing mode, frozen layers' forwards included. A fresh allocator pool's segments
beside those activations are planned too: a small and a large segment, and a quarter of
the activations for fragmentation, measured up to a fifth. A padded batch whose
microbatches, padding included, exceed the token bound is refused before it runs. The
training stream's cuBLAS workspace is bound after the GEMM warm-up drops every stream's
and before the inference state is sized. A residency mode says which components stay on
the device while inference runs; today the master weights, moments and staging buffers
do, so the KV budget leaves them out. The optimizer allocates its moments at startup.
With a memory manager, training then leaves nothing else in the default allocator: each
microbatch and optimizer step allocates in the training window's pool, which is replaced
when the segments it cached for earlier shapes leave less free memory than one of them
maps in a fresh pool. Near the fit limit that means a fresh pool for every scope. A logical batch's loss
evidence is folded into a few scalars in the persistent region as its microbatches
complete and moves to the host when the batch does. Once everything inference keeps
through training exists, startup fails with `TrainingMemoryError`
(`cti.worker`) if what training keeps beside inference does not fit the
free memory, or if the allocator configuration changes how blocks are placed. If the
estimated peak does not fit that plus what releasing the KV frees within the peak budget
(`memory.peak_ratio`), startup logs one warning with the room, the estimate, the ratio and
the largest terms, and proceeds: the estimate is not a hard bound. A runtime without a
memory manager releases nothing, so its peak budget is only the free memory.

The memory manager refuses a resume whose suspended memory no longer fits the device's
free memory (on unified memory, GB10, the host's `MemAvailable`, as the memory plan reads
it; `ResidencyRestoreError`) before anything is mapped, so the process survives: the switch
is rejected and the current phase keeps the device. The check assumes no other process
allocates on the device meanwhile; the provider cannot fail a resume recoverably. A
rejected switch raises `SwitchRejectedError` and keeps the current phase: an entry keeps
inference, a close or abort keeps the session open on the device (an abort has already
discarded the accumulation), and an explicit serving pause leaves training running.
After any rejected switch to inference (a serving pause, explicit or automatic, a
close or an abort), held requests and every later one are rejected
(`RequestRejectedError`) and automatic pauses stop, until an optimizer step, a close, or
a switch that brings inference memory back; an explicit pause, close or abort may still
retry. A close or abort that takes the device back from serving, paused or not, is
refused with `SwitchRejectedError` when training memory cannot come back; a resume whose
training memory cannot come back fails the runtime with the `ResidencyRestoreError`.

## GRPO

The V0 GRPO path is reference-free (`beta = 0`) and supports right-padded and native
packed batches through the same batching planner and training lifecycle as SFT. A
grouped rollout preserves exact sampled token IDs, request-local seeds, selected-token
sampling log-probabilities, and the producing model revision. Sampling
log-probabilities are generation evidence, while the atomic same-revision update uses
detached teacher-forced policy log-probabilities as its old-policy denominator. The
policy update uses the same model and revision owner as serving. The public path has
graph-on single-GPU lifecycle evidence. Reference KL, moving-reference policies, and
training execution optimization are later qualification slices.
