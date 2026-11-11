# Configuration

One tree of options configures every entry point. Python passes it directly:

```python
from cti.config import EngineOptions, ModelOptions, SchedulingOptions, ServerOptions
from cti.llm import LLM
from cti.server import OnlineServer

options = EngineOptions(
    model=ModelOptions(path="Qwen/Qwen3-0.6B"),
    scheduling=SchedulingOptions(max_running_requests=32),
)
llm = LLM(options)
server = OnlineServer(options, ServerOptions(port=8080))
```

A launch file spells the same tree in YAML: `LLM.from_yaml(path)` reads it offline and
`OnlineServer.from_yaml(path)` serves it. The CLI reads one with `--config` and takes one
flag per option, the option's path with `_` written `-`:

```bash
python -m cti --config cookbook/configs/inference.yaml --server.port 8080
python -m cti --model.path Qwen/Qwen3-0.6B --scheduling.max-running-requests 32
```

A flag overrides the file, and the file overrides the default. A flag's value is read as
YAML (`--lora.targets "[attention, mlp]"`, `--model.weights "{kind: random_weights}"`),
except that an option that only ever holds a string takes the text as given. Option names
have no aliases.

## Options

| Option | Default | Meaning |
|---|---|---|
| `model.path` | required | Local directory or repository id on `model.source` |
| `model.source` | `huggingface` | The hub a repository id (target and draft) downloads from: `huggingface` or `modelscope` (needs the `modelscope` extra). Its configs and tokenizer download at resolve time, at one revision; its weights after the checkpoint is validated, reporting progress |
| `model.dtype` | `bfloat16` | `bfloat16` (the only execution dtype), or `auto`, which also checks that the checkpoint stores BF16, FP16 or FP32 weights; they load converted to BF16 |
| `model.weights` | `{kind: checkpoint}` | `{kind: random_weights, seed: 0}` draws normal weights instead of loading them |
| `model.max_seq_len` | `null` | Caps the sequence length below the model's maximum position |
| `model.initialization_seed` | `42` | Seeds trainable parts such as LoRA factors |
| `model.text_only` | `false` | Loads an image-text checkpoint as text only: no vision tower is built or loaded, and requests with images are refused |
| `parallel.num_ranks` | `1` | GPUs one engine spans ([tensor parallelism](tensor-parallel.md)) |
| `parallel.num_nodes` | `1` | Hosts the ranks span, an equal share on each ([two hosts](tensor-parallel.md#two-hosts)) |
| `parallel.node_rank` | `0` | This host's node; node 0 serves, the others join it |
| `parallel.dist_init_addr` | `null` | Node 0's `host:port` the other nodes join; required with `num_nodes > 1` |
| `parallel.enable_expert_parallel` | `true` | A MoE model's routed experts: whole experts per rank, or (`false`) every expert's width split ([MoE](tensor-parallel.md#moe)) |
| `memory.inference_ratio` | `null` | Fraction of boot-time free GPU memory for weights and KV cache; `null` is 0.9 on a discrete GPU and 0.6 on unified memory (GB10), which the host shares |
| `memory.peak_ratio` | `null` | Fraction for the peak-planning budget; `null` is the inference ratio |
| `memory.recurrent_to_kv_ratio` | `0.9` | Recurrent-state bytes per full-attention KV byte (SGLang's `mamba_full_memory_ratio`): recurrent rows, running and prefix snapshots, take ratio / (1 + ratio) of the budget left after weights |
| `scheduling.max_running_requests` | `null` | Requests one step runs; `null` takes as many as the memory budget holds, at most 256 ([recurrent state](features.md#radix-cache)) |
| `scheduling.max_batch_tokens` | `8192` | Query tokens one step carries |
| `scheduling.max_prompt_tokens` | `null` | Prompt tokens one step carries after its rounds or decodes; `null` is, on an online server, SGLang's tier for the device's total memory (2048 below 20 GiB, 4096 below 60 GiB, 8192 below 160 GiB, else 16384; within `max_batch_tokens`) and `max_batch_tokens` offline; with recurrent prefix snapshots, at least their grid `lcm(page_size, 64)` |
| `cache.prefix` | `auto` | Prefix cache: `naive`, `radix`, or `auto` (radix where the model reuses prefixes; see [radix cache](features.md#radix-cache)) |
| `speculative.model_path` | no section | A DFlash draft checkpoint ([DFlash](dflash.md)) |
| `speculative.num_speculative_tokens` | the draft's trained block − 1 | Draft tokens per round, the bonus token aside: 1–16, whatever block the draft was trained on |
| `lora.targets` | required in `lora` without `adapter` | LoRA targets the model declares (`attention`, `linear_attention`, `mlp`) |
| `lora.rank` | `8` | LoRA rank |
| `lora.alpha` | `16.0` | LoRA scale numerator |
| `lora.seed` | `0` | Seed of the adapter's initial factors, drawn per state-dict path |
| `lora.adapter` | none | A PEFT adapter directory loaded at launch; its targets, rank, alpha and expert layout replace these ([features](features.md#peft-adapters)) |
| `lora.expert_layout` | `shared_a` | How the `experts` target adapts each routed expert's gate and up: `shared_a` (one A for both, PEFT's fused `gate_up_proj` adapter; the only layout that trains) or `per_projection` (an A for each, as a Transformers 4 per-expert adapter imports; served only). An imported adapter sets it |
| `training.trainable_plan.selected_names` | `null` | Parameter names to train; `null` (`trainable_plan: {}`) trains every one the model declares. A MoE layer's routed-expert LoRA factors train all together or not at all |
| `training.optimizer.lr` | `1e-4` | AdamW learning rate |
| `training.optimizer.betas` | `[0.9, 0.999]` | AdamW betas |
| `training.optimizer.eps` | `1e-8` | AdamW epsilon |
| `training.optimizer.weight_decay` | `0.0` | AdamW weight decay |
| `training.optimizer.max_grad_norm` | `1.0` | Gradient-norm clip; `null` disables it |
| `training.batching.layout` | required | `padded` or `packed` |
| `training.batching.policy` | required | `{kind: static_batching, microbatch_size, max_tokens_per_sample}` or, packed only, `{kind: dynamic_batching, max_tokens_per_microbatch, max_sequences_per_microbatch}` (the last defaults to 256) |
| `training.activation_checkpointing` | `none` | `none`, `full`, or `full_offload` ([features](features.md#sft-and-switching)) |
| `training.training_pause_boundary` | `microbatch` | `microbatch` or `logical_batch` |
| `training.training_state_residency` | by device | `resident`, `offload_optimizer` or `offload_all`; by default `offload_optimizer` on a discrete GPU and `resident` on unified memory ([features](features.md#sft-and-switching)) |
| `training.sft_profile.sequence_lengths` | required in `sft_profile` | SFT shapes to warm up |
| `training.sft_profile.padded_width` | `null` | Padded width of the warm-up batch |
| `training.sft_profile.num_target_tokens` | `null` | Target tokens of the warm-up batch |
| `server.host` | `127.0.0.1` | Online server address |
| `server.port` | `1919` | Online server port |
| `server.num_tokenizers` | `1` | Tokenizer processes (at least 1); each preprocesses a call's prompts on a thread pool |
| `server.allowed_media_domains` | `[]` | Hosts http(s) image URLs may name, redirects included; empty admits any |
| `server.allowed_local_media_path` | `""` | Existing directory `file://` image URLs may read under; empty refuses them |
| `server.media_fetch_max_retries` | `3` | Attempts per image URL on timeouts, connection errors and 5xx (`0` is one) |
| `server.tool_call_parser` | `auto` | Tool-call format: `auto` (from the chat template), `hermes` or `qwen3_coder` |

Sections `speculative`, `lora`, `training` and `training.sft_profile` exist only when
given. `training` requires `trainable_plan`, `optimizer` (`{}` keeps the defaults) and
`batching`. `LLM.from_yaml` checks the `server` section and ignores it, so one file serves
both offline and online use. `OnlineServer.from_yaml` runs local training and exposes it
as `server.training_session()`; the CLI refuses `training`, as there is no HTTP
training API. Resuming from a training checkpoint is a Python argument:
`LLM(options, resume_from_checkpoint=path)` or `LLM.from_yaml(path, resume_from_checkpoint=...)`.

YAML is read strictly: duplicate keys, unknown keys and Python tags are errors. There are
no includes or environment-variable interpolation. Relative paths follow the process
working directory, not the file's.

## One local SFT update

With [`cookbook/configs/training.yaml`](../cookbook/configs/training.yaml), the caller
provides a CPU batch and controls the optimizer boundary:

```python
import torch

from cti.llm import LLM
from cti.training import SFT, PaddedTokens, TrainingBatch

def main():
    tokens = PaddedTokens(
        token_ids=torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.int64),
        lengths=torch.tensor([3, 2], dtype=torch.int64),
        pad_token_id=0,
    )
    # Target positions: position p of a row predicts its token p + 1.
    targets = torch.tensor([[False, True], [True, False]], dtype=torch.bool)
    batch = TrainingBatch(tokens, targets, SFT())
    llm = LLM.from_yaml("cookbook/configs/training.yaml")
    try:
        with llm.training_session() as session:
            loss = session.forward_backward(batch).result()
            step = session.optim_step().result()
            # Inference serves the adapter from here on.
            revision = session.publish()
        print(loss.loss_sum, step.step, revision)
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
```

For packed SFT, replace `training.batching` with:

```yaml
layout: packed
policy:
  kind: dynamic_batching
  max_tokens_per_microbatch: 1024
```

and construct `TrainingBatch.sft(data, BatchLayout.PACKED)` instead; [`cookbook/sft.py`](../cookbook/sft.py)
shows both layouts. Launch files do not describe user data, target masks, rewards, the
objective, loop structure, accumulation count, or optimizer-step timing; those stay in
Python.
