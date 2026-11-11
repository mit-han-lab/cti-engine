# cti cookbook

These recipes compose maintained public interfaces into complete workflows. They
live at the repository root and are not included in the release wheel.

## YAML configuration

[`configs/inference.yaml`](configs/inference.yaml) is a portable serving starting
point, and [`configs/training.yaml`](configs/training.yaml) adds LoRA and padded SFT
construction policy. Load them with `OnlineServer.from_yaml(path)` or
`LLM.from_yaml(path)` as appropriate. The CLI serves inference-only YAML:

```bash
python -m cti --config cookbook/configs/inference.yaml
```

Explicit CLI flags override YAML values. Configuration constructs the runtime;
the caller still owns data, target masks, objectives, rewards, accumulation, and the
training loop. See [`docs/configuration.md`](../docs/configuration.md) for the schema
and a complete local update.

## Training workflows

```bash
python cookbook/sft.py --model Qwen/Qwen3-0.6B --layout padded
python cookbook/sft.py --model Qwen/Qwen3-0.6B --layout packed
python cookbook/grpo.py --model Qwen/Qwen3-0.6B --layout padded --max-new-tokens 16
python cookbook/grpo.py --model Qwen/Qwen3-0.6B --layout packed --max-new-tokens 16
```

Both recipes use explicit accumulation boundaries and checkpoint published model
revisions. The SFT recipe constructs caller-owned target masks; the GRPO recipe
keeps reward and zero-variation policy outside the engine. They demonstrate
orchestration, not a dataset framework, Trainer, or rollout manager. V0 GRPO is
reference-free and does not promise model-level batch-invariant sampling.

## Online serving with local training

Run OpenAI-compatible serving and a local typed SFT update on the same engine and
model revision:

```bash
python cookbook/online_training.py
```

`OnlineServer` owns the HTTP frontend and its engine; `server.training_session()`
opens a local `TrainingSession` on that engine. The recipe does not add a
training route or a second model owner. The operator calls `pause_for_serving()` and
`resume_training()` on that session; the HTTP frontend stays alive throughout. The
pause preserves accumulated gradients and neither transition commits an optimizer
step implicitly. Async applications may await `pause_for_serving_async()` and
`resume_training_async()`; these are event-loop-safe wrappers over the same control
path, not a second transition implementation.

## Terminal chat

Start the OpenAI-compatible server, then run the client in another terminal:

```bash
python -m cti --model.path Qwen/Qwen3-0.6B
python cookbook/chat.py
```

The client owns only local conversation history and presentation. It discovers the
model automatically when the server advertises exactly one, streams responses, and
supports `/reset` and `/exit`. It never launches, imports, or shuts down the engine.
Use `--base-url` and `--model` for another endpoint or a multi-model server.

Cookbook files demonstrate application composition. They are not packaged client
APIs or compatibility promises.
