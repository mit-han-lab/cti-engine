"""Qwen3.5's Transformers reference beyond the model-neutral truth (``tests/support/hf_truth.py``):
image prompts processed as Transformers processes them, the scored continuations of the default
accuracy tier and the placement a reference cache entry records."""

from __future__ import annotations

import sys

from tests.support.hf_truth import (
    BLOCKED_PACKAGES,
    check_ieee_fp32,
    dtype_name,
    kernel_bindings,
    loaded_blocked_modules,
    parameter_dtypes,
    reference_architecture,
)

# The Gated DeltaNet kernels modeling_qwen3_5 binds at import, which must stay torch's.
_KERNEL_BOUND = reference_architecture("Qwen3_5ForConditionalGeneration").kernel_bound
# The FP32 first-row self-check: a continuation's first row against its prompt's last row, one
# position from two forwards of different lengths, whose GEMM and attention reductions are
# ordered by shape, so the rows agree to rounding only. Qwen3-8B, RTX PRO 6000, 19-1019-token
# prompts: up to 1.34e-4 nats, every shape bit-exact on repeat; TF32 matmuls give 0.014-0.020,
# so this bound passes the rounding and still fails TF32 or a misindexed row.
FP32_FIRST_ROW_MAX_NATS = 1e-3


def reference_placement(model):
    """Actual parameter storage and offload execution; compare with a fixed cache recipe."""
    import platform

    import torch
    from accelerate.hooks import SequentialHook

    execution = {}
    for name, module in model.named_modules():
        hook = getattr(module, "_hf_hook", None)
        for index, part in enumerate(hook.hooks if isinstance(hook, SequentialHook) else (hook,)):
            if getattr(part, "execution_device", None) is not None:
                execution[f"{name}:{index}"] = str(part.execution_device)
    parameters = {n: str(p.device) for n, p in model.named_parameters()}
    devices = set(parameters.values()) | set(execution.values())
    architecture = {
        d: [torch.cuda.get_device_name(d), list(torch.cuda.get_device_capability(d))]
        if d.startswith("cuda")
        else platform.machine()
        for d in devices
        if d != "meta"
    }
    parameters.update(
        (f"device_map:{n}", str(d)) for n, d in getattr(model, "hf_device_map", {}).items()
    )
    return {"parameters": parameters, "execution": execution, "architecture": architecture}


def prepare_image_inputs(checkpoint, messages, engine_prompt, *, max_image_tokens=None):
    """Run before cache lookup: HF-native FP32 pixels, independent IDs/grid parity."""
    import torch
    from transformers import AutoTokenizer, Qwen2VLImageProcessorPil, Qwen3VLProcessor

    class ImageOnlyProcessor(Qwen3VLProcessor):
        def check_argument_for_proper_class(self, name, argument):
            # Images never use a video processor (CPU installs can lack torchvision).
            if name == "video_processor":
                return type(argument)
            return super().check_argument_for_proper_class(name, argument)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    processor = ImageOnlyProcessor(
        tokenizer=tokenizer,
        chat_template=tokenizer.chat_template,
        image_processor=Qwen2VLImageProcessorPil.from_pretrained(checkpoint, local_files_only=True),
    )
    images = [
        part["image"]
        for message in messages
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image"
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    options = {}
    if max_image_tokens is not None:
        image_processor = processor.image_processor
        pixels_per_token = (image_processor.patch_size * image_processor.merge_size) ** 2
        options["size"] = {
            "shortest_edge": image_processor.size.shortest_edge,
            "longest_edge": min(
                image_processor.size.longest_edge, max_image_tokens * pixels_per_token
            ),
        }
    inputs = processor(text=[text], images=images, return_tensors="pt", **options)
    pixels, engine = inputs["pixel_values"], engine_prompt.images
    if engine is None or (pixels.dtype, engine.pixel_values.dtype) != (
        torch.float32,
        torch.uint8,
    ):
        raise ValueError("expected native HF FP32 and engine uint8 pixels")
    # HF's resized uint8 levels before its rescale and normalization (which the engine's
    # vision tower applies), one patch per row, repeated over the patch's frames.
    levels = processor.image_processor(
        images=images, do_rescale=False, do_normalize=False, return_tensors="pt", **options
    )["pixel_values"]
    channels = engine.pixel_values.view(len(levels), 3, 1, -1)
    frames = channels.expand(-1, -1, processor.image_processor.temporal_patch_size, -1)
    expected = torch.tensor([span.grid_thw for span in engine.spans])
    if (
        not torch.equal(inputs["input_ids"][0], engine_prompt.input_ids)
        or not torch.equal(inputs["image_grid_thw"], expected)
        or not torch.equal(levels, frames.reshape(levels.shape).float())
        or not torch.isfinite(pixels).all()
    ):
        raise ValueError("HF/engine processor parity failed")
    return {
        **dict(inputs),
        "originals": [(im.mode, im.size, im.tobytes()) for im in images],
        "engine_positions": engine_prompt.rope_layout.positions(0, len(engine_prompt.input_ids)),
    }


def score_continuations(model, requests, *, device, attention="eager"):
    """Score prompt_ids + continuation_ids in FP32, with optional native images, and self-check
    indexing."""
    import torch
    from transformers.integrations import hub_kernels
    from transformers.models.qwen3_5 import modeling_qwen3_5

    if not requests:
        raise ValueError("nonempty requests required")
    config = model.config.get_text_config()
    if attention not in ("eager", "sdpa") or config._attn_implementation != attention:
        raise ValueError("reference attention mismatch")
    check_ieee_fp32()
    if (
        hub_kernels._kernels_enabled
        or loaded_blocked_modules()
        or any(sys.modules.get(p, ...) is not None for p in BLOCKED_PACKAGES)
        or set(kernel_bindings(modeling_qwen3_5, _KERNEL_BOUND).values()) != {"torch"}
    ):
        raise RuntimeError("reference must use isolated torch kernels")
    dtypes = parameter_dtypes(model)
    if not dtypes or any(names != [dtype_name(torch.float32)] for names in dtypes.values()):
        raise RuntimeError(f"reference parameter dtype mismatch: {dtypes}")

    def forward(request, prompt_only=False):
        prompt, tokens = request["prompt_ids"], request["continuation_ids"]
        if (
            not prompt
            or not tokens
            or any(type(t) is not int or not 0 <= t < config.vocab_size for t in [*prompt, *tokens])
        ):
            raise ValueError("invalid prompt/continuation token IDs")
        sequence = list(prompt) + ([] if prompt_only else list(tokens))
        ids = torch.tensor([sequence], dtype=torch.int64, device=device)
        inputs = {}
        if (images := request.get("images")) is not None:
            if images["pixel_values"].dtype != torch.float32:
                raise ValueError("reference images must retain native FP32 pixels")
            if images["input_ids"][0].tolist() != list(prompt):
                raise ValueError("image prompt IDs disagree")
            inputs = {k: images[k].to(device) for k in ("pixel_values", "image_grid_thw")}
            inputs["mm_token_type_ids"] = (ids == model.config.image_token_id).int()
            positions, _ = model.model.get_rope_index(
                ids, **{k: v for k, v in inputs.items() if k != "pixel_values"}
            )
            if not torch.equal(positions[:, 0, : len(prompt)].cpu(), images["engine_positions"]):
                raise ValueError("HF/engine M-RoPE parity failed")
        count = 1 if prompt_only else len(tokens) + 1
        output = model(input_ids=ids, use_cache=False, logits_to_keep=count, **inputs).logits
        if output.shape != (1, count, config.vocab_size) or output.dtype != torch.float32:
            raise RuntimeError("reference output dtype/row indexing mismatch")
        rows = output[0, : 1 if prompt_only else len(tokens)].detach().to("cpu", copy=True)
        if not torch.isfinite(rows).all():
            raise RuntimeError("nonfinite reference rows")
        return rows

    with torch.inference_mode():
        rows = [forward(request) for request in requests]
        shortest = min(range(len(requests)), key=lambda i: len(requests[i]["prompt_ids"]))
        first, alone = rows[shortest][0], forward(requests[shortest], True)[0]
        selected = requests[shortest]["continuation_ids"][0]
        ids = list({*first.topk(min(8, len(first))).indices.tolist(), selected})
        error = float((first.log_softmax(-1)[ids] - alone.log_softmax(-1)[ids]).abs().max())
    if not error <= FP32_FIRST_ROW_MAX_NATS:
        raise RuntimeError(f"reference first-row self-check failed: {error} nats")
    check_ieee_fp32()
    proof = dict.fromkeys(("torch_paths", "ieee", "indexing"), True)
    proof.update(dtype=str(torch.float32), max_nats=error, shortest=shortest)
    return rows, proof


__all__ = ["prepare_image_inputs", "reference_placement", "score_continuations"]
