"""The truth child's ``--mode stream`` (``tests/support/hf_truth.py``): a causal LM's teacher-forced
forward with one decoder layer resident at a time, every prompt through a layer before the next
one loads, and the query-blocked eager attention a long prompt needs."""

from __future__ import annotations

import argparse
import functools
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tests.support.hf_truth import reduce_rows, reference_architecture, run_dump

if TYPE_CHECKING:
    import torch

# Rows the streamed LM head projects at once: one call up to this many, as the whole model's
# forward does; a longer sequence in blocks of it, its logits never held whole.
_STREAMED_HEAD_ROWS = 4096
# The query-blocked eager attention of --attention-chunk, as Transformers' attention and mask
# interfaces know it.
CHUNKED_ATTENTION = "cti_query_chunked_eager"


class LayerStream:
    """A causal LM's teacher-forced forward, one decoder layer resident at a time.

    The model is built on the meta device by its own Transformers class; each module's weights
    load from the checkpoint through Transformers' own weight conversions just before the
    module runs every sequence, and are dropped after. Each sequence computes what the whole
    model's forward over it computes, step for step: the same embedding, rotary table, causal
    mask, decoder layers, final norm and LM head, so on the same device a sequence of up to
    ``_STREAMED_HEAD_ROWS`` tokens gets the same bits. A longer one runs the LM head over blocks
    of that many rows, and with ``attention_chunk`` attention runs Transformers' eager attention
    over blocks of that many query rows against the keys up to the block's end, with the eager
    causal mask's rows of the block: the same arithmetic over other GEMM shapes, so their
    roundings may differ from the unblocked forward's.
    """

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
        attention_chunk: int | None = None,
    ) -> None:
        import torch
        import transformers
        from safetensors import safe_open
        from transformers.conversion_mapping import get_model_conversion_mapping

        config = transformers.AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        name = (config.architectures or [None])[0]
        if not reference_architecture(name).streams:
            raise ValueError(f"layer streaming runs attention-only decoders, not {name}")
        if getattr(config, "sliding_window", None) is not None:
            raise ValueError("layer streaming runs full causal attention, not a sliding window")
        if config.tie_word_embeddings:
            raise ValueError("layer streaming needs an untied LM head")
        attention = "eager"
        if attention_chunk is not None:
            if attention_chunk < 1:
                raise ValueError(f"attention chunk {attention_chunk} must be positive")
            _register_chunked_attention(attention_chunk)
            attention = CHUNKED_ATTENTION
        with torch.device("meta"):
            model = transformers.AutoModelForCausalLM.from_config(
                config,
                dtype=dtype,
                attn_implementation=attention,
                experts_implementation="eager",
            )
        if type(model).__name__ != name:
            raise TypeError(f"{checkpoint} built {type(model).__name__}, not {name}")
        self._model = model.eval()
        # The device as tensors report it: "cuda" is the current CUDA device, "cpu:0" the CPU.
        self._device = torch.empty(0, device=device).device
        self._dtype = dtype
        self._weight_mapping = get_model_conversion_mapping(model)
        self._files: dict[str, Path] = {}
        for file in sorted(checkpoint.glob("*.safetensors")):
            with safe_open(file, framework="pt", device="cpu") as handle:
                self._files.update(dict.fromkeys(handle.keys(), file))
        self.loaded_bytes = 0

    @property
    def vocab_size(self) -> int:
        return self._model.config.vocab_size

    def forward(self, sequences: Sequence[Sequence[int]]) -> list[torch.Tensor]:
        """Each sequence's final-norm hidden states ``[1, T, hidden]``, every layer streamed
        once for all of them."""

        import torch
        from transformers.masking_utils import create_causal_mask

        decoder = self._model.model
        self._load(decoder.embed_tokens, "model.embed_tokens.")
        hidden = []
        for sequence in sequences:
            input_ids = torch.tensor([list(sequence)], dtype=torch.int64, device=self._device)
            hidden.append(decoder.embed_tokens(input_ids))
        self._drop(decoder.embed_tokens)
        # Transformers computes the rotary table on the CPU and copies it to the device.
        rotary = type(decoder.rotary_emb)(config=decoder.config).to(self._device)
        arguments = []
        for states in hidden:
            positions = torch.arange(states.shape[1], device=self._device).unsqueeze(0)
            mask = create_causal_mask(
                config=decoder.config,
                inputs_embeds=states,
                attention_mask=None,
                past_key_values=None,
                position_ids=positions,
            )
            arguments.append((mask, positions, rotary(states, position_ids=positions)))
        for index, layer in enumerate(decoder.layers):
            self._load(layer, f"model.layers.{index}.")
            for row, (mask, positions, embeddings) in enumerate(arguments):
                hidden[row] = layer(
                    hidden[row],
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=embeddings,
                )
            self._drop(layer)
        self._load(decoder.norm, "model.norm.")
        normalized = [decoder.norm(states) for states in hidden]
        self._drop(decoder.norm)
        return normalized

    def logits(self, hidden: torch.Tensor) -> Iterator[torch.Tensor]:
        """The LM head's logits ``[rows, vocab]`` of ``[1, T, hidden]`` states, in row blocks of
        ``_STREAMED_HEAD_ROWS``; the head stays loaded until the caller drops it."""

        head = self._model.lm_head
        if head.weight.device.type == "meta":
            self._load(head, "lm_head.")
        for start in range(0, hidden.shape[1], _STREAMED_HEAD_ROWS):
            yield head(hidden[:, start : start + _STREAMED_HEAD_ROWS])[0]

    def drop_head(self) -> None:
        self._drop(self._model.lm_head)

    def _load(self, module: Any, prefix: str) -> None:
        """Load the checkpoint tensors under ``prefix`` into ``module`` on the device, converted
        as ``from_pretrained`` converts them (transformers 5.16.1's loader, which the pinned
        transformers version in ``code_version`` names)."""

        from safetensors import safe_open
        from transformers.core_model_loading import convert_and_load_state_dict_in_model
        from transformers.modeling_utils import LoadStateDictConfig

        names = [name for name in self._files if name.startswith(prefix)]
        if not names:
            raise KeyError(f"the checkpoint holds no tensor under {prefix!r}")
        with ExitStack() as stack:
            handles = {
                file: stack.enter_context(safe_open(file, framework="pt", device="cpu"))
                for file in {self._files[name] for name in names}
            }
            state = {name: handles[self._files[name]].get_slice(name) for name in names}
            info, _ = convert_and_load_state_dict_in_model(
                self._model,
                state,
                LoadStateDictConfig(
                    device_map={"": self._device},
                    dtype=self._dtype,
                    weight_mapping=self._weight_mapping,
                ),
            )
        problems = {
            "unexpected": sorted(info.unexpected_keys),
            "mismatched": sorted(info.mismatched_keys),
            "conversion_errors": sorted(info.conversion_errors),
            "errors": list(info.error_msgs),
        }
        if any(problems.values()):
            raise RuntimeError(f"loading {prefix!r} failed: {problems}")
        for name, tensor in module.state_dict().items():
            if (tensor.device, tensor.dtype) != (self._device, self._dtype):
                raise RuntimeError(
                    f"{prefix}{name} is {tensor.dtype} on {tensor.device} after loading, not "
                    f"{self._dtype} on {self._device}"
                )
            self.loaded_bytes += tensor.numel() * tensor.element_size()

    def _drop(self, module: Any) -> None:
        module.to("meta")


def stream_prompts(
    stream: LayerStream,
    prompts: Sequence[tuple[str, list[int]]],
    continuations: Mapping[str, list[int]],
    *,
    top_k: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Each prompt's teacher-forced dump over the prompt and its given continuation, keyed as
    ``run_prompt``'s: ``greedy_ids`` and ``teacher_forced_ids`` both hold the continuation,
    ``greedy_margins`` this run's top-1 minus top-2 at the rows that predict it, and
    ``prefill_logits`` the logits of the prompt's last row."""

    import torch

    sequences = [input_ids + continuations[name] for name, input_ids in prompts]
    hidden = stream.forward(sequences)
    dumps = {}
    for (name, input_ids), states in zip(prompts, hidden, strict=True):
        continuation = continuations[name]
        pieces, prefill_logits = [], None
        last_prompt_row = len(input_ids) - 1
        start = 0
        for block in stream.logits(states):
            if start <= last_prompt_row < start + len(block):
                prefill_logits = block[last_prompt_row - start].float().to("cpu", copy=True)
            pieces.append(reduce_rows(block, top_k=top_k))
            start += len(block)
        reduced = {key: torch.cat([piece[key] for piece in pieces]) for key in pieces[0]}
        predicting = reduced["topk_logits"][last_prompt_row : last_prompt_row + len(continuation)]
        dumps[name] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "greedy_ids": torch.tensor(continuation, dtype=torch.int64),
            "greedy_margins": predicting[:, 0] - predicting[:, 1],
            "prefill_logits": prefill_logits,
            "teacher_forced_ids": torch.tensor(continuation, dtype=torch.int64),
            **reduced,
        }
    stream.drop_head()
    return dumps


def _register_chunked_attention(chunk: int) -> None:
    """Register ``CHUNKED_ATTENTION`` for blocks of ``chunk`` query rows: its attention, and
    a mask function that builds no whole mask (each block builds its own rows)."""

    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    AttentionInterface.register(
        CHUNKED_ATTENTION, functools.partial(_query_chunked_eager_attention, chunk=chunk)
    )
    AttentionMaskInterface.register(CHUNKED_ATTENTION, _no_whole_mask)


def _no_whole_mask(*args: Any, **kwargs: Any) -> None:
    return None


def _query_chunked_eager_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    chunk: int,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """The module's own eager attention over blocks of ``chunk`` query rows, each against the
    keys up to its end and the eager causal mask's rows of the block."""

    import torch
    from transformers.masking_utils import eager_mask

    if attention_mask is not None:
        raise ValueError("query-chunked attention builds its own causal mask")
    eager_attention = sys.modules[type(module).__module__].eager_attention_forward
    length = query.shape[2]
    outputs = []
    for start in range(0, length, chunk):
        stop = min(start + chunk, length)
        mask = eager_mask(
            batch_size=query.shape[0],
            q_length=stop - start,
            kv_length=stop,
            q_offset=start,
            dtype=query.dtype,
            device=query.device,
        )
        output, _ = eager_attention(
            module,
            query[:, :, start:stop],
            key[:, :, :stop],
            value[:, :, :stop],
            mask,
            scaling=scaling,
            dropout=dropout,
        )
        outputs.append(output)
    return torch.cat(outputs, dim=1), None


class StreamRun:
    """``--mode stream``: every prompt through one resident layer at a time, on the
    continuations of the run ``--continuation`` names."""

    def __init__(self, args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
        self._stream = LayerStream(
            args.checkpoint, device=device, dtype=dtype, attention_chunk=args.attention_chunk
        )
        self._args = args
        self.vocab_size = self._stream.vocab_size

    def loaded(self) -> str:
        return "the model on the meta device, its layers loaded as they run"

    def prompts(
        self, prompts: Sequence[tuple[str, list[int]]]
    ) -> Iterator[tuple[str, dict[str, torch.Tensor], float, dict[str, set[str]]]]:
        args = self._args
        started = time.monotonic()
        continuations = {}
        for name, input_ids in prompts:
            ids = run_dump(args.continuation, name, input_ids)["greedy_ids"].tolist()
            if len(ids) != args.greedy_tokens:
                raise ValueError(f"{name}: the continuation has {len(ids)} tokens")
            continuations[name] = ids
        dumps = stream_prompts(self._stream, prompts, continuations, top_k=args.top_k)
        # Every prompt shares each layer's pass, so the time is the whole run's.
        seconds = round(time.monotonic() - started, 3)
        for name, _ in prompts:
            yield name, dumps[name], seconds, {}

    def prompt_report(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        """Continuation tokens that are not this run's top-1 at the row that predicts them."""

        first = len(tensors["input_ids"]) - 1
        predicted = tensors["topk_ids"][first : first + len(tensors["greedy_ids"]), 0]
        return {"continuation_not_top1": int((predicted != tensors["greedy_ids"]).sum())}

    def manifest(self) -> dict[str, Any]:
        return {
            "attention": "eager" if self._args.attention_chunk is None else CHUNKED_ATTENTION,
            "experts": "eager",
            "attention_chunk": self._args.attention_chunk,
            "parameter_bytes": {"streamed": self._stream.loaded_bytes},
            "parameter_dtypes": {"streamed": [self._args.dtype]},
            "pinned_offloaded_bytes": 0,
        }


__all__ = ["CHUNKED_ATTENTION", "LayerStream", "StreamRun", "stream_prompts"]
