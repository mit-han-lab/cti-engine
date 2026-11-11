from __future__ import annotations

from collections.abc import Iterator

import torch

from cti.layers import prepare_inference
from cti.layers.attention import AttentionLayer
from cti.layers.gated_delta_net.layer import GatedDeltaNet


class _Attention(AttentionLayer):
    def __init__(self, prologue: object) -> None:
        self._prologue_op = prologue


class _Recurrent(GatedDeltaNet):
    def __init__(self, name: str, calls: list) -> None:
        self._name, self._calls = name, calls

    def warm_up(self, max_tokens: int, *, max_sequences: int) -> None:
        self._calls.append((self._name, max_tokens, max_sequences))


class _Model:
    """An op tree: its row warm-up records each call; its layers are the ones given."""

    def __init__(self, name: str, layers: list, calls: list) -> None:
        self._name, self._layers, self._calls = name, layers, calls

    def named_ops(self) -> Iterator[tuple[str, object]]:
        yield "", self
        yield from ((f"layers.{index}", layer) for index, layer in enumerate(self._layers))

    def inference_gemms(self) -> Iterator[object]:
        return iter(())

    def settle_inference(self) -> None:
        pass

    def warm_up_rows(self, rows: int, **options: object) -> None:
        self._calls.append(
            (self._name, rows, options["dtype"], options["device"], id(options["warmed"]))
        )


class _Prologue:
    def __init__(self, name: str, calls: list) -> None:
        self._name, self._calls = name, calls

    def warm_up(self, tokens: int) -> None:
        self._calls.append((self._name, tokens))


def test_each_shared_prologue_and_every_recurrent_layer_warm_up_over_doubling_bounds() -> None:
    calls: list[tuple[object, ...]] = []
    target, draft = _Prologue("target", calls), _Prologue("draft", calls)
    recurrent = [_Recurrent(f"recurrent {index}", calls) for index in range(2)]
    # A composed prologue (None) compiles nothing; the target's two layers share one.
    target_layers = [_Attention(target), *recurrent, _Attention(target), _Attention(None)]
    models = [
        _Model("target model", target_layers, calls),
        _Model("draft model", [_Attention(draft)], calls),
    ]

    heard: list[str] = []
    size = prepare_inference(
        models,
        max_rows=6,
        max_sequences=3,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        progress=heard.append,
    )

    assert size == 0
    # Both models' row-dispatched ops share one set of warmed keys, at the largest forward.
    warmed = calls[0][-1]
    assert calls[:2] == [
        ("target model", 6, torch.bfloat16, torch.device("cpu"), warmed),
        ("draft model", 6, torch.bfloat16, torch.device("cpu"), warmed),
    ]
    # Each step reports its shared prologues (two) and recurrent layers as each begins.
    assert heard == [
        f"{step}/4, {tokens} tokens: {what}"
        for step, tokens in enumerate((1, 2, 4, 6), 1)
        for what in ("prologue 1/2", "prologue 2/2", "recurrent layer 1/2", "recurrent layer 2/2")
    ]
    assert calls[2:] == [
        call
        for tokens, sequences in ((1, 1), (2, 2), (4, 3), (6, 3))
        for call in (
            ("target", tokens),
            ("draft", tokens),
            ("recurrent 0", tokens, sequences),
            ("recurrent 1", tokens, sequences),
        )
    ]
