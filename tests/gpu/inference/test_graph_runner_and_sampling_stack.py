"""The decode-graph runner and the sampler on the installed CUDA stack: a captured graph
replays new inputs through its attention backend, and the sampler draws in-vocabulary tokens."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.support.gpu import isolated_runtime_env, run_isolated_process
from tests.support.paths import REPO_ROOT

pytestmark = pytest.mark.checkpoints()


def _graph_runner_smoke() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    import torch

    from cti.forward import ForwardInputs, ForwardRows
    from cti.inference._execution._graphs import DecodeGraph
    from cti.inference._step import QueryKind, Row
    from cti.memory import GraphPool

    class GraphTestBackend:
        """Owns the graphs' static write slots, as an attention backend does."""

        def __init__(self, device: torch.device) -> None:
            self.capture_calls = 0
            self.replay_calls = 0
            self.cache_write_indices = torch.zeros(1, dtype=torch.int32, device=device)

        def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
            assert max_seq_len == 8
            assert bs_list == [1]

        def prepare_for_capture(self, rows: ForwardRows, inputs: ForwardInputs) -> ForwardInputs:
            assert rows.is_uniform_decode
            static = self.cache_write_indices[: rows.padded_size]
            self.capture_calls += 1
            return replace(inputs, attn_metadata=SimpleNamespace(cache_write_indices=static))

        def prepare_for_replay(self, batch: SimpleNamespace, bs: int) -> None:
            static = self.cache_write_indices[:bs]
            static.copy_(batch.attn_metadata.cache_write_indices)
            self.replay_calls += 1

    class GraphTestModel:
        def __init__(self, device: torch.device) -> None:
            self.offsets = torch.arange(4, device=device, dtype=torch.float32)
            self.forward_calls = 0

        def forward(self, batch: ForwardInputs) -> torch.Tensor:
            self.forward_calls += 1
            writes = batch.attn_metadata.cache_write_indices
            values = batch.input_ids + 10 * batch.rope_positions + 100 * writes
            return values.float()[:, None] + self.offsets[None, :]

    device = torch.device("cuda")
    stream = torch.cuda.Stream(device=device)
    dummy_entry = Row(
        request_index=0,
        query_kind=QueryKind.DECODE,
        kv_prefix_len=0,
        sequence_len=1,
        write_position=1,
    )
    backend = GraphTestBackend(device)
    model = GraphTestModel(device)
    with torch.cuda.stream(stream):
        runner = DecodeGraph(
            stream=stream,
            device=device,
            model=model,
            participants=(backend,),
            batch_sizes=[1],
            max_seq_len=8,
            vocab_size=4,
            logits_dtype=torch.float32,
            dummy_entry=dummy_entry,
            pool=GraphPool(),
        )
    capture_forward_calls = model.forward_calls

    def replay(input_id: int, position: int, cache_write_index: int) -> torch.Tensor:
        rows = runner.rows((dummy_entry,), ())
        assert runner.can_replay(rows)
        with torch.cuda.stream(stream):
            writes = torch.tensor([cache_write_index], device=device, dtype=torch.int32)
            batch = ForwardInputs(
                torch.tensor([input_id], device=device, dtype=torch.int32),
                torch.tensor([position], device=device, dtype=torch.int64),
                SimpleNamespace(cache_write_indices=writes),
                rows.size,
                ragged=False,
            )
            output = runner.replay(rows, batch).clone()
        stream.synchronize()
        return output.cpu()

    first = replay(input_id=1, position=2, cache_write_index=3)
    second = replay(input_id=4, position=5, cache_write_index=6)
    torch.testing.assert_close(first, torch.arange(321, 325, dtype=torch.float32)[None, :])
    torch.testing.assert_close(second, torch.arange(654, 658, dtype=torch.float32)[None, :])
    assert backend.capture_calls == 1
    assert backend.replay_calls == 2
    assert model.forward_calls == capture_forward_calls
    runner.close()


def _stack() -> None:
    import torch

    from cti.inference._execution._sampling import (
        BatchSamplingArgs,
        LegacySamplingLaw,
        LogitProcessing,
        Sampler,
    )
    from cti.utils import DeviceCapability

    capability = torch.cuda.get_device_capability(0)
    assert capability[0] >= 1 and capability[1] >= 0
    assert torch.cuda.get_arch_list()
    _graph_runner_smoke()

    torch.manual_seed(0)
    logits = torch.randn(8, 128, device="cuda")
    temperatures = torch.linspace(0.5, 1.5, 8, device="cuda")
    sampler = Sampler(
        torch.device("cuda"),
        vocab_size=128,
        law=LegacySamplingLaw(DeviceCapability(*torch.cuda.get_device_capability(0))),
        processing=LogitProcessing(torch.zeros(1, 1, dtype=torch.int32, device="cuda"), 128),
        batch_capacity=len(logits),
        logits_dtype=logits.dtype,
    )
    sampled_tokens = sampler.sample(logits, BatchSamplingArgs(temperatures=temperatures)).token_ids
    torch.cuda.synchronize()
    assert sampled_tokens.shape == (8,)
    assert bool(((0 <= sampled_tokens) & (sampled_tokens < 128)).all())

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "graph_runner_replay": "pass",
                "sampling": "pass",
            }
        )
    )


def test_graph_runner_and_sampling_stack(tmp_path: Path) -> None:
    env, log_root = isolated_runtime_env(tmp_path, "stack")
    output = run_isolated_process(
        (sys.executable, "-m", "tests.gpu.inference.test_graph_runner_and_sampling_stack"),
        log_root / "stack.log",
        env=env,
        cwd=REPO_ROOT,
    )

    assert '"graph_runner_replay": "pass"' in output
    assert '"sampling": "pass"' in output


if __name__ == "__main__":
    _stack()
