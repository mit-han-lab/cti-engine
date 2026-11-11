"""A CPU stand-in for an engine's device, so its processes run on a machine without one.

``CpuEngine.injections()`` swaps the resolved checkpoint, the device binding, the GPU worker,
the tokenizer loader and the memory-saver provider for these stand-ins in every process a test spawns; the rest of the engine, its
processes, protocol, scheduling, training accounting and switch order, runs as it does on
a GPU. Each step samples ``TOKEN`` for every row; each forward/backward runs
``MICROBATCHES`` microbatches, each taking ``MICROBATCH_SECONDS`` (from the environment) so
a test can act while one runs; startup runs one phase per ``STARTUP_SECONDS`` entry, each
reported before it sleeps (``exit``: the rank exits there; ``compile:S``: it runs a child
named ``nvcc`` for S seconds, reporting nothing), then loads a checkpoint of one
shard per ``WEIGHT_SHARD_SECONDS`` entry through the engine's loader, each read taking that
long; every residency call and
microbatch is written to ``CpuEngine.LOG`` in its process, when set.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

from tests.support.fault_injection import Replace

# A stand-in is resolved as its target's module finishes loading, maybe in the middle of
# importing the engine or torch: engine names and torch are imported where they are used.


class CpuEngine:
    """The stand-ins, as the injections that install them."""

    LOG = "CTI_TEST_DEVICE_LOG"
    MICROBATCH_SECONDS = "CTI_TEST_MICROBATCH_SECONDS"
    STEP_SECONDS = "CTI_TEST_STEP_SECONDS"  # each forward's duration
    STARTUP_SECONDS = "CTI_TEST_STARTUP_SECONDS"  # comma-separated startup phases' durations
    WEIGHT_SHARD_SECONDS = "CTI_TEST_WEIGHT_SHARD_SECONDS"  # comma-separated shards' read times
    SEEDED = "CTI_TEST_SEEDED_ROWS"  # serve only seeded rows and log each forward's seeds
    STEPS = "CTI_TEST_LOG_STEPS"  # log each forward's row count
    STEP_SECONDS = "CTI_TEST_STEP_SECONDS"  # how long each forward takes
    TOKENIZE_SECONDS = "CTI_TEST_TOKENIZE_SECONDS"  # how long each prompt takes to encode
    TOKEN = 7
    STOP_TOKEN = 2
    MICROBATCHES = 2

    @staticmethod
    def injections() -> tuple[Replace, ...]:
        here = __name__
        return (
            Replace("cti.resolution:ResolvedModel", replacement=f"{here}:CpuCheckpoint"),
            Replace("torch.cuda:set_device", replacement=f"{here}:no_device"),
            Replace(
                "torch.cuda:get_device_properties", replacement=f"{here}:cpu_device_properties"
            ),
            Replace("cti.runtime._core:Rank", replacement=f"{here}:CpuRank"),
            Replace("cti.tokenizer.text:load_tokenizer", replacement=f"{here}:CpuTokenizer"),
            Replace("cti.memory:MemoryManager", replacement=f"{here}:CpuMemoryManager"),
            Replace(
                "cti.memory:configure_memory_saver_subprocess",
                replacement=f"{here}:no_preload",
            ),
        )

    @staticmethod
    def frontend(monkeypatch: object) -> None:
        """Install the stand-in tokenizer in this process, the engine's frontend."""

        import cti.tokenizer.text as text

        monkeypatch.setattr(text, "load_tokenizer", CpuTokenizer)


@contextmanager
def no_preload() -> Iterator[None]:
    """No provider hook to preload on a machine without its library."""

    yield


def no_device(_rank: int) -> None:
    """No device to bind the rank to."""


def cpu_device_properties(_device: object) -> SimpleNamespace:
    """A discrete 32 GiB device's properties: the online prompt budget's 4096-token tier."""

    return SimpleNamespace(total_memory=32 << 30, is_integrated=0)


class CpuTokenizer:
    """Token ids as text: what the engine decodes is visible in its results."""

    eos_token_id = CpuEngine.STOP_TOKEN
    chat_template = None

    @property
    def added_tokens_decoder(self) -> dict[int, object]:
        return {}

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def encode(self, text: str, **_options: object) -> list[int]:
        time.sleep(float(os.environ.get(CpuEngine.TOKENIZE_SECONDS, "0")))
        return [int(part) for part in text.split()]

    def apply_chat_template(self, messages: list[dict[str, str]], **_options: object) -> str:
        """A chat's message texts, joined: its token ids."""

        return " ".join(message["content"] for message in messages)

    def __len__(self) -> int:
        return CpuWorker.VOCAB_SIZE

    def decode(self, token_ids: object, **_options: object) -> str:
        return " ".join(str(int(token)) for token in token_ids)

    def batch_decode(self, sequences: list[list[int]], **_options: object) -> list[str]:
        return [self.decode(sequence) for sequence in sequences]


def _log(event: str) -> None:
    path = os.environ.get(CpuEngine.LOG)
    if path:
        with open(path, "a") as log:
            log.write(f"{event}\n")


class CpuMemoryManager:
    """A provider-bound memory manager whose residency calls are logged in call order;
    its process is reclaimed, never unwound."""

    @classmethod
    def from_stock_provider(cls, *, unified: bool) -> CpuMemoryManager:
        return cls()

    def kv_allocation(self) -> AbstractContextManager[None]:
        return nullcontext()

    def persistent_training_allocation(self) -> AbstractContextManager[None]:
        return nullcontext()

    def transient_training_allocation(self, needed_bytes: int) -> AbstractContextManager[None]:
        return nullcontext()

    kv_bytes = 0

    def suspend_kv(self) -> None:
        _log("suspend_kv")

    def resume_kv(self, _fewest: object) -> None:
        _log("resume_kv")

    def suspend_training(self) -> None:
        _log("suspend_training")

    def resume_training(self, _fewest: object) -> None:
        _log("resume_training")


@dataclass(frozen=True)
class _Event:
    """A completion that has always happened."""

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


@dataclass(frozen=True)
class _Forward:
    """A forward that sampled ``TOKEN`` for each of its rows."""

    rows: int
    completion_event = _Event()

    def output(self) -> object:
        import torch

        from cti.inference import Sampled

        tokens = torch.full((self.rows,), CpuEngine.TOKEN, dtype=torch.int32)
        return Sampled(tokens.numpy().tobytes())


@dataclass(frozen=True)
class CpuConfig:
    """The stand-in model's checkpoint config, as a frontend reads it."""

    architectures: tuple[str, ...] = ("CpuForCausalLM",)

    def get_text_config(self) -> SimpleNamespace:
        return SimpleNamespace(vocab_size=CpuWorker.VOCAB_SIZE)


class CpuCheckpoint:
    """The stand-in model at any path: it admits every request and plans what the stand-in
    worker serves."""

    def __init__(self) -> None:
        self.facts = SimpleNamespace(hf_config=CpuConfig())
        self.model_config = SimpleNamespace(vocab_size=CpuWorker.VOCAB_SIZE)

    @classmethod
    def load(cls, _model_path: str, *, hub: object, text_only: bool) -> CpuCheckpoint:
        return cls()

    def check(self, options: object, _tuning: object, *, speculative: object) -> object:
        return options

    def plan(self, options: object, **_device: object) -> SimpleNamespace:
        from cti.multimodal import ImageLimits
        from cti.plan import PromptBudget

        return SimpleNamespace(
            model_path=options.model.path,
            num_ranks=options.parallel.num_ranks,
            prefix_cache="naive",
            max_seq_len=128,
            max_batch_tokens=options.scheduling.max_batch_tokens,
            max_running_requests=options.scheduling.max_running_requests or 256,
            seeds_every_row=bool(os.environ.get(CpuEngine.SEEDED)),
            image_limits=ImageLimits(128, 0),
            prompt_budget=PromptBudget(
                options.scheduling.max_prompt_tokens or options.scheduling.max_batch_tokens, "cpu"
            ),
            training=options.training,
        )


def _compile(seconds: float) -> None:
    """Wait on a JIT compiler stand-in: a script named ``nvcc``, which the kernel names its
    shell process by, waiting on a ``sleep`` child (a multi-call ``sleep`` dispatches on its
    name, so it cannot be renamed itself)."""

    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        compiler = os.path.join(directory, "nvcc")
        with open(compiler, "w") as script:
            # The exit after sleep keeps the shell from replacing itself with sleep.
            script.write('#!/bin/sh\nsleep "$1"\nexit 0\n')
        os.chmod(compiler, 0o755)
        subprocess.run([compiler, str(seconds)], check=True)


def _load_shards(seconds: list[float], progress: Callable[[str], None]) -> None:
    """Load a checkpoint of one one-weight shard per entry of ``seconds`` through the
    engine's loader, each shard's read taking its entry's seconds."""

    import tempfile

    import safetensors.torch
    import torch

    from cti.layers import Param
    from cti.models.mapping import CheckpointMapping
    from cti.models.weight import load_weight

    def read(phase: str) -> None:
        progress(phase)
        time.sleep(next(reads))

    reads = iter(seconds)
    names = [f"model.w{index}" for index in range(len(seconds))]
    with tempfile.TemporaryDirectory() as folder:
        for index, name in enumerate(names):
            safetensors.torch.save_file(
                {name: torch.zeros(1)}, f"{folder}/shard-{index}.safetensors"
            )
        params = [Param(name, (1,), torch.float32) for name in names]
        loaded = load_weight(
            folder, torch.device("cpu"), mapping=CheckpointMapping(), params=params, progress=read
        )
        assert len(dict(loaded)) == len(names)


class CpuRank:
    """The rank's parts on the CPU: the stand-in worker runs its own steps as the executor,
    beside the stand-in training runner."""

    def __init__(self, plan: SimpleNamespace, checkpoint: CpuCheckpoint, **composition: object):
        self.worker = self.executor = CpuWorker(plan, checkpoint, **composition)
        training, rank = plan.training, composition.get("rank", 0)
        self.training = None if training is None else CpuTrainingRunner(training, rank)


class CpuWorker:
    """The rank's GPU worker on the CPU; it runs its own steps as the model runner."""

    VOCAB_SIZE = 32

    def __init__(
        self,
        plan: SimpleNamespace,
        _checkpoint: CpuCheckpoint,
        *,
        rank: int = 0,
        progress: Callable[[str], None],
        **_rest: object,
    ) -> None:
        from cti.inference import ServingLimits
        from cti.kvcache import KVLayout, RecurrentRows, TableGeometry

        phases = os.environ.get(CpuEngine.STARTUP_SECONDS)
        for index, seconds in enumerate(phases.split(",") if phases else ()):
            progress(f"running stand-in phase {index}")
            if seconds == "exit":
                os._exit(1)  # the rank dies mid-startup, reporting nothing
            if seconds.startswith("compile:"):
                _compile(float(seconds.removeprefix("compile:")))
            else:
                time.sleep(float(seconds))
        if shards := os.environ.get(CpuEngine.WEIGHT_SHARD_SECONDS):
            _load_shards([float(seconds) for seconds in shards.split(",")], progress)
        self._rank = rank
        self._steps = 0
        self.ranks = SimpleNamespace(
            size=plan.num_ranks,
            is_primary=lambda: rank == 0,
            barrier=lambda: None,
            require_same=lambda value, what: None,
        )
        self.limits = ServingLimits(
            capacity=TableGeometry(1, plan.max_running_requests, 128).capacity(
                KVLayout((), RecurrentRows(())), 64, 0
            ),
            rounds=None,
            stop_token_ids=(CpuEngine.STOP_TOKEN,),
            vocab_size=CpuWorker.VOCAB_SIZE,
            image_rows=0,
            prefix_cache=plan.prefix_cache,
            max_batch_tokens=plan.max_batch_tokens,
            max_prompt_tokens=plan.prompt_budget.tokens,
        )

    def submit(self, op: object, table: object) -> _Forward:
        self._all_reduce()
        time.sleep(float(os.environ.get(CpuEngine.STEP_SECONDS, "0")))
        if os.environ.get(CpuEngine.SEEDED):
            seeds = [table[row.request_index].seed for row in op.rows]
            _log(f"rank{self._rank} seeds {seeds}")
        if os.environ.get(CpuEngine.STEPS):
            _log(f"rank{self._rank} step {len(op.rows)}")
        time.sleep(float(os.environ.get(CpuEngine.STEP_SECONDS, "0")))
        return _Forward(len(op.rows))

    def graph_stats(self) -> object:
        from cti.inference import GraphStats

        return GraphStats((), 0, 0, 0)

    def reset_inference_state(self) -> None:
        pass

    def _all_reduce(self) -> None:
        """A forward's collective: every rank arrives before any goes on, so a rank waits
        on a dead peer as on a GPU, until the supervisor reclaims the group."""

        size, log = self.ranks.size, os.environ.get(CpuEngine.LOG)
        if size == 1 or not log:
            return
        step, self._steps = self._steps, self._steps + 1
        marker = f"{log}.step{step}.rank{{}}"
        open(marker.format(self._rank), "w").close()
        while not all(os.path.exists(marker.format(rank)) for rank in range(size)):
            time.sleep(0.001)

    def quiesce(self) -> None:
        _log(f"rank{self._rank} quiesce")

    def release(self, mode: object) -> None:
        from cti.memory import Mode

        if mode is Mode.TRAINING:
            _log(f"rank{self._rank} suspend_training")

    def acquire(self, mode: object) -> None:
        from cti.memory import Mode

        if mode is Mode.SERVING:
            _log(f"rank{self._rank} resume_inference")
            return
        _log(f"rank{self._rank} switch_to_training")

    def close(self) -> None:
        pass


@dataclass
class _Work:
    """One batch's microbatches, run one per call."""

    index: int
    normalization_count: int
    next_microbatch: int = 0


class CpuTrainingRunner:
    """The rank's training runner on the CPU: each batch runs ``MICROBATCHES`` microbatches and
    reports a loss of 1 per target; a commit steps a one-weight optimizer."""

    def __init__(self, config: object, rank: int) -> None:
        self._config = config
        self._rank = rank
        self._batches = 0
        self._steps = 0
        self.holds_gradients = False
        self._accepted: _Work | None = None
        self._suspended = False  # an accepted batch's work is off the device

    def forward_backward(self, batch: object) -> None:
        self._accepted = _Work(self._batches, batch.objective.normalization_count(batch))
        self._batches += 1

    def microbatch(self) -> object | None:
        import torch

        work = self._accepted
        time.sleep(float(os.environ.get(CpuEngine.MICROBATCH_SECONDS, "0")))
        _log(f"rank{self._rank} microbatch {work.index}.{work.next_microbatch}")
        work.next_microbatch += 1
        self.holds_gradients = True
        if work.next_microbatch < CpuEngine.MICROBATCHES:
            return None
        self._accepted = None
        # A batch's metrics as the runner's read back: its loss sum first.
        count = work.normalization_count
        loss = torch.tensor(float(count))
        return SimpleNamespace(
            values=(("loss_sum", "sum", loss),), normalization_count=count, num_targets=count
        )

    def commit(self, normalization_count: int, results: tuple[object, ...]) -> object:
        from cti.training import ForwardBackwardCompletion
        from cti.training.runner import TrainingCommitResult

        self.holds_gradients = False
        self._steps += 1
        completions = tuple(
            ForwardBackwardCompletion(float(result.values[0][2]), result.normalization_count, {})
            for result in results
        )
        return TrainingCommitResult(completions, 0.5)

    def discard(self) -> None:
        self.holds_gradients = False

    def save(self, path: str, *, num_optimizer_steps: int, model_revision: int) -> None:
        self.checkpoint(
            num_optimizer_steps=num_optimizer_steps, model_revision=model_revision
        ).save(path)

    def checkpoint(self, *, num_optimizer_steps: int, model_revision: int) -> object:
        import torch

        from cti.training import TrainingCheckpoint

        state = {
            "step": torch.tensor(float(self._steps)),
            "exp_avg": torch.zeros(1),
            "exp_avg_sq": torch.zeros(1),
        }
        adamw = {"state": {0: state}, "param_groups": []}
        return TrainingCheckpoint(
            model_signature="cpu-model",
            num_optimizer_steps=num_optimizer_steps,
            model_revision=model_revision,
            # The checkpoint's optimizer state, as its fields: one FP32 weight.
            optimizer=SimpleNamespace(
                config=self._config.optimizer,
                names=("weight",),
                live=(torch.ones(1),),
                master=(torch.ones(1),),
                adamw=adamw,
            ),
        )

    def restore(self, checkpoint: object) -> None:
        self._steps = checkpoint.num_optimizer_steps

    def suspend(self) -> None:
        if self._accepted is not None:
            self._suspended = True
            _log(f"rank{self._rank} suspend_training_execution")

    def resume(self) -> None:
        if self._suspended:
            self._suspended = False
            _log(f"rank{self._rank} resume_training_execution")

    def close(self) -> None:
        pass
