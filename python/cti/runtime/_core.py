"""A rank's engine process: one thread that takes in commands and requests, lets the mode
switch pick serving or training, and runs that side's next step."""

from __future__ import annotations

import gc
import os
import signal
from contextlib import ExitStack
from functools import cache
from typing import TYPE_CHECKING, NoReturn, Protocol

from cti.env import ENV
from cti.memory import Mode, ResidencyRestoreError
from cti.runtime._protocol import (
    Answered,
    CoreFailed,
    EngineInfo,
    FailurePhase,
    Progress,
    RankProgress,
    ReadRevision,
    Refused,
    Shutdown,
    Started,
    Starting,
    TrainingCommand,
)
from cti.training import CommandStateError, Refusal
from cti.utils import init_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from cti.config import EngineOptions, EngineTuning
    from cti.inference import Executor
    from cti.memory import MemoryManager
    from cti.plan import Plan
    from cti.resolution import ResolvedModel
    from cti.runtime._control import Completion, ControlPlane
    from cti.training.runner import TrainingMemoryPlan, TrainingRunner
    from cti.worker import GPUWorker

logger = init_logger(__name__)


class Switchable(Protocol):
    """One side of the mode switch: inference or training. Each sends its own replies."""

    def wants_to_run(self) -> bool: ...

    def step(self) -> None:
        """Run its next step on the device it holds."""
        ...

    def drained(self) -> bool:
        """Whether it may give the device up now: nothing it started must finish first."""
        ...

    def release(self) -> None:
        """Let go of what lives in the shared memory before the worker moves it away."""
        ...

    def switched(self, holds: bool, refusal: ResidencyRestoreError | None) -> None:
        """The device changed hands, or ``refusal`` (the taker's memory did not fit some
        rank's device, which every rank's worker agreed) kept it with its holder; ``holds``
        says whether this side has it now."""
        ...


class ServingSide(Switchable, Protocol):
    """Serving's side, whose parked step the loop may wait on."""

    @property
    def completion(self) -> Completion | None:
        """The parked step's completion, until the step settles."""
        ...


class TrainingSide(Switchable, Protocol):
    """Training's side, which also asks for the device to move."""

    @property
    def requested_holds(self) -> bool | None:
        """Where a command waits for the device: training (True), serving (False), or
        nowhere (None)."""
        ...

    @property
    def has_active(self) -> bool:
        """Whether an accepted batch has microbatches left to run."""
        ...

    def renew_awaiting(self) -> None:
        """The holder still drains toward the move a command waits for: renew its wait."""
        ...


class ModeSwitch:
    """Owns which side holds the device and when it changes hands, always in one order:
    the holder drains and releases, the worker moves the shared memory, both sides hear
    where the device went. Serving comes first: training yields at its pause boundary
    whenever requests wait, and serving keeps the device, admitting new requests, until
    none runs or waits, however long training's work waits; a session's close or abort
    takes the device back, serving draining for it. A side whose memory does not fit
    refuses the switch, and the holder keeps the device."""

    def __init__(
        self, inference: ServingSide, training: TrainingSide | None, worker: GPUWorker
    ) -> None:
        self._inference = inference
        self._training = training
        self._sides: dict[Mode, Switchable | None] = {
            Mode.SERVING: inference,
            Mode.TRAINING: training,
        }
        self._worker = worker
        self.mode = Mode.SERVING
        self._target: Mode | None = None  # where the device goes once its holder drains

    def runs(self) -> Switchable:
        """The side whose next step the loop runs."""

        return self._sides[self.mode]

    @property
    def switching(self) -> bool:
        return self._target is not None

    @property
    def busy(self) -> bool:
        """Whether the loop has work without new input: the holder wants to run, or the
        device is to change hands, including for a command a step just ran."""

        return self.runs().wants_to_run() or self.switching or self._wanted() is not None

    def completion(self) -> Completion | None:
        """The parked step serving waits on, which the loop waits for with new input,
        while serving keeps the device."""

        if self.mode is not Mode.SERVING or self.switching or self._wanted() is not None:
            return None
        return self._inference.completion

    @property
    def admits_requests(self) -> bool:
        """Whether serving takes new requests: it holds the device and keeps it."""

        return self.mode is Mode.SERVING and self._target is None

    def advance(self) -> None:
        """Make the switch's next move."""

        if self._target is None:
            self._target = self._wanted()
        target = self._target
        if target is None:
            return
        if target is self.mode:  # a command asked for where the device already is
            self._target = None
            self._tell(None)
            return
        holder = self._sides[self.mode]
        if not holder.drained():
            holder.step()  # settle what it started before handing the device over
            self._training.renew_awaiting()  # only training asks for a move it waits on
            return
        self._target = None
        worker = self._worker
        worker.quiesce()
        holder.release()
        worker.release(self.mode)
        try:
            worker.acquire(target)
        except ResidencyRestoreError as error:
            worker.acquire(self.mode)
            self._tell(error)
            return
        self.mode = target
        self._tell(None)

    def _wanted(self) -> Mode | None:
        """Where the device should go: where a training command asks, else to serving
        while requests run or wait (serving first), else to training's work."""

        inference, training = self._inference, self._training
        if training is None:
            return None
        if training.requested_holds is not None:
            return Mode.TRAINING if training.requested_holds else Mode.SERVING
        if inference.wants_to_run():
            return Mode.SERVING if self.mode is Mode.TRAINING else None
        if self.mode is Mode.SERVING and training.wants_to_run():
            return Mode.TRAINING
        return None

    def _tell(self, refusal: ResidencyRestoreError | None) -> None:
        for mode, side in self._sides.items():
            if side is not None:
                side.switched(mode is self.mode, refusal)

    def run_out_training(self) -> None:
        """Give training the device until its accepted batch has run out."""

        training = self._training
        while training is not None and training.has_active:
            self._target = None if self.mode is Mode.TRAINING else Mode.TRAINING
            self.advance()
            if not self.switching:
                self.runs().step()


class Rank:
    """A rank's parts as the composition root assembles them: the worker
    with the device's resources, inference's executor on them and, when configured,
    training's runner. ``close`` releases them in reverse: training, then inference's
    graphs, then the worker."""

    def __init__(
        self,
        plan: Plan,
        checkpoint: ResolvedModel,
        *,
        rank: int = 0,
        init_method: str | None = None,
        memory: MemoryManager | None = None,
        checks: bool,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        """Assembles ``plan``, the plan of ``checkpoint`` for this rank's device. ``checks``
        runs the inference rounds' and training's internal checks; ``progress`` hears each
        long startup phase as it begins; the rest passes on to the worker. A failed
        assembly releases nothing, as the worker's construction does not."""

        from cti.inference import build_executor
        from cti.memory import native_backing
        from cti.training.runner import (
            TrainingRunner,
            device_stack_limit,
            prepare_model_training,
        )
        from cti.worker import GPUWorker

        progress = progress or _quiet
        self._resources = resources = ExitStack()
        self.worker = worker = GPUWorker(
            plan,
            checkpoint,
            rank=rank,
            init_method=init_method,
            memory=memory,
            progress=progress,
        )
        resources.callback(worker.close)
        host, streams, ranks = worker.host, worker.streams, worker.ranks
        training_config = plan.training

        @cache
        def prepare_training() -> TrainingMemoryPlan:
            progress("compiling training kernels")
            adapter = host.adapter
            return prepare_model_training(
                training_config,
                host.model,
                trainables=host,
                rank=0 if adapter is None else adapter.config.rank,
                stream=streams.training,
                allocator=native_backing,
                residency=worker.training_state_residency,
            )

        self.executor: Executor = build_executor(
            host,
            worker.memory,
            plan,
            ranks=ranks,
            preparation_stream=streams.preparation,
            execution_stream=streams.execution,
            resources=resources,
            beside=None if training_config is None else prepare_training,
            checks=checks,
            progress=progress,
        )
        self.training: TrainingRunner | None = None
        # What the training state was planned to hold, which its usage is checked against.
        self.training_plan: TrainingMemoryPlan | None = None
        if training_config is None:
            return
        # Prepared while the executor warmed up, before it sized the KV.
        training = self.training_plan = prepare_training()
        worker.memory.check_training_memory(training)
        self.training = TrainingRunner.create(
            host.model,
            training_config,
            trainables=host,
            stream=streams.training,
            after=streams.execution,
            memory=memory,
            scope_bytes=training.scope_bytes,
            model_signature=host.signature,
            adapter_writer=host.save_adapter,
            ranks=ranks if ranks.size > 1 else None,
            placed=host.adapter,
            stack_limit=device_stack_limit() if checks else None,
            checks=checks,
            residency=worker.training_state_residency,
        )
        resources.callback(self.training.close)
        logger.info_rank0(
            "Training memory held at startup: %s", self.training.memory_usage().against(training)
        )

    def close(self) -> None:
        """Release the rank's parts, for a process that goes on without them; a serving rank
        never does, it exits."""

        self._resources.close()


class EngineCore:
    """One rank's engine. Its control plane hands it the records rank 0 took in, the
    parent's commands and the online requests, in the same order on every rank; each rank
    then runs the side the mode switch picks. A failure that is not a refusal ends the
    process: the parent sees it and reclaims every rank."""

    # A mode's first work builds its kernels, which the first request after startup or a
    # switch waits for: that loop turn may take this long before the parent calls it hung.
    FIRST_WORK_SECONDS = 600.0

    def __init__(
        self,
        options: EngineOptions,
        tuning: EngineTuning,
        plane: ControlPlane,
        *,
        rank: int,
        init_method: str,
        checkpoint: str | None,
        checks: bool,
        overlap: bool,
        online: bool,
        progress: RankProgress,
    ) -> None:
        """``checkpoint`` is a training checkpoint to resume from; ``checks`` runs the
        engine's internal invariant checks; ``overlap`` keeps two inference steps in flight;
        ``online`` plans for an online server rather than offline calls; ``progress`` is
        the record of the loop's progress the parent watches."""

        self._options = options
        self._tuning = tuning
        self._plane = plane
        self._checks = checks
        self._overlap = overlap
        self._online = online
        self._rank = rank
        self._init_method = init_method
        self._checkpoint = checkpoint
        self._phase = FailurePhase.STARTUP
        self._progress = progress
        self._warm: set[Mode] = set()  # the modes that ran work, their first kernels built

    # ================================== Process ==================================

    def run(self) -> NoReturn:
        configure_rank_process()
        try:
            self._start()
            self._phase = FailurePhase.RUNNING
            while True:
                self._iterate()
        except BaseException as error:  # noqa: BLE001 - the process's failure boundary
            self._fail(error)

    def _start(self) -> None:
        import torch

        from cti.inference import InferenceEngine, RequestSeeds, load_speculative
        from cti.memory import MemoryManager, unified_memory
        from cti.resolution import ResolvedModel
        from cti.tokenizer import load_text_decoder
        from cti.training.scheduler import TrainingScheduler
        from cti.utils import get_device_capability

        options, tuning = self._options, self._tuning
        # The rank's one composition root: reject what the checkpoint cannot serve before any
        # device state exists, then plan for this rank's device and inject the plan. The
        # speculative method is loaded against the target and checked and planned with it.
        hub = options.model.source
        resolved = ResolvedModel.load(
            options.model.path, hub=hub, text_only=options.model.text_only
        )
        speculative = (
            None
            if options.speculative is None
            else load_speculative(
                options.speculative.model_path,
                target=resolved.model_config,
                hub=hub,
                speculative_tokens=options.speculative.num_speculative_tokens,
            )
        )
        checked = resolved.check(options, tuning, speculative=speculative)
        device = torch.device("cuda", options.parallel.local_device(self._rank))
        torch.cuda.set_device(device)
        plan = resolved.plan(
            checked,
            device=device,
            capability=get_device_capability(),
            device_memory=torch.cuda.get_device_properties(device).total_memory,
            online=self._online,
        )
        # The rank's parts live as long as the process: their resources hold what the
        # executor's captured graphs read (the graph pool's workspaces) alive.
        self._parts = parts = Rank(
            plan,
            resolved,
            rank=self._rank,
            init_method=self._init_method,
            memory=MemoryManager.from_stock_provider(unified=unified_memory(device)),
            checks=self._checks,
            progress=self._starting,
        )
        worker, executor = parts.worker, parts.executor
        text_decoder = load_text_decoder(plan.model_path)
        self._inference = InferenceEngine(
            executor.limits,
            executor,
            publish=self._plane.publish,
            agree_on_step=worker.ranks.require_same
            if self._checks and worker.ranks.size > 1
            else None,
            text_decoder=text_decoder,
            overlap=self._overlap,
            checks=self._checks,
            model_revision=self._model_revision,
        )
        self._training = None
        if parts.training is not None:
            self._training = TrainingScheduler(
                parts.training,
                self._options.training.training_pause_boundary,
                replies=self,
            )
        self._switch = ModeSwitch(self._inference, self._training, worker)
        if self._training is not None and self._checkpoint is not None:
            self._starting("restoring the training checkpoint")
            self._training.restore(self._checkpoint)
        # Weights are loaded and the checkpoint restored with every thread. From here on the
        # loop's CPU torch ops are small and serial: one thread keeps any of them from forking
        # an OpenMP team (as vLLM's set_torch_threads_for_runtime and SGLang's model runner
        # do). The tokenizer processes are spawned apart and keep their threads.
        torch.set_num_threads(1)
        worker.quiesce()
        # An engine that serves only seeded rows seeds every request on rank 0, before the
        # record carries it to the peers.
        self._plane.start(seeds=RequestSeeds() if plan.seeds_every_row else None)
        gc.collect()
        gc.freeze()  # the startup heap is never scanned again
        self._plane.reply(
            Started(
                EngineInfo(
                    plan.model_path,
                    resolved.facts.hf_config,
                    plan.image_limits,
                    self._checks,
                    _default_template(text_decoder.tokenizer.chat_template),
                )
            )
        )

    # TrainingReplies: the training scheduler answers the parent's commands through these.

    def answer(self, command_id: int, value: object = None) -> None:
        self._plane.reply(Answered(command_id, value))

    def progress(self, command_id: int) -> None:
        self._plane.reply(Progress(command_id))

    def refuse(self, command_id: int, error: Refusal) -> None:
        """Refuse the command ``command_id`` with ``error``; the engine goes on."""

        self._plane.reply(Refused.capture(command_id, error))

    def _starting(self, phase: str) -> None:
        """Report a startup phase to the parent, renewing its wait; rank 0's report stands
        for its peers, which stall it at their next collective if they stall."""

        self._plane.reply(Starting(phase))

    def _fail(self, error: BaseException) -> NoReturn:
        logger.exception("Engine rank %d failed", self._rank)
        try:
            self._plane.fail(CoreFailed.capture(self._phase, error))
        finally:
            os._exit(1)

    # =================================== Loop ====================================

    def _iterate(self) -> None:
        """One turn of the loop. While it has work, its progress record holds the time a turn
        may take: the distributed timeout, or longer for a mode's first work, which builds
        kernels."""

        completion = self._switch.completion()
        if not self._switch.busy and completion is None:
            self._progress.idle()
        record = self._plane.take(block=not self._switch.busy, until=completion)
        mode = self._switch.mode
        timeout = self._tuning.distributed_timeout
        self._progress.working(
            timeout if mode in self._warm else max(timeout, self.FIRST_WORK_SECONDS)
        )
        for command in record.commands:
            match command:
                case TrainingCommand():
                    self._train(command)
                case ReadRevision():
                    self._read_revision(command)
                case Shutdown():
                    self._shutdown(command)
        self._inference.handle(record.requests, admitting=self._switch.admits_requests)
        self._switch.advance()
        if not self._switch.switching:
            side = self._switch.runs()
            if side.wants_to_run():
                self._warm.add(self._switch.mode)
            side.step()
        self._progress.finished()

    # ================================== Commands =================================

    def _read_revision(self, command: ReadRevision) -> None:
        self._plane.reply(Answered(command.id, self._model_revision()))

    def _model_revision(self) -> int:
        """The revision serving samples at: optimizer steps taken, or a checkpoint's."""

        return 0 if self._training is None else self._training.model_revision

    def _train(self, command: TrainingCommand) -> None:
        """A training command: the training scheduler answers it, running it at once on the
        device training holds. An engine without training refuses it."""

        training = self._training
        if training is None:
            self.refuse(command.id, CommandStateError("this engine was started without training"))
            return
        holds = self._switch.mode is Mode.TRAINING
        training.handle(command.id, command.verb, command.args, holds=holds)

    def _shutdown(self, command: Shutdown) -> NoReturn:
        self._phase = FailurePhase.SHUTDOWN
        self._switch.run_out_training()
        self._inference.abort_all()
        self._inference.settle_all()
        self._parts.worker.quiesce()
        self._plane.reply(Answered(command.id))
        while True:
            signal.pause()  # the parent reclaims the process with its provider state


def configure_rank_process() -> None:
    """What a process that builds a GPU worker sets first: the parent owns interrupts, and
    tqdm takes a thread lock, so an exiting or killed process leaks no multiprocessing
    semaphore."""

    from threading import RLock

    from tqdm import tqdm

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    tqdm.set_lock(RLock())


def run_engine_core(
    options: EngineOptions,
    tuning: EngineTuning | None,
    plane: ControlPlane,
    checkpoint: str | None,
    *,
    rank: int,
    init_method: str,
    online: bool,
    progress: RankProgress,
) -> NoReturn:
    """A rank process's entry point, and its composition root; ``tuning`` None keeps its
    defaults."""

    from cti.config import EngineTuning

    EngineCore(
        options,
        EngineTuning() if tuning is None else tuning,
        plane,
        rank=rank,
        init_method=init_method,
        checkpoint=checkpoint,
        checks=ENV.checks,
        overlap=not ENV.disable_overlap_scheduling,
        online=online,
        progress=progress,
    ).run()


def _default_template(chat_template: str | dict[str, str] | None) -> str | None:
    """A tokenizer's chat template; of named ones, the default."""

    return chat_template.get("default") if isinstance(chat_template, dict) else chat_template


def _quiet(phase: str) -> None:
    pass


__all__ = [
    "configure_rank_process",
    "EngineCore",
    "ModeSwitch",
    "Rank",
    "ServingSide",
    "Switchable",
    "TrainingSide",
    "run_engine_core",
]
