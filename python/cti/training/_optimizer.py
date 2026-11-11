from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import torch

from cti.config import AdamWConfig

from ._checkpoint import OptimizerState
from ._errors import InvalidGradientError
from ._parameter_layout import GradientLayout
from ._parameters import TrainableParameterSet

# AdamW's foreach implementation materializes intermediates for every parameter
# in a parameter group. Bound that payload, except for an indivisible oversized
# parameter, without exposing a backend tuning knob in the public configuration.
_CUDA_FOREACH_GROUP_TARGET_BYTES = 16 * 1024 * 1024

# Zero FP32 tensors of the given shapes on a device: where a class of optimizer state lives.
StateStorage = Callable[[Sequence[torch.Size], torch.device], list[torch.Tensor]]


def separate_storage(shapes: Sequence[torch.Size], device: torch.device) -> list[torch.Tensor]:
    """One tensor per shape, from the allocation scope around the call."""

    return [torch.zeros(shape, dtype=torch.float32, device=device) for shape in shapes]


def foreach_group_sizes(nbytes: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Split parameters of ``nbytes`` each, in order, into AdamW's foreach groups."""

    groups: list[tuple[int, ...]] = []
    current: list[int] = []
    current_bytes = 0
    for parameter_bytes in nbytes:
        if current and current_bytes + parameter_bytes > _CUDA_FOREACH_GROUP_TARGET_BYTES:
            groups.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(parameter_bytes)
        current_bytes += parameter_bytes
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _cuda_foreach_parameter_groups(
    parameters: tuple[torch.Tensor, ...],
) -> tuple[tuple[torch.Tensor, ...], ...]:
    remaining = iter(parameters)
    return tuple(
        tuple(next(remaining) for _ in group)
        for group in foreach_group_sizes(
            [parameter.numel() * parameter.element_size() for parameter in parameters]
        )
    )


@dataclass(frozen=True)
class _MasterParameterBinding:
    name: str
    live: torch.Tensor
    master: torch.nn.Parameter


class AdamWOptimizer:
    """Own selected gradients, AdamW state, and commit failure semantics.

    Every selected parameter trains as an FP32 master (``training_weights``), which the
    model reads once bound and its training kernels cast at each use, so its gradients
    accumulate in FP32; inference reads the live tensors, which change only when
    ``publish`` copies the masters into them."""

    def __init__(
        self,
        trainable_parameters: TrainableParameterSet,
        config: AdamWConfig,
        *,
        checks: bool,
        gradients: GradientLayout,
        deferred: bool = False,
    ) -> None:
        """``checks`` runs the internal finiteness checks of gradients and moments;
        ``gradients`` places this rank's parameters in their logical tensors, for commits
        and checkpoints. ``deferred`` leaves the FP32 masters and AdamW moments to
        ``materialize``, in the caller's allocation scope; nothing else runs before it."""
        self._checks = checks
        self._layout = gradients

        named_parameters = trainable_parameters.named_parameters()
        parameters = tuple(parameter for _, parameter in named_parameters)
        if any(parameter.dtype not in (torch.float32, torch.bfloat16) for parameter in parameters):
            raise ValueError("Training supports FP32 and BF16 selected parameters")

        devices = {parameter.device for parameter in parameters}
        if len(devices) != 1:
            raise ValueError("All selected parameters must reside on one device")

        self.device = next(iter(devices))
        self.config = config
        self._named_parameters = named_parameters
        self._named_complement = trainable_parameters.named_complement()
        self._uses_foreach = self.device.type == "cuda"
        # FP32 gradients, one per updated element.
        self.gradient_bytes = 4 * sum(parameter.numel() for parameter in parameters)
        self._materialized = False
        if not deferred:
            self.materialize()

    @property
    def materialized(self) -> bool:
        return self._materialized

    def materialize(
        self,
        *,
        masters: StateStorage = separate_storage,
        moments: StateStorage = separate_storage,
    ) -> None:
        """Create the FP32 masters from the live parameters in ``masters``' storage, and the
        AdamW moments, in parameter order and both of a parameter in turn, in ``moments``'."""

        if self._materialized:
            return
        self._create_masters(masters)
        self._create_moments(moments)
        self._materialized = True

    def _create_masters(self, storage: StateStorage) -> None:
        # An FP32 live value gets its own master too: training never writes what inference
        # reads.
        values = storage([parameter.shape for _, parameter in self._named_parameters], self.device)
        with torch.no_grad():
            for value, (_, parameter) in zip(values, self._named_parameters, strict=True):
                value.copy_(parameter.detach())
        master_bindings = tuple(
            _MasterParameterBinding(name=name, live=parameter, master=torch.nn.Parameter(value))
            for value, (name, parameter) in zip(values, self._named_parameters, strict=True)
        )
        named_update_parameters = tuple(
            (binding.name, binding.master) for binding in master_bindings
        )
        update_parameters = tuple(parameter for _, parameter in named_update_parameters)

        self._master_bindings = master_bindings
        # The addresses of the master gradients placed in the caller's allocation scope.
        self._placed_gradients: set[int] = set()
        self._named_update_parameters = named_update_parameters
        self._update_parameters = update_parameters

    def _create_moments(self, storage: StateStorage) -> None:
        config, update_parameters = self.config, self._update_parameters
        optimizer_groups = [{"params": update_parameters}]
        if self._uses_foreach:
            optimizer_groups = [
                {"params": group} for group in _cuda_foreach_parameter_groups(update_parameters)
            ]
        self._torch_optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            foreach=self._uses_foreach,
            fused=False,
        )
        # The state AdamW would create at its first step, created here in parameter order:
        # steps then allocate only temporaries, and a restore copies into it in place.
        moments = iter(
            storage(
                [parameter.shape for parameter in update_parameters for _ in range(2)],
                self.device,
            )
        )
        for parameter in update_parameters:
            self._torch_optimizer.state[parameter] = {
                "step": torch.tensor(0.0, dtype=torch.float32),
                "exp_avg": next(moments),
                "exp_avg_sq": next(moments),
            }

    def training_weights(self) -> dict[str, torch.nn.Parameter]:
        """The FP32 masters the model trains on, by parameter name, for its
        ``bind_training_weights``."""

        return {binding.name: binding.master for binding in self._master_bindings}

    def accumulate_gradients(self) -> None:
        """Move each master gradient autograd just created into the caller's scope; autograd
        adds later microbatches' gradients to it in place, in FP32."""

        with torch.no_grad():
            for parameter in self._update_parameters:
                gradient = parameter.grad
                if gradient is not None and gradient.data_ptr() not in self._placed_gradients:
                    parameter.grad = gradient.clone()
                    self._placed_gradients.add(parameter.grad.data_ptr())

    def validate_gradients(
        self,
        normalization_count: int,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Return device-side commit inputs without reading CUDA values on the host."""

        try:
            gradients = self._validated_gradients()
        except InvalidGradientError:
            self._layout.abandon(self._update_parameters)
            self._clear_gradients()
            raise
        return gradients, self._layout.reduce(gradients) / normalization_count

    def apply_gradients(
        self,
        normalization_count: int,
        *,
        gradients: tuple[torch.Tensor, ...],
        normalized_norm: torch.Tensor,
    ) -> None:
        """Apply gradients whose topology and finite norm were already validated."""

        with torch.no_grad():
            if self._uses_foreach:
                torch._foreach_div_(gradients, normalization_count)
            else:
                for gradient in gradients:
                    gradient.div_(normalization_count)
            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grads_with_norm_(
                    self._update_parameters,
                    self.config.max_grad_norm,
                    normalized_norm,
                    foreach=self._uses_foreach,
                )
        self._torch_optimizer.step()
        self._clear_gradients()

    def publish(self) -> None:
        """Copy the masters into the live tensors inference reads, in place (their addresses,
        which captured graphs hold, unchanged), rounded to the live dtype."""

        with torch.no_grad():
            lives = tuple(binding.live for binding in self._master_bindings)
            masters = tuple(binding.master for binding in self._master_bindings)
            if self._uses_foreach:
                torch._foreach_copy_(lives, masters)
            else:
                for live, master in zip(lives, masters, strict=True):
                    live.copy_(master)

    def discard_accumulation(self) -> None:
        """Discard an uncommitted accumulation window without updating parameters."""

        if self._materialized:
            self._clear_gradients()
        else:
            for _, tensor in (*self._named_parameters, *self._named_complement):
                tensor.grad = None

    @property
    def has_pending_accumulation(self) -> bool:
        if not self._materialized:
            return any(
                tensor.grad is not None
                for _, tensor in (*self._named_parameters, *self._named_complement)
            )
        return (
            any(parameter.grad is not None for parameter in self._update_parameters)
            or any(binding.live.grad is not None for binding in self._master_bindings)
            or any(tensor.grad is not None for _, tensor in self._named_complement)
        )

    def device_state(self) -> dict[str, tuple[torch.Tensor, ...]]:
        """The tensors this optimizer keeps: FP32 masters, AdamW moments and accumulated
        gradients, by ``TrainingMemoryPlan`` component; none before ``materialize``."""

        if not self._materialized:
            return {"master_weights": (), "optimizer_state": (), "gradients": ()}
        state = self._torch_optimizer.state
        return {
            "master_weights": tuple(binding.master for binding in self._master_bindings),
            "optimizer_state": tuple(
                state[parameter][moment]
                for parameter in self._update_parameters
                for moment in ("exp_avg", "exp_avg_sq")
            ),
            "gradients": tuple(
                parameter.grad
                for parameter in self._update_parameters
                if parameter.grad is not None
            ),
        }

    def accumulated_gradients(self) -> tuple[torch.Tensor, ...]:
        """Host copies of the FP32 gradients the open window holds, in parameter order."""

        gradients = tuple(parameter.grad for parameter in self._update_parameters)
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("the accumulation window holds no gradients")
        return tuple(gradient.detach().to("cpu", copy=True) for gradient in gradients)

    def require_no_pending_accumulation(self) -> None:
        """Require every selected gradient to have been committed or discarded."""

        if self.has_pending_accumulation:
            raise RuntimeError(
                "training accumulation must be committed or discarded before suspension"
            )

    def suspend_gradients(self, host: torch.Tensor) -> None:
        """Copy the open accumulation window's gradients into ``host``, FP32 in parameter
        order, without waiting. The gradients keep their device memory, whose contents a
        suspension does not keep."""

        if not self._materialized:
            return
        for parameter, stash in self._stash_views(host):
            if parameter.grad is not None:
                stash.copy_(parameter.grad, non_blocking=True)

    def resume_gradients(self, host: torch.Tensor) -> None:
        """Copy what ``suspend_gradients`` stashed in ``host`` back in place, without
        waiting."""

        if not self._materialized:
            return
        for parameter, stash in self._stash_views(host):
            if parameter.grad is not None:
                parameter.grad.copy_(stash, non_blocking=True)

    def _stash_views(self, host: torch.Tensor) -> Iterator[tuple[torch.nn.Parameter, torch.Tensor]]:
        offset = 0
        for parameter in self._update_parameters:
            numel = parameter.numel()
            yield parameter, host[offset : offset + numel].view(parameter.shape)
            offset += numel

    def trained_state(self) -> dict[str, torch.Tensor]:
        """Copy the masters, rounded to the live dtype, to the host as logical tensors by
        name: what training has reached, published or not. Every accumulation window must
        be settled."""

        self.require_no_pending_accumulation()
        return {
            binding.name: self._logical(index, binding.master.detach().to(binding.live.dtype))
            for index, binding in enumerate(self._master_bindings)
        }

    def checkpoint_state(self) -> OptimizerState:
        """Copy the published live values, the masters and AdamW's state to the host, as
        logical tensors, once every accumulation window is settled."""

        self.require_no_pending_accumulation()
        adamw = self._torch_optimizer.state_dict()
        live, master = [], []
        for index, binding in enumerate(self._master_bindings):
            live.append(self._logical(index, binding.live))
            master.append(self._logical(index, binding.master))
            # state_dict() shares each parameter's live state dictionary; copy, not mutate.
            state = adamw["state"][index]
            adamw["state"][index] = {
                "step": state["step"].clone(),
                "exp_avg": self._logical(index, state["exp_avg"]),
                "exp_avg_sq": self._logical(index, state["exp_avg_sq"]),
            }
        names = tuple(name for name, _ in self._named_update_parameters)
        return OptimizerState(self.config, names, tuple(live), tuple(master), adamw)

    def restore_checkpoint_state(self, state: OptimizerState) -> None:
        """Copy a checkpoint's state into this optimizer's tensors in place: the published
        values into the live tensors and the masters and moments into theirs, publishing
        nothing the checkpoint had not."""

        self._check_checkpoint(state)
        with torch.no_grad():
            for index, binding in enumerate(self._master_bindings):
                binding.live.copy_(self._local(index, state.live[index]))
                binding.master.copy_(self._local(index, state.master[index]))
                saved = state.adamw["state"][index]
                current = self._torch_optimizer.state[binding.master]
                current["step"].copy_(saved["step"])
                for moment in ("exp_avg", "exp_avg_sq"):
                    current[moment].copy_(self._local(index, saved[moment]))

    def restore_live_state(self, state: OptimizerState) -> None:
        """Copy a checkpoint's live parameter values in place, before ``materialize``: what
        inference reads; ``restore_checkpoint_state`` restores the rest once it runs."""

        self._check_checkpoint(state)
        with torch.no_grad():
            for index, (_, live) in enumerate(self._named_parameters):
                live.copy_(self._local(index, state.live[index]))

    def _check_checkpoint(self, state: OptimizerState) -> None:
        self.require_no_pending_accumulation()
        names = tuple(name for name, _ in self._named_parameters)
        if state.config != self.config:
            raise ValueError("Checkpoint AdamW configuration does not match this optimizer")
        if state.names != names:
            raise ValueError(f"Checkpoint parameters {state.names} do not match {names}")
        for index, (name, live) in enumerate(self._named_parameters):
            if state.live[index].shape != self._logical_shape(index, live):
                raise ValueError(f"Checkpoint shape of {name!r} does not match")
            if state.live[index].dtype != live.dtype:
                raise ValueError(f"Checkpoint dtype of {name!r} does not match")

    def _logical(self, index: int, tensor: torch.Tensor) -> torch.Tensor:
        """Parameter ``index``'s logical (unsharded) value of ``tensor``, on the host."""

        logical = self._layout.logical(index, tensor.detach())
        return logical.to(device="cpu", copy=True).contiguous()

    def _local(self, index: int, logical: torch.Tensor) -> torch.Tensor:
        """This rank's window of parameter ``index``'s ``logical`` value."""

        return self._layout.local(index, logical)

    def _logical_shape(self, index: int, tensor: torch.Tensor) -> torch.Size:
        """The logical shape of parameter ``index``'s ``tensor``."""

        return torch.Size(self._layout.logical_shape(index, tensor))

    def _validated_gradients(self) -> tuple[torch.Tensor, ...]:
        for name, tensor in self._named_complement:
            if tensor.grad is not None:
                raise InvalidGradientError(f"Frozen complement gradient is present for {name}")
        for binding in self._master_bindings:
            if binding.live.grad is not None:
                raise InvalidGradientError(
                    f"Live tensor {binding.name} received a gradient: training reads its master"
                )

        gradients = []
        for name, parameter in self._named_update_parameters:
            gradient = parameter.grad
            if gradient is None:
                raise InvalidGradientError(f"Selected gradient is missing for {name}")
            gradients.append(gradient)
        return tuple(gradients)

    def _clear_gradients(self) -> None:
        self._torch_optimizer.zero_grad(set_to_none=True)
        self._placed_gradients.clear()
        for binding in self._master_bindings:
            binding.live.grad = None
        for _, tensor in self._named_complement:
            tensor.grad = None


__all__ = [
    "AdamWOptimizer",
]
