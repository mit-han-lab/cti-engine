"""KV capacity of tensor-parallel ranks: what each rank records and what the parent checks.

Every rank plans its KV pages and snapshot rows with ``kvcache.plan_capacity`` from two values
combined across ranks: the image encoder's activation peak, the largest of any rank, and the
memory available for inference after the rank's attention workspace, the smallest.
``install_capacity_recorder`` wraps the two sizing steps
(``DeviceMemory.peak_bytes``, ``DeviceMemory.plan``) in an engine process, and
records the local value each rank gave those collectives, what they returned, and the plan
it made from them, with the budget it planned. ``assert_published_capacity`` checks that
every rank planned from the agreed values and published that plan.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from cti.kvcache import (
        KVLayout,
        SnapshotPolicy,
        StateBudget,
        StateCapacity,
        TableGeometry,
    )
    from cti.worker import DeviceMemory

# One min_max_int collective: this rank's value and [minimum, maximum] across ranks.
_Collective = tuple[int, list[int]]


def install_capacity_recorder(evidence: dict[str, Any]) -> None:
    """Record this rank's capacity sizing into ``evidence``, in JSON-ready values."""

    from cti import kvcache
    from cti.distributed.runtime import ParallelGroup
    from cti.worker import DeviceMemory

    measure_peak = DeviceMemory.peak_bytes
    determine = DeviceMemory.plan
    plan_capacity = kvcache.plan_capacity
    min_max_int = ParallelGroup.min_max_int
    image_encoder_calls: list[_Collective] = []
    capacity_calls: list[_Collective] = []
    evidence.update(image_encoder=image_encoder_calls, capacity=capacity_calls)

    # Installed before the worker builds its device memory, which keeps the method it binds;
    # a sizing step names the calls list its collectives go to.
    active: list[list[_Collective]] = []

    def record(self: ParallelGroup, value: int) -> tuple[int, int]:
        result = min_max_int(self, value)
        if active:
            active[-1].append((value, list(result)))
        return result

    ParallelGroup.min_max_int = record

    @contextmanager
    def recording(calls: list[_Collective]) -> Iterator[None]:
        active.append(calls)
        try:
            yield
        finally:
            active.pop()

    def record_image_encoder(self: DeviceMemory, run: Callable[[], bool]) -> int:
        # The one peak capacity measures: the image encoder's.
        with recording(image_encoder_calls):
            return measure_peak(self, run)

    def record_capacity(
        self: DeviceMemory,
        layout: KVLayout,
        *,
        geometry: TableGeometry,
        snapshots: SnapshotPolicy | None,
        workspace_bytes: int,
        image_encoder_bytes: int,
        image_rows_bytes: int,
        beside_bytes: int,
        allocations: Callable[[int, int], tuple[int, ...]] | None,
        fit_slots: bool,
    ) -> StateCapacity:
        # The peak this rank was given, and the peak reserve it shares with the text model.
        evidence["image_encoder_peak_bytes"] = image_encoder_bytes
        evidence["peak_reserve_bytes"] = self.reserve_bytes
        with recording(capacity_calls):
            plan = determine(
                self,
                layout,
                geometry=geometry,
                snapshots=snapshots,
                workspace_bytes=workspace_bytes,
                image_encoder_bytes=image_encoder_bytes,
                image_rows_bytes=image_rows_bytes,
                beside_bytes=beside_bytes,
                allocations=allocations,
                fit_slots=fit_slots,
            )
        # Reserved before the KV pages, from this rank's own (sharded) state specs.
        evidence["recurrent_bytes"] = geometry.num_slots * layout.rows.row_bytes
        # The agreed available memory, before what runs beside inference comes off it.
        available = evidence["available"] + beside_bytes
        evidence["plan"] = [available, plan.num_pages, len(plan.snapshot_rows)]
        return plan

    def record_budget(
        layout: KVLayout,
        geometry: TableGeometry,
        budget: StateBudget,
        *,
        snapshots: SnapshotPolicy | None,
        allocations: Callable[[int, int], tuple[int, ...]] | None = None,
    ) -> StateCapacity:
        evidence["available"] = budget.available
        return plan_capacity(layout, geometry, budget, snapshots=snapshots, allocations=allocations)

    kvcache.plan_capacity = record_budget
    DeviceMemory.peak_bytes = record_image_encoder
    DeviceMemory.plan = record_capacity


def assert_published_capacity(
    ranks: Sequence[Mapping[str, Any]],
    published_pages: int,
    published_rows: int,
) -> None:
    """Every rank planned its pages and snapshot rows from the smallest available memory
    and the largest image encoder peak, and published that plan."""

    # DeviceMemory._free's free memory, then the available memory.
    calls = [evidence["capacity"] for evidence in ranks]
    assert all(len(rank_calls) == 2 for rank_calls in calls), calls
    local_available = [available for _, (available, _) in calls]
    measured = [evidence["image_encoder"] for evidence in ranks]
    # Without an image encoder, or with no room for an image, no rank measures a peak.
    assert all(measured) or not any(measured), measured
    local_peaks = [rank_calls[-1][0] for rank_calls in measured if rank_calls]
    image_encoder_peak = max(local_peaks, default=0)
    for evidence, (_, (_, available)) in zip(ranks, calls, strict=True):
        assert available == [min(local_available), max(local_available)]
        if local_peaks:
            assert evidence["image_encoder"][-1][1] == [min(local_peaks), max(local_peaks)]
        assert evidence["image_encoder_peak_bytes"] == image_encoder_peak
        # Every rank reserves the same recurrent state for its own heads.
        assert evidence["recurrent_bytes"] == ranks[0]["recurrent_bytes"]
        assert evidence["plan"] == [min(local_available), published_pages, published_rows]


__all__ = ["assert_published_capacity", "install_capacity_recorder"]
