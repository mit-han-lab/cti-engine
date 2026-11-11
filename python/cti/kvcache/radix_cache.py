# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/kvcache/radix_cache.py`, `python/minisgl/utils/misc.py`);
# see THIRD_PARTY_NOTICES.

"""Radix prefix reuse: the KV tree, the snapshot rows of recurrent state, and the
coordinator that alone joins them."""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, TypeAlias

import torch

from .base import (
    BaseCacheHandle,
    BasePrefixCache,
    InsertResult,
    RestorePlan,
    SizeInfo,
)

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


def align_down(a: int, b: int) -> int:
    """Aligns a to the previous multiple of b"""
    return (a // b) * b


class RadixTreeNode:
    def __init__(self, key_fn: KEY_FN, tic: int = 0) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}
        self.parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.timestamp = tic

        # Set by set_key_value before the node joins the tree.
        self.key: torch.Tensor
        self.value: torch.Tensor

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """``key``: the node's tokens, whole pages; ``value``: their page IDs."""
        self.key = key
        self.value = value

    def set_parent(self, parent: RadixTreeNode) -> None:
        self.parent = parent
        parent.children[self.key_fn(self.key)] = self

    @property
    def length(self) -> int:
        return len(self.key)

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from ctikernels import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self.key, input_ids)

    def split_at(self, pos: int, page_size: int) -> RadixTreeNode:
        assert 0 < pos < self.length and pos % page_size == 0 and self.parent is not None
        parent = self.parent
        page = pos // page_size

        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self.key[:pos], self.value[:page])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set_key_value(self.key[pos:], self.value[page:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        if node.parent.is_root():
            return node.value
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return torch.cat(value_list)


class RadixTree:
    """The token identity of cached KV: page-aligned keys, the KV path locks, and the KV
    LRU on a logical clock (the same requests always evict the same prefixes)."""

    def __init__(self, device: torch.device, page_size: int):
        self.page_size = page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected
        self._tick = 0

    def next_tick(self) -> int:
        self._tick += 1
        return self._tick

    def path(
        self, ids: torch.Tensor
    ) -> tuple[List[RadixTreeNode], tuple[RadixTreeNode, int] | None]:
        """The nodes ``ids`` match whole, root excluded, and the next node with the
        page-aligned tokens ``ids`` match of it, if fewer than its length. Read-only."""

        nodes: List[RadixTreeNode] = []
        node, depth = self.root_node, 0
        while depth < len(ids):
            child = node.children.get(self.key_fn(ids[depth:]))
            if child is None:
                break
            # NOTE: at least 1 page is matched, so matched >= page_size
            matched = align_down(child.get_match_len(ids[depth:]), self.page_size)
            if matched != child.length:
                return nodes, (child, matched)
            nodes.append(child)
            node, depth = child, depth + matched
        return nodes, None

    def stamp(self, nodes: Sequence[RadixTreeNode]) -> None:
        """Mark ``nodes`` the most recently used."""

        tic = self.next_tick()
        for node in nodes:
            node.timestamp = tic

    def insert(self, ids: torch.Tensor, pages: torch.Tensor) -> tuple[int, RadixTreeNode]:
        """Insert the page-aligned prefix of ``ids`` stored in ``pages``, one page ID per
        page; return the tokens cached before and the node at its end."""

        page_size = self.page_size
        insert_len = align_down(len(ids), page_size)
        ids, pages = ids[:insert_len], pages[: insert_len // page_size]
        assert len(pages) * page_size == insert_len
        nodes, partial = self.path(ids)
        if partial is not None:
            nodes.append(partial[0].split_at(partial[1], page_size))
        self.stamp(nodes)
        node = nodes[-1] if nodes else self.root_node
        prefix_len = sum(node.length for node in nodes)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            new_node = RadixTreeNode(self.key_fn, self.next_tick())
            new_node.set_key_value(ids[prefix_len:], pages[prefix_len // page_size :].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return prefix_len, node

    def lock(self, node: RadixTreeNode) -> None:
        while not node.is_root():
            if node.ref_count == 0:
                self.evictable_size -= node.length
                self.protected_size += node.length
            node.ref_count += 1
            node = node.parent

    def unlock(self, node: RadixTreeNode) -> None:
        current = node
        while not current.is_root():
            if current.ref_count <= 0:
                raise RuntimeError("radix cache handle is not locked")
            current = current.parent
        while not node.is_root():
            node.ref_count -= 1
            if node.ref_count == 0:
                self.evictable_size += node.length
                self.protected_size -= node.length
            node = node.parent

    def evict(
        self, size: int, *, order: Callable[[RadixTreeNode], int]
    ) -> tuple[torch.Tensor, List[RadixTreeNode]]:
        """Evict unlocked leaves, lowest ``order`` then least recently used first, until at
        least ``size`` tokens are free; return their page IDs and the evicted nodes."""

        if size == 0:
            return self.empty_tensor, []
        assert size <= self.evictable_size, (
            f"Cannot evict {size}, only {self.evictable_size} is evictable"
        )

        leave_nodes = [(order(n), n) for n in self._collect_leave_nodes_for_evict()]
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted: List[RadixTreeNode] = []
        evicted_size = 0

        while evicted_size < size:
            assert leave_nodes, (
                f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            )
            _, node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            evicted.append(node)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node.key)]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, (order(parent), parent))

        return torch.cat(evicted_indices), evicted

    def reset(self) -> None:
        """Discard all evictable prefixes after every handle is unprotected."""
        if self.protected_size != 0:
            raise RuntimeError(
                f"cannot reset radix cache while {self.protected_size} protected tokens remain"
            )

        new_root = RadixTreeNode(self.key_fn)
        new_root.ref_count = 1

        nodes = [self.root_node]
        while nodes:
            node = nodes.pop()
            nodes.extend(node.children.values())
            node.children.clear()
            node.parent = None

        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = new_root

    def nodes(self) -> Iterator[RadixTreeNode]:
        """Every node in the tree, root included."""

        nodes = [self.root_node]
        while nodes:
            node = nodes.pop()
            nodes.extend(node.children.values())
            yield node

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes


@dataclass(slots=True)
class _HeldRow:
    """A snapshot row a node holds: the state after the node's prefix."""

    row: int
    tick: int
    locks: int = 0


class RowInfo(NamedTuple):
    free: int
    leased: int
    cached: int  # held by nodes
    locked: int  # restore locks on held rows
    evictable: int  # held rows no restore locks


class StateRowIndex:
    """The recurrent-state snapshot rows on the grid ``grid``: each is free, leased to a
    writer, or held by the radix node whose prefix's state it holds. A held row is locked
    while a restore reads it and otherwise reclaimable, the least recently used first."""

    def __init__(self, rows: range, grid: int) -> None:
        self.rows = rows
        self.grid = grid
        self._free = deque(rows)
        self._leased: set[int] = set()
        self._held: dict[RadixTreeNode, _HeldRow] = {}
        self._tick = 0  # a logical clock, as the tree's

    def __len__(self) -> int:
        return len(self.rows)

    def lease(self) -> int | None:
        """Lease a row: free, else the least recently used unlocked snapshot's, else None."""

        if not self._free:
            oldest = min(
                (node for node, held in self._held.items() if held.locks == 0),
                key=lambda node: self._held[node].tick,
                default=None,
            )
            if oldest is None:
                return None
            self.drop(oldest)
        row = self._free.popleft()
        self._leased.add(row)
        return row

    def free(self, row: int) -> None:
        """Return a leased row unpublished."""

        self._leased.remove(row)
        self._free.append(row)

    def unlock(self, handle: BaseCacheHandle) -> None:
        """Release the restore source ``StateCoordinator.acquire`` locked, once read."""

        held = self._live(handle.node)
        if held.locks == 0:
            raise RuntimeError("snapshot is not locked")
        held.locks -= 1

    @property
    def can_lease(self) -> bool:
        return bool(self._free) or any(not held.locks for held in self._held.values())

    @property
    def info(self) -> RowInfo:
        locks = [held.locks for held in self._held.values()]
        return RowInfo(len(self._free), len(self._leased), len(locks), sum(locks), locks.count(0))

    # The coordinator's side: rows join, leave and lock with their nodes.

    def is_leased(self, row: int) -> bool:
        return row in self._leased

    def row_of(self, node: RadixTreeNode) -> int | None:
        held = self._held.get(node)
        return None if held is None else held.row

    def touch(self, node: RadixTreeNode) -> int:
        """Mark ``node``'s row the most recently used; return it."""

        held = self._held[node]
        held.tick = self._next_tick()
        return held.row

    def lock(self, node: RadixTreeNode) -> None:
        self._live(node).locks += 1

    def adopt(self, node: RadixTreeNode, row: int) -> None:
        """Hand leased ``row`` to ``node``; a node that holds one keeps it, freeing ``row``."""

        self._leased.remove(row)
        if node in self._held:
            self._free.append(row)
        else:
            self._held[node] = _HeldRow(row, self._next_tick())

    def drop(self, node: RadixTreeNode) -> None:
        """Free the unlocked row ``node`` holds, if any."""

        held = self._held.pop(node, None)
        if held is not None:
            assert held.locks == 0
            self._free.append(held.row)

    def reset(self) -> None:
        info = self.info
        if info.leased or info.locked:
            raise RuntimeError("cannot reset live snapshots")
        self._held.clear()
        self._free = deque(self.rows)

    def check_integrity(self, reachable: set[RadixTreeNode]) -> None:
        rows = [*self._free, *self._leased, *(held.row for held in self._held.values())]
        if len(rows) != len(set(rows)) or set(rows) != set(self.rows):
            raise RuntimeError("snapshot row ownership is inconsistent")
        for node, held in self._held.items():
            if node not in reachable or held.locks < 0:
                raise RuntimeError("invalid snapshot record")
            if held.locks and node.ref_count == 0:
                raise RuntimeError("snapshot lock without KV protection")

    def _live(self, node: RadixTreeNode) -> _HeldRow:
        held = self._held.get(node)
        if held is None or node.ref_count == 0:
            raise RuntimeError("snapshot lock requires a live KV path lock")
        return held

    def _next_tick(self) -> int:
        self._tick += 1
        return self._tick


class StateCoordinator(BasePrefixCache):
    """Radix prefix reuse: the KV of cached prefixes (``RadixTree``) and, for recurrent
    layers, the snapshot rows of their state (``rows``). The joint rules live here alone:
    a hit restores at most all of a prompt but its final token, which it recomputes, and
    with rows only where a row holds the state; with rows, KV is published only with a
    row; evicting KV frees the rows of its nodes, and KV holding a row goes last."""

    def __init__(self, device: torch.device, page_size: int, rows: StateRowIndex | None):
        super().__init__()
        self.page_size = page_size
        self._tree = RadixTree(device, page_size)
        self.rows = rows

    def warm_up(self) -> None:
        from ctikernels.radix import warm_up

        # Builds (first use in a fresh build cache) or loads the key comparison module.
        warm_up()

    def hit_limit(self, prompt_len: int) -> int:
        """All of the prompt but its final token (H <= P-1), page-aligned."""

        return align_down(max(prompt_len - 1, 0), self.page_size)

    def restorable(self, end: int, exports: tuple[int, ...]) -> tuple[int, ...]:
        """Its whole prefix; with rows, where it exports."""

        return (end,) if self.rows is None else exports

    def restored_len(self, prompt: torch.Tensor, leader: torch.Tensor, position: int) -> int:
        """Without rows, all a hit restores of an identical prompt (an n > 1 sibling):
        waiting on other shared prefixes is a policy of its own. With rows, the exported
        prefix, whole, when the prompts share it and a hit may restore it."""

        limit = self.hit_limit(len(prompt))
        if self.rows is None:
            if position < len(prompt) or not torch.equal(prompt, leader[: len(prompt)]):
                return 0
            return limit
        if position > limit or not torch.equal(prompt[:position], leader[:position]):
            return 0
        return position

    def plan_restore(self, prompt: torch.Tensor) -> RestorePlan:
        """The hit on ``prompt``. Read-only but for the LRU: like the hit it plans, it
        refreshes the nodes it matches, the one the match ends inside too, and the row it
        restores. Without rows the hit may end inside a node, which ``acquire`` splits."""

        ids = prompt[: self.hit_limit(len(prompt))]
        nodes, partial = self._visit(ids)
        kv_match_len = sum(node.length for node in nodes) + (partial[1] if partial else 0)
        if self.rows is not None:
            while nodes and self.rows.row_of(nodes[-1]) is None:
                nodes.pop()
            partial = None
        node = nodes[-1] if nodes else self._tree.root_node
        row = self.rows.touch(node) if self.rows is not None and nodes else None
        cached_len = sum(node.length for node in nodes)
        unprotected_len = sum(node.length for node in nodes if node.ref_count == 0)
        if partial is not None:
            cached_len += partial[1]
            unprotected_len += partial[1] if partial[0].ref_count == 0 else 0
        return RestorePlan(cached_len, kv_match_len, row, unprotected_len, partial or (node, 0))

    def acquire(self, plan: RestorePlan) -> RadixCacheHandle:
        node, split = plan.path
        if split:
            node = node.split_at(split, self.page_size)
        self._tree.lock(node)
        if plan.row is not None:
            self.rows.lock(node)
        return RadixCacheHandle(plan.cached_len, node)

    def lock(self, handle: BaseCacheHandle) -> None:
        self._tree.lock(handle.node)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self._tree.unlock(handle.node)

    def publish(
        self, ids: torch.Tensor, pages: torch.Tensor, row: int | None = None
    ) -> InsertResult:
        """With rows, KV without its ``row`` would hit without the state to restore: it
        publishes nothing, the handle naming the deepest cached prefix of ``ids``. A node
        that holds a row keeps it, and ``row`` is freed."""

        if self.rows is not None and row is None:
            nodes, _ = self._visit(ids[: align_down(len(ids), self.page_size)])
            cached_len = sum(node.length for node in nodes)
            node = nodes[-1] if nodes else self._tree.root_node
            return InsertResult(cached_len, RadixCacheHandle(cached_len, node))
        if row is not None and (
            self.rows is None
            or not self.rows.is_leased(row)
            or not len(ids)
            or len(ids) % self.page_size
        ):
            raise RuntimeError("snapshot publication requires a leased row and aligned prefix")
        cached_len, node = self._tree.insert(ids, pages)
        if row is not None:
            self.rows.adopt(node, row)
        return InsertResult(
            cached_len, RadixCacheHandle(align_down(len(ids), self.page_size), node)
        )

    def evict(self, size: int) -> torch.Tensor:
        pages, evicted = self._tree.evict(size, order=self._eviction_order)
        if self.rows is not None:
            for node in evicted:
                self.rows.drop(node)
        return pages

    def reset(self) -> None:
        if self.rows is not None:
            self.rows.reset()
        self._tree.reset()

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self._tree.evictable_size,
            protected_size=self._tree.protected_size,
        )

    def check_integrity(self) -> None:
        if self.rows is not None:
            self.rows.check_integrity(set(self._tree.nodes()))

    def _visit(
        self, ids: torch.Tensor
    ) -> tuple[List[RadixTreeNode], tuple[RadixTreeNode, int] | None]:
        """``RadixTree.path``, refreshing the LRU of every node visited, the one the match
        ends inside too; never splits."""

        nodes, partial = self._tree.path(ids)
        self._tree.stamp(nodes if partial is None else [*nodes, partial[0]])
        return nodes, partial

    def _eviction_order(self, node: RadixTreeNode) -> int:
        return int(self.rows is not None and self.rows.row_of(node) is not None)


def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
