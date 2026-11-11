import time

import pytest
import torch

from tests.support.inference.cache import prefix_cache
from tests.support.inference.prefix_cache import reference_radix_comparison  # noqa: F401


def cache(rows=3):
    return prefix_cache(snapshot_rows=range(4, 4 + rows))


def radix(page_size=1):
    return prefix_cache(page_size)


def ids(n, offset=0):
    return torch.arange(n, dtype=torch.int32) + offset


def publish(c, tokens):
    """Publish ``tokens`` (page IDs equal to them) with a leased row; return the row."""
    row = c.rows.lease()
    assert row is not None
    c.publish(tokens, tokens.clone(), row)
    return row


def hit(c, n, offset=0):
    """The hit on a prompt one token past ``n`` tokens, which may restore all of them."""
    return c.plan_restore(ids(n + 1, offset))


def test_partial_match_never_splits_and_falls_back_to_deepest_snapshot():
    c = cache()
    shallow = publish(c, ids(64))
    deep = publish(c, ids(192))
    plan = hit(c, 128)
    assert (plan.cached_len, plan.kv_match_len, plan.row) == (64, 128, shallow)
    # Ordinary publication cannot expose a KV-only tail.
    result = c.publish(ids(256), ids(256))
    assert result.handle.cached_len == 192
    assert hit(c, 256).kv_match_len == 192
    branch = publish(c, ids(128))
    # The split's new parent at 128 holds only the snapshot published there.
    plan = hit(c, 160)
    assert (plan.cached_len, plan.kv_match_len, plan.row) == (128, 160, branch)
    plan = hit(c, 192)
    assert (plan.cached_len, plan.row) == (192, deep)
    c.check_integrity()


def test_lease_takes_a_free_row_then_the_lru_unlocked_snapshot_then_none():
    c = cache(2)
    a_row = publish(c, ids(64))
    publish(c, ids(128))
    held = c.acquire(hit(c, 128))  # locks the second's KV path and its row
    assert held.cached_len == 128 and torch.equal(held.get_matched_indices(), ids(128))
    assert c.rows.info.locked == 1
    row = c.rows.lease()
    assert row == a_row and hit(c, 80).cached_len == 0
    assert c.rows.lease() is None  # the other snapshot is locked
    c.rows.unlock(held)
    c.unlock(held)
    # A restore source unlocks only under its KV path lock.
    with pytest.raises(RuntimeError, match="live KV path lock"):
        c.rows.unlock(held)
    c.rows.free(row)
    c.evict(128)
    c.check_integrity()
    assert c.rows.info == (2, 0, 0, 0, 0)
    c.reset()


@pytest.mark.parametrize("hybrid", [False, True])
def test_eviction_follows_access_order_under_a_constant_clock(monkeypatch, hybrid):
    # Ranks share no clock: equal readings must not change which prefix is evicted.
    monkeypatch.setattr(time, "monotonic_ns", lambda: 1)
    c = cache() if hybrid else radix()
    prefixes = [ids(4) + 100 * group for group in range(3)]
    for prefix in prefixes:
        c.publish(prefix, prefix, c.rows.lease() if hybrid else None)
    c.plan_restore(torch.cat((prefixes[0], prefixes[0][:1])))
    assert torch.equal(c.evict(4), prefixes[1])
    assert torch.equal(c.evict(4), prefixes[2])


def test_a_duplicate_publication_keeps_the_existing_row_and_frees_its_own():
    c = cache(2)
    first = publish(c, ids(64))
    row = c.rows.lease()
    duplicate = c.publish(ids(64), ids(64) + 1000, row)
    assert duplicate.cached_len == 64
    assert torch.equal(duplicate.handle.get_matched_indices(), ids(64))
    assert hit(c, 64).row == first != row
    assert c.rows.info == (1, 0, 1, 0, 1)
    leased = c.rows.lease()
    with pytest.raises(RuntimeError, match="live snapshots"):
        c.reset()
    c.rows.free(leased)
    c.reset()
    assert c.rows.info.free == 2


# The joint rules' counter-examples.


def test_a_kv_match_ending_inside_a_node_restores_it_only_without_rows_and_splits_on_acquire():
    plain, hybrid = radix(4), prefix_cache(4, snapshot_rows=range(1))
    for c in (plain, hybrid):
        c.publish(ids(16), ids(4), c.rows.lease() if c.rows else None)  # 4 pages
    plan = plain.plan_restore(ids(10))  # 9 restorable tokens: two pages
    assert (plan.cached_len, plan.kv_match_len, plan.unprotected_len) == (8, 8, 8)
    plan = hybrid.plan_restore(ids(10))
    assert (plan.cached_len, plan.kv_match_len, plan.row) == (0, 8, None)
    for c in (plain, hybrid):  # planning never splits: the prefix goes whole
        assert torch.equal(c.evict(4), ids(4))
    plain.publish(ids(16), ids(4))
    plain.acquire(plain.plan_restore(ids(10)))
    assert plain.size_info == (8, 8)  # the hit protects its two pages alone


def test_a_hit_restores_the_deepest_row_shallower_than_the_kv_match():
    c = cache()
    shallow = publish(c, ids(64))
    publish(c, ids(128))
    deep = publish(c, ids(192))
    hit(c, 64)
    hit(c, 192)
    c.rows.lease()  # reclaims the least recently used row, the middle one
    plan = hit(c, 160)
    assert (plan.cached_len, plan.kv_match_len, plan.row) == (64, 160, shallow)
    assert hit(c, 128).cached_len == 64
    plan = hit(c, 192)
    assert (plan.cached_len, plan.row) == (192, deep)


def test_a_row_only_at_the_full_prompt_falls_back():
    c = cache()
    shallow = publish(c, ids(64))
    publish(c, ids(128))
    # A 128-token prompt's hit recomputes its final token: the row at 128 is out of reach.
    plan = c.plan_restore(ids(128))
    assert (plan.cached_len, plan.kv_match_len, plan.row) == (64, 127, shallow)
    assert radix().plan_restore(ids(128)).cached_len == 0
    assert c.restored_len(ids(128), ids(200), 128) == 0
    assert c.restored_len(ids(129), ids(200), 128) == 128


def test_evicting_kv_frees_its_rows_after_rowless_kv():
    c = cache(2)
    a = publish(c, ids(64))
    publish(c, ids(64, 1000))
    hit(c, 64)  # the first is the more recently used
    c.rows.lease()  # reclaims the second's row: its KV stays, rowless
    assert c.size_info.evictable_size == 128 and c.rows.info.cached == 1
    assert torch.equal(c.evict(64), ids(64, 1000))  # rowless KV goes first
    assert hit(c, 64).row == a
    assert torch.equal(c.evict(64), ids(64))
    assert c.rows.info == (1, 1, 0, 0, 0)
    c.check_integrity()


def test_reclaiming_a_row_keeps_its_kv():
    c = cache(1)
    publish(c, ids(64))
    row = c.rows.lease()
    assert c.size_info.evictable_size == 64
    plan = hit(c, 64)
    assert (plan.cached_len, plan.kv_match_len, plan.row) == (0, 64, None)
    c.rows.free(row)
    c.check_integrity()


def test_a_planned_hit_counts_the_tokens_acquiring_it_protects():
    c = radix()
    c.publish(ids(64), ids(64))
    plan = hit(c, 64)
    assert plan.unprotected_len == 64
    handle = c.acquire(plan)
    assert c.size_info == (0, 64)
    assert hit(c, 64).unprotected_len == 0
    c.unlock(handle)
