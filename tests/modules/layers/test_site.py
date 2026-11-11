"""A test ``Site``: a dry group that refuses to communicate, and the windows a site's
rank reads."""

from __future__ import annotations

import pytest
import torch

from cti.distributed import Topology
from cti.layers import LinearQKVMerged
from cti.layers.placement import WeightShard
from tests.support.tp_shards import site, windows


@pytest.mark.parametrize(
    "collective",
    [
        lambda group: group.all_reduce(torch.ones(1)),
        lambda group: group.all_gather(torch.ones(1)),
    ],
    ids=["all_reduce", "all_gather"],
)
def test_a_dry_sites_group_refuses_to_communicate(collective) -> None:
    dry = site(Topology(1, 2))

    assert (dry.group.rank, dry.group.size) == (1, 2)
    with pytest.raises(RuntimeError, match="does not communicate"):
        collective(dry.group)


@pytest.mark.parametrize("rank", range(4))
def test_kv_heads_fewer_than_ranks_name_their_replicas(rank: int) -> None:
    # TP4 over 2 KV heads: ranks 0-1 hold head 0 and ranks 2-3 head 1.
    qkv = LinearQKVMerged(8, 2, 4, 2, has_bias=False, site=site(Topology(rank, 4)))

    q, k, v = windows(qkv)["weight"]
    head = rank // 2
    assert q == WeightShard(0, 2 * rank, 2 * rank + 2)
    assert k == v == WeightShard(0, 2 * head, 2 * head + 2, replicas=2, replica=rank % 2)
