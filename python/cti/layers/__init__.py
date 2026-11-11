# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/layers/__init__.py`); see THIRD_PARTY_NOTICES.

from .activation import SiluAndMul
from .attention import AttentionLayer, BidirectionalAttention, PrologueFactory
from .base import BaseOP, OPList
from .block_conv import BlockConv
from .embedding import ParallelLMHead, VocabParallelEmbedding
from .linear import (
    LinearColParallelMerged,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
)
from .lora import Adapter
from .moe import MoELayer
from .norm import LayerNorm, ResidualStreamRMSNorm, RMSNorm, RMSNormFused
from .placement import ExpertParallel, ExpertPlacement, ExpertTensorParallel, Param, Placement
from .prepare import prepare_inference
from .rotary import RotaryEmbedding, VisionRotaryEmbedding
from .services import FeatureTaps, LayerServices
from .site import Site

__all__ = [
    "SiluAndMul",
    "AttentionLayer",
    "BidirectionalAttention",
    "BlockConv",
    "PrologueFactory",
    "BaseOP",
    "OPList",
    "VocabParallelEmbedding",
    "ParallelLMHead",
    "LinearColParallelMerged",
    "LinearRowParallel",
    "LinearQKVMerged",
    "Param",
    "LayerNorm",
    "ResidualStreamRMSNorm",
    "RMSNorm",
    "RMSNormFused",
    "RotaryEmbedding",
    "VisionRotaryEmbedding",
    "Site",
    "FeatureTaps",
    "LayerServices",
    "LinearReplicated",
    "MoELayer",
    "Placement",
    "ExpertParallel",
    "ExpertTensorParallel",
    "ExpertPlacement",
    "Adapter",
    "prepare_inference",
]
