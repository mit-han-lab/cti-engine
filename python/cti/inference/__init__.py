from typing import TYPE_CHECKING

from cti.utils import lazy_getattr

if TYPE_CHECKING:
    from ._control._engine import InferenceEngine as InferenceEngine
    from ._control._scheduling.stops import TextDecoder as TextDecoder
    from ._control._seeds import RequestSeeds as RequestSeeds
    from ._execution._hosting import DeviceBudget as DeviceBudget
    from ._execution._hosting import HostedModels as HostedModels
    from ._execution._hosting import ResidentBeside as ResidentBeside
    from ._execution._speculative import load_speculative as load_speculative
    from ._execution.runner import build_executor as build_executor
    from ._step import Executor as Executor
    from ._step import Forward as Forward
    from ._step import GraphStats as GraphStats
    from ._step import PageMap as PageMap
    from ._step import RequestTable as RequestTable
    from ._step import Round as Round
    from ._step import RoundCompletion as RoundCompletion
    from ._step import Sampled as Sampled
    from ._step import ServingLimits as ServingLimits

# Imported on first use: frontends never load the runner or torch.
_LAZY_ATTRS = {
    "Executor": "._step",
    "Forward": "._step",
    "GraphStats": "._step",
    "PageMap": "._step",
    "Round": "._step",
    "RoundCompletion": "._step",
    "RequestTable": "._step",
    "Sampled": "._step",
    "ServingLimits": "._step",
    "load_speculative": "._execution._speculative",
    "InferenceEngine": "._control._engine",
    "DeviceBudget": "._execution._hosting",
    "HostedModels": "._execution._hosting",
    "ResidentBeside": "._execution._hosting",
    "build_executor": "._execution.runner",
    "RequestSeeds": "._control._seeds",
    "TextDecoder": "._control._scheduling.stops",
}

__getattr__ = lazy_getattr(__name__, _LAZY_ATTRS, globals())

__all__ = list(_LAZY_ATTRS)
