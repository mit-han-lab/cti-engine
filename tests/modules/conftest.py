"""Keep Transformers' optional CUDA kernels out of the CPU module tests.

transformers 5.16.1 ``modeling_qwen3_5`` binds its Gated DeltaNet and causal
convolution to the ``fla`` and ``causal_conv1d`` kernels when those packages import,
once, at its first import. GPU environments install ``fla`` for ``ctikernels.gdn``;
the CPU tests compare against Transformers' torch path, so both packages are hidden
before any test module imports Transformers.
"""

import sys

sys.modules.setdefault("fla", None)
sys.modules.setdefault("causal_conv1d", None)
