"""Import cti and cti-kernels and print their versions, PyTorch's and the first GPU; with
--model, also generate a few tokens offline. scripts/install.sh and the container image run it."""

from __future__ import annotations

import argparse
from importlib.metadata import version

import torch

import cti  # noqa: F401
import ctikernels  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        help="generate 16 tokens with this model (a local directory or a Hugging Face id)",
    )
    args = parser.parse_args()
    print(
        "cti",
        version("cti"),
        "| cti-kernels",
        version("cti-kernels"),
        "| torch",
        torch.__version__,
        "| GPU:",
        torch.cuda.get_device_name(0),
    )
    if args.model is None:
        return

    from cti.config import EngineOptions, ModelOptions
    from cti.llm import LLM
    from cti.protocol import SamplingParams

    llm = LLM(EngineOptions(model=ModelOptions(path=args.model)))
    try:
        params = SamplingParams(max_new_tokens=16, temperature=0.0)
        print(repr(llm.generate(["The capital of France is"], params)[0]["text"]))
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
