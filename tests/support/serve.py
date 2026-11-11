"""Serve a command line with internal engine tuning, for GPU tests:
``python tests/support/serve.py '<EngineTuning as YAML>' <server flags>``."""

import sys

from cti.config import EngineTuning
from cti.server import launch_server
from cti.utils.dataclass_tree import DataclassTree

if __name__ == "__main__":
    launch_server(
        sys.argv[2:],
        tuning=DataclassTree(EngineTuning).decode(DataclassTree.parse_yaml(sys.argv[1])),
    )
