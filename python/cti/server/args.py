# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/server/args.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import argparse
from collections.abc import Sequence

from cti.config import EngineOptions, LaunchFile, ServerOptions
from cti.utils import DataclassTree, init_logger


def parse_args(argv: Sequence[str]) -> tuple[EngineOptions, ServerOptions]:
    """The engine and server options a command line spells: one flag per option path, read
    over the ``--config`` launch file. Serving runs no training."""
    import yaml

    tree = DataclassTree(LaunchFile, exclude=("training",))
    parser = argparse.ArgumentParser(description="cti server", allow_abbrev=False)
    parser.add_argument("--config", help="A YAML launch file; flags override its values.")
    tree.add_flags(parser)
    namespace = parser.parse_args(argv)
    try:
        document = {} if namespace.config is None else tree.read_yaml(namespace.config)
        launch = tree.decode(document, tree.flag_values(namespace))
        options = launch.engine
        if options.training is not None:
            raise ValueError("CLI serving does not run training; use OnlineServer.from_yaml()")
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    init_logger(__name__).info("Parsed options:\n%s\n%s", options, launch.server)
    return options, launch.server
