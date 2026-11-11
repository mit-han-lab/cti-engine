# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/server/__init__.py`); see THIRD_PARTY_NOTICES.

from typing import TYPE_CHECKING

from cti.utils import lazy_getattr

from .launch import launch_server

if TYPE_CHECKING:
    from .online_server import OnlineServer

__getattr__ = lazy_getattr(__name__, {"OnlineServer": ".online_server"}, globals())

__all__ = ["OnlineServer", "launch_server"]
