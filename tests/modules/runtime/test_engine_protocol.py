from __future__ import annotations

import dataclasses
import inspect
import pickle

import pytest

from cti.runtime import _protocol as protocol
from tests.support.imports import loaded_modules

# What the runtime facade loads beside the protocol: the parent's side and utils.
_PARENT_MODULES = ("cti.runtime._client", "cti.runtime._process", "cti.utils")

_MESSAGES = tuple(
    export
    for export in (getattr(protocol, name) for name in protocol.__all__)
    if inspect.isclass(export) and dataclasses.is_dataclass(export)
)


@pytest.mark.parametrize("message_type", _MESSAGES, ids=lambda message_type: message_type.__name__)
def test_each_message_is_frozen_and_pickles_back_to_its_own_type(message_type: type) -> None:
    message = message_type(
        **{field.name: f"{field.name}-value" for field in dataclasses.fields(message_type)}
    )

    restored = pickle.loads(pickle.dumps(message))

    assert message_type.__dataclass_params__.frozen
    assert type(restored) is message_type
    assert restored == message


def test_import_loads_no_torch_and_nothing_beyond_the_runtime_parent() -> None:
    loaded = loaded_modules("import cti.runtime._protocol", "cti", "torch")

    parent = {"cti", "cti.runtime", "cti.runtime._protocol"}
    # Training's facade, which holds the commands' vocabulary, loads no torch either.
    training = {"cti.training", "cti.training._errors", "cti.training._protocol"}
    assert parent <= loaded
    assert all(name in parent | training or name.startswith(_PARENT_MODULES) for name in loaded), (
        loaded
    )
