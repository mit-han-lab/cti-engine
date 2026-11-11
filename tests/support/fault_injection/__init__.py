"""Injections into the engine's own processes, for black-box tests; the library has no hooks.

A test lists injections in ``FaultInjector.ENV`` (JSON) and puts this directory first on
``PYTHONPATH``; its ``sitecustomize`` installs them at interpreter start in every process
the test spawns, applying each to its module as that module finishes loading. ``Raise``
fails the n-th call of a function in the processes it names, ``Replace`` swaps a module
or class attribute for a stand-in, and ``Count`` appends one line per call to a file per
process, so a killed process keeps what it counted.
"""

from __future__ import annotations

import fnmatch
import functools
import importlib
import importlib.abc
import importlib.machinery
import inspect
import json
import multiprocessing
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import ClassVar


@dataclass(frozen=True)
class Injection:
    """Replaces the function or attribute at ``target`` ("module:qualname") with
    ``wrap(it)`` once its module has loaded."""

    KIND: ClassVar[str]

    target: str

    @property
    def module(self) -> str:
        return self.target.partition(":")[0]

    def apply(self, module: ModuleType) -> None:
        *owners, name = self.target.partition(":")[2].split(".")
        owner: object = module
        for part in owners:
            owner = getattr(owner, part)
        setattr(owner, name, self.wrap(inspect.getattr_static(owner, name)))

    def wrap(self, original: Callable) -> object:
        raise NotImplementedError

    @staticmethod
    def in_process(pattern: str) -> bool:
        return fnmatch.fnmatchcase(multiprocessing.current_process().name, pattern)


@dataclass(frozen=True)
class Raise(Injection):
    """The ``nth`` call (from 1) in a process named like ``process`` raises ``message``."""

    KIND = "raise"

    process: str = "*"
    nth: int = 1
    message: str = "injected fault"

    def wrap(self, original: Callable) -> Callable:
        calls = [0]

        @functools.wraps(original)
        def failing(*args: object, **kwargs: object) -> object:
            if self.in_process(self.process):
                calls[0] += 1
                if calls[0] == self.nth:
                    raise RuntimeError(self.message)
            return original(*args, **kwargs)

        return failing


@dataclass(frozen=True)
class Exit(Injection):
    """The ``nth`` call (from 1) in a process named like ``process`` ends it at once with
    ``code``, reporting nothing: a hard crash."""

    KIND = "exit"

    process: str = "*"
    nth: int = 1
    code: int = 9

    def wrap(self, original: Callable) -> Callable:
        calls = [0]

        @functools.wraps(original)
        def exiting(*args: object, **kwargs: object) -> object:
            if self.in_process(self.process):
                calls[0] += 1
                if calls[0] == self.nth:
                    os._exit(self.code)
            return original(*args, **kwargs)

        return exiting


@dataclass(frozen=True)
class Count(Injection):
    """Every call in a process named like ``process`` appends ``target`` to
    ``<directory>/<process name>-<pid>``."""

    KIND = "count"

    directory: str = ""
    process: str = "*"

    def wrap(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def counted(*args: object, **kwargs: object) -> object:
            if self.in_process(self.process):
                name = f"{multiprocessing.current_process().name}-{os.getpid()}"
                with open(Path(self.directory) / name, "a") as log:
                    log.write(self.target + "\n")
            return original(*args, **kwargs)

        return counted


@dataclass(frozen=True)
class Replace(Injection):
    """Swaps the attribute at ``target`` for the object at ``replacement``."""

    KIND = "replace"

    replacement: str = ""

    def wrap(self, original: Callable) -> object:
        module, _, name = self.replacement.partition(":")
        value: object = importlib.import_module(module)
        for part in name.split("."):
            value = getattr(value, part)
        return value


class FaultInjector:
    """The injections of one test, carried to the processes it spawns by the environment."""

    ENV = "CTI_TEST_INJECTIONS"
    KINDS = MappingProxyType({kind.KIND: kind for kind in (Raise, Exit, Count, Replace)})

    def __init__(self, injections: tuple[Injection, ...] = ()) -> None:
        self._injections = injections

    @classmethod
    def from_env(cls) -> FaultInjector:
        rows = json.loads(os.environ.get(cls.ENV, "[]"))
        return cls(tuple(cls.KINDS[row.pop("kind")](**row) for row in rows))

    def environment(self, pythonpath: str) -> dict[str, str]:
        """The variables a spawned process needs to install these injections."""

        rows = [{"kind": injection.KIND, **asdict(injection)} for injection in self._injections]
        here = Path(__file__).parent  # its sitecustomize, then the repository root
        roots = (str(here), str(here.parents[2]), pythonpath)
        return {self.ENV: json.dumps(rows), "PYTHONPATH": os.pathsep.join(filter(None, roots))}

    def install(self) -> None:
        pending: dict[str, list[Injection]] = {}
        for injection in self._injections:
            module = sys.modules.get(injection.module)
            if module is not None:
                injection.apply(module)
            else:
                pending.setdefault(injection.module, []).append(injection)
        if pending:
            sys.meta_path.insert(0, _Finder(pending))


class _Finder(importlib.abc.MetaPathFinder):
    """Applies a module's pending injections as the module finishes loading."""

    def __init__(self, pending: dict[str, list[Injection]]) -> None:
        self._pending = pending

    def find_spec(self, name: str, path: object, target: object = None) -> object:
        injections = self._pending.pop(name, None)
        if injections is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, injections)
        return spec


class _Loader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader, injections: list[Injection]) -> None:
        self._loader = loader
        self._injections = injections

    def create_module(self, spec: object) -> object:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        for injection in self._injections:
            injection.apply(module)


__all__ = ["Count", "Exit", "FaultInjector", "Injection", "Raise", "Replace"]
