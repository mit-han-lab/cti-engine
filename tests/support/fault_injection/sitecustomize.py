"""Installs a test's injections in every process it spawns (see ``fault_injection``)."""

from tests.support.fault_injection import FaultInjector

FaultInjector.from_env().install()
