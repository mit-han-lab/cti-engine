"""Every engine process a test starts runs with ``CTI_CHECKS`` on.

In-process objects take ``checks`` from their constructors, which have no default: a test
passes it as an engine rank's composition root does.
"""

import os
import sys
from collections.abc import Iterator

import pytest

# A CPU test that hangs fails under its own name, well inside the quality job's limit; GPU
# tests are bounded by the run that starts them, not per test.
_CPU_TEST_TIMEOUT_SECONDS = 600
_GPU_TESTS = ("tests/gpu/", "tests/hf_parity/")
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _cap_xdist_worker_threads() -> None:
    """Each xdist worker, and every process it starts, uses its share of the CPUs: uncapped
    workers oversubscribe a runner, and starved gloo ranks then stall."""

    workers = os.environ.get("PYTEST_XDIST_WORKER_COUNT")
    if workers is None:
        return
    threads = max(1, len(os.sched_getaffinity(0)) // int(workers))
    for variable in _THREAD_VARIABLES:
        os.environ.setdefault(variable, str(threads))
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(int(os.environ["OMP_NUM_THREADS"]))


_cap_xdist_worker_threads()


@pytest.fixture(autouse=True, scope="session")
def _engine_ranks_run_checks() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CTI_CHECKS", "1")
        yield


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # The cpu-ci dependency group installs pytest-timeout; a venv without it runs untimed.
    if not config.pluginmanager.hasplugin("timeout"):
        config.issue_config_time_warning(
            pytest.PytestConfigWarning("pytest-timeout is not installed: CPU tests run untimed"),
            stacklevel=2,
        )
        return
    for item in items:
        if not item.nodeid.startswith(_GPU_TESTS) and item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(_CPU_TEST_TIMEOUT_SECONDS))
