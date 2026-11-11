import pytest

from tests.support.engine_worker import EngineWorker
from tests.support.gpu import isolated_runtime_env
from tests.support.paths import REPO_ROOT


@pytest.fixture(scope="package")
def engine_worker(tmp_path_factory: pytest.TempPathFactory):
    """The DFlash tests' engine children run here in turn, so the CuTe kernels their
    engines share compile once (see ``tests.support.engine_worker``)."""

    env, log_root = isolated_runtime_env(tmp_path_factory.mktemp("engine-worker"), "worker")
    with EngineWorker(env=env, cwd=REPO_ROOT, log_path=log_root / "worker.log") as worker:
        yield worker
