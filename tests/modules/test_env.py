import os
import subprocess
import sys

import pytest

from tests.support.paths import source_pythonpath


def _read(variables: dict[str, str | None], field: str) -> str:
    """``ENV.<field>`` as a fresh process reads it under ``variables`` (None unsets one)."""

    environment = {**os.environ, "PYTHONPATH": source_pythonpath()}
    for name, value in variables.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    completed = subprocess.run(
        (sys.executable, "-c", f"from cti.env import ENV; print(ENV.{field})"),
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


@pytest.mark.parametrize(("checks", "expected"), [("1", "True"), (None, "False")])
def test_processes_tests_start_run_checks_which_are_off_by_default(
    checks: str | None, expected: str
) -> None:
    assert os.environ["CTI_CHECKS"] == "1"
    assert _read({"CTI_CHECKS": checks}, "checks") == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", "True"), ("TRUE", "True"), ("yes", "True"), ("0", "False"), ("", "False")],
)
def test_a_switch_is_on_for_one_true_or_yes(value: str, expected: str) -> None:
    variables = {"CTI_DISABLE_OVERLAP_SCHEDULING": value}
    assert _read(variables, "disable_overlap_scheduling") == expected
