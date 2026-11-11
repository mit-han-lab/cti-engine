"""Every rank places exactly what ``goldens/<model>.txt`` records (``tests.support.tp_goldens``),
so moving placement out of the constructors must reproduce it line for line."""

from __future__ import annotations

import difflib

import pytest

from tests.support.tp_goldens import PlacementGoldens


@pytest.fixture(scope="module")
def placed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[str]]:
    return PlacementGoldens().render(tmp_path_factory.mktemp("placement"))


@pytest.mark.parametrize("model", PlacementGoldens.MODELS)
def test_every_rank_places_what_the_golden_records(
    placed: dict[str, list[str]], model: str
) -> None:
    golden = (PlacementGoldens.FOLDER / f"{model}.txt").read_text(encoding="utf-8").splitlines()
    diff = difflib.unified_diff(golden, placed[model], "golden", "placed", lineterm="")
    assert placed[model] == golden, "\n".join(diff)
