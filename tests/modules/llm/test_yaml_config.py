from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from cti.config import EngineOptions, LaunchFile, ModelOptions, SchedulingOptions
from cti.llm import LLM
from cti.protocol import ServingEndpoints

_TRAINING = """
training:
  trainable_plan: {}
  optimizer: {}
  batching:
    layout: padded
    policy: {kind: static_batching, microbatch_size: 1, max_tokens_per_sample: 32}
"""


def _capture_engine(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, dict]]:
    """Record each engine request LLM makes instead of starting an engine."""

    requests = []

    class _EngineClient:
        def __init__(self, options, tuning, **kwargs) -> None:
            requests.append((options, kwargs))
            self.endpoints = ServingEndpoints.create()

        def close_at_exit(self, _owner) -> Mock:
            return Mock()

    monkeypatch.setattr("cti.runtime.EngineClient", _EngineClient)
    return requests


def _config(tmp_path: Path, contents: str, *, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(contents, encoding="utf-8")
    return path


def test_llm_from_yaml_passes_the_files_options_to_the_constructor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _config(
        tmp_path,
        """
model: {path: org/model, initialization_seed: 7}
scheduling: {max_running_requests: 2, max_batch_tokens: 64}
server: {port: 8080}
""",
    )
    constructor = Mock(return_value=None)
    monkeypatch.setattr(LLM, "__init__", constructor)

    llm = LLM.from_yaml(path, resume_from_checkpoint="checkpoint")

    assert isinstance(llm, LLM)
    constructor.assert_called_once_with(
        EngineOptions(
            model=ModelOptions(path="org/model", initialization_seed=7),
            scheduling=SchedulingOptions(max_running_requests=2, max_batch_tokens=64),
        ),
        resume_from_checkpoint="checkpoint",
    )


def test_paths_keep_the_callers_cwd_and_expand_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    caller = tmp_path / "caller"
    (caller / "model").mkdir(parents=True)
    config_dir = tmp_path / "elsewhere"
    config_dir.mkdir()
    path = _config(config_dir, f"model: {{path: ./model}}\n{_TRAINING}")
    monkeypatch.chdir(caller)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    requests = _capture_engine(monkeypatch)

    LLM.from_yaml(path, resume_from_checkpoint="~/checkpoints/latest")

    [(options, kwargs)] = requests
    # The engine resolves the model path in this working directory.
    assert options.model.path == "./model"
    assert kwargs["checkpoint"] == str((tmp_path / "home/checkpoints/latest").resolve())


def test_resume_requires_a_training_section(tmp_path: Path) -> None:
    path = _config(tmp_path, "model: {path: org/model}\n")

    with pytest.raises(ValueError, match="resume_from_checkpoint requires a training section"):
        LLM.from_yaml(path, resume_from_checkpoint="checkpoint")


@pytest.mark.parametrize(
    "contents",
    ("", "- model\n- org/model\n", "model: {path: a}\n---\nmodel: {path: b}\n"),
)
def test_document_must_be_one_mapping(tmp_path: Path, contents: str) -> None:
    path = _config(tmp_path, contents)

    with pytest.raises((ValueError, yaml.YAMLError)):
        LaunchFile.load(path)


def test_duplicate_keys_fail_in_nested_mappings(tmp_path: Path) -> None:
    path = _config(tmp_path, "model:\n  path: a\n  path: b\n")

    with pytest.raises(ValueError, match="duplicate key 'path'"):
        LaunchFile.load(path)


def test_unsafe_yaml_tags_cannot_construct_arbitrary_python_objects(tmp_path: Path) -> None:
    path = _config(tmp_path, "model: {path: !!python/object/apply:builtins.eval ['40 + 2']}\n")

    with pytest.raises(yaml.YAMLError):
        LaunchFile.load(path)


def test_the_launch_file_path_expands_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(tmp_path, "model: {path: org/model}\n")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert LaunchFile.load(Path("~/config.yaml")).engine.model.path == "org/model"
