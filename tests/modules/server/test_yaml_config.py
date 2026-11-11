from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from cti.config import (
    EngineOptions,
    MemoryOptions,
    ModelOptions,
    SchedulingOptions,
    ServerOptions,
)
from cti.server import OnlineServer
from cti.server.args import parse_args


def _config(tmp_path: Path, contents: str, *, name: str = "server.yaml") -> Path:
    path = tmp_path / name
    path.write_text(contents, encoding="utf-8")
    return path


def test_online_server_from_yaml_passes_both_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _config(
        tmp_path,
        """
model: {path: org/model}
server: {host: 0.0.0.0, port: 8080, num_tokenizers: 2}
training:
  trainable_plan: {}
  optimizer: {}
  batching:
    layout: padded
    policy: {kind: static_batching, microbatch_size: 1, max_tokens_per_sample: 32}
""",
    )
    constructor = Mock(return_value=None)
    monkeypatch.setattr(OnlineServer, "__init__", constructor)

    server = OnlineServer.from_yaml(path)

    assert isinstance(server, OnlineServer)
    engine, served = constructor.call_args.args
    assert engine.model == ModelOptions(path="org/model")
    assert engine.training is not None
    assert served == ServerOptions(host="0.0.0.0", port=8080, num_tokenizers=2)


def test_cli_flags_override_the_launch_file_and_absent_flags_do_not(tmp_path: Path) -> None:
    path = _config(
        tmp_path,
        """
model: {path: yaml/model, dtype: auto}
memory: {inference_ratio: 0.5, peak_ratio: 0.75}
scheduling: {max_running_requests: 8, max_batch_tokens: 256}
server: {host: 0.0.0.0, port: 8080, num_tokenizers: 3}
""",
    )

    from_file = parse_args(["--config", str(path)])
    overridden = parse_args(
        [
            "--config",
            str(path),
            "--model.path",
            "cli/model",
            "--scheduling.max-running-requests",
            "4",
            "--memory.peak-ratio",
            "0.9",
            "--server.port",
            "9090",
        ]
    )

    assert from_file == (
        EngineOptions(
            model=ModelOptions(path="yaml/model", dtype="auto"),
            memory=MemoryOptions(inference_ratio=0.5, peak_ratio=0.75),
            scheduling=SchedulingOptions(max_running_requests=8, max_batch_tokens=256),
        ),
        ServerOptions(host="0.0.0.0", port=8080, num_tokenizers=3),
    )
    assert overridden == (
        EngineOptions(
            model=ModelOptions(path="cli/model", dtype="auto"),
            memory=MemoryOptions(inference_ratio=0.5, peak_ratio=0.9),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=256),
        ),
        ServerOptions(host="0.0.0.0", port=9090, num_tokenizers=3),
    )


def test_cli_supplies_the_model_path_a_launch_file_leaves_out(tmp_path: Path) -> None:
    path = _config(tmp_path, "scheduling: {max_running_requests: 4, max_batch_tokens: 64}\n")

    options, _ = parse_args(["--config", str(path), "--model.path", "cli/model"])

    assert options.model.path == "cli/model"
    assert options.scheduling.max_running_requests == 4


def test_cli_requires_a_model_path(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        parse_args([])

    assert caught.value.code == 2
    assert "model is required" in capsys.readouterr().err


def test_image_fetch_options_are_checked(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="existing directory"):
        ServerOptions(allowed_local_media_path=str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="non-negative"):
        ServerOptions(media_fetch_max_retries=-1)
