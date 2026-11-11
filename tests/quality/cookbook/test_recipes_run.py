"""Each training recipe's ``main`` runs end to end against a fake engine: its argument parsing,
its engine and training options, and the batches it builds go through the real public API, so a
recipe cannot drift from it unnoticed. Only the engine, the tokenizer download and HTTP are
faked."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

import cookbook.grpo as grpo
import cookbook.online_training as online_training
import cookbook.sft as sft
from cti.config import EngineOptions
from cti.protocol import AdmissionPriority, SampledSequence, SamplingResult
from cti.training import PackedTokens, PaddedTokens, TrainingBatch


def _tokenizer() -> PreTrainedTokenizerFast:
    words = ("<unk>", "<pad>", "user", "assistant", "OK", "Reply", "with", "OK:")
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<pad>"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
        "{% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
    )
    return tokenizer


class _Future:
    def __init__(self, value: object) -> None:
        self._value = value

    def result(self) -> object:
        return self._value


class _Session:
    """The training session calls a recipe makes, checking the batches are real ones."""

    def __init__(self, engine: _Engine) -> None:
        self._engine = engine
        self.batches: list[TrainingBatch] = []
        self.steps = 0

    def forward_backward(self, batch: TrainingBatch) -> _Future:
        assert isinstance(batch, TrainingBatch)
        self.batches.append(batch)
        return _Future(SimpleNamespace(loss_sum=1.0, normalization_count=1))

    def optim_step(self) -> _Future:
        self.steps += 1
        return _Future(SimpleNamespace(step=self.steps, grad_norm=1.0))

    def publish(self) -> int:
        self._engine.model_revision += 1
        return self._engine.model_revision

    def save_checkpoint(self, path: Path) -> None:
        self._engine.checkpoints.append(Path(path))

    def pause_for_serving(self) -> None:
        pass

    def resume_training(self) -> None:
        pass


class _Engine:
    """``LLM`` and ``OnlineServer``: built from the recipe's options, serving fake samples."""

    built: list[_Engine] = []

    def __init__(self, options: EngineOptions, *_args: object, **_kwargs: object) -> None:
        assert isinstance(options, EngineOptions)
        self.options = options
        self.model_revision = 0
        self.checkpoints: list[Path] = []
        self.priorities: list[object] = []
        self.session = _Session(self)
        _Engine.built.append(self)

    def __enter__(self) -> _Engine:
        return self

    def __exit__(self, *_exc: object) -> None:
        pass

    @contextlib.contextmanager
    def training_session(self):
        yield self.session

    def generate(self, prompts: list[str], params: object) -> list[object]:
        return self.sample([[1] for _ in prompts], params).generation_outputs()

    def sample(
        self, prompts: list[str | list[int]], params: object, *, priority: object = None
    ) -> SamplingResult:
        self.priorities.append(priority)
        if not isinstance(params, list):
            params = [params] * len(prompts)
        samples = tuple(
            SampledSequence(
                request_id=index,
                prompt_token_ids=(1,) if isinstance(prompt, str) else tuple(prompt),
                sampled_token_ids=(index + 1,),
                sampling_logprobs=(-0.1,) if param.logprobs else None,
                cached_prompt_tokens=0,
                sampling=param.snapshot(),
                finish_reason="length",
            )
            for index, (prompt, param) in enumerate(zip(prompts, params, strict=True))
        )
        return SamplingResult(samples, ("",) * len(samples), self.model_revision)

    def shutdown(self) -> None:
        pass


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> type[_Engine]:
    _Engine.built = []
    tokenizer = _tokenizer()
    for module in (sft, grpo, online_training):
        monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *_a, **_k: tokenizer)
    monkeypatch.setattr(sft, "LLM", _Engine)
    monkeypatch.setattr(grpo, "LLM", _Engine)
    monkeypatch.setattr(online_training, "OnlineServer", _Engine)
    monkeypatch.setattr(
        online_training, "_stream_one", lambda *_a: {"text": "OK", "first_token_seconds": 0.1}
    )
    return _Engine


@pytest.mark.parametrize("layout", ["padded", "packed"])
def test_sft_recipe_runs(engine, monkeypatch, tmp_path: Path, layout: str) -> None:
    checkpoint = tmp_path / "sft.pt"
    argv = ["sft.py", "--layout", layout, "--updates", "2", "--checkpoint", str(checkpoint)]
    monkeypatch.setattr(sys, "argv", argv)

    sft.main()

    (built,) = engine.built
    assert built.model_revision == 2 and built.checkpoints == [checkpoint, checkpoint]
    expected = PaddedTokens if layout == "padded" else PackedTokens
    assert all(isinstance(batch.tokens, expected) for batch in built.session.batches)


@pytest.mark.parametrize("layout", ["padded", "packed"])
def test_grpo_recipe_runs(engine, monkeypatch, tmp_path: Path, layout: str) -> None:
    checkpoint = tmp_path / "grpo.pt"
    argv = [
        "grpo.py",
        "--layout",
        layout,
        "--rounds",
        "1",
        "--max-new-tokens",
        "16",
        "--checkpoint",
        str(checkpoint),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    grpo.main()

    (built,) = engine.built
    assert built.model_revision == 1 and built.checkpoints == [checkpoint]
    assert built.session.batches


def test_online_training_recipe_runs(engine, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["online_training.py"])

    online_training.main()

    (built,) = engine.built
    # An SFT step, then a GRPO round sampled at low priority beside HTTP traffic.
    assert built.model_revision == 2 and len(built.session.batches) == 2
    assert built.priorities == [AdmissionPriority.LOW]
