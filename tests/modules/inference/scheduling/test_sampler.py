from __future__ import annotations

import math
from contextlib import nullcontext

import pytest
import torch

import cti.inference._execution._sampling as sample_module
import ctikernels.sampling
from cti.inference._execution._sampling import (
    BatchSamplingArgs,
    LegacySamplingLaw,
    LogitProcessing,
    LogitRows,
    RequestLocalSamplingLaw,
    SampledLogprobs,
    Sampler,
    TokenSampling,
)
from cti.inference._step import QueryKind, RequestSpec, Row
from cti.protocol import SamplingParams
from cti.sampling import LogprobMode, SamplingError
from cti.utils import DeviceCapability


class RecordingCategoricalSampling:
    def __init__(self) -> None:
        self.softmax_calls: list[tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor]] = []
        self.sample_calls: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]] = []

    def softmax(self, logits, temperatures, *, enable_pdl):
        probabilities = torch.softmax(logits / temperatures[:, None], dim=-1)
        self.softmax_calls.append((logits, temperatures, enable_pdl, probabilities))
        return probabilities

    def sample(self, probabilities, *, top_k=None, top_p=None):
        self.sample_calls.append((probabilities, top_k, top_p))
        return probabilities.argmax(dim=-1)


class RecordingRequestLocalCategorical:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def sample(self, probabilities, seeds, positions, *, top_k=None, top_p=None):
        self.calls.append((probabilities, seeds, positions, top_k, top_p))
        return probabilities.argmin(dim=-1)


class RecordingChainSampling:
    """K1, as the sampler builds it: every row's p is uniform."""

    def __init__(self) -> None:
        self.block_sizes: list[int] = []
        self.workspace_sizes: list[tuple[int, str, int]] = []
        self.calls: list[tuple[torch.Tensor, ...]] = []
        self.warm_ups: list[dict[str, object]] = []

    def __call__(self, _device, block_size: int, _vocab_size: int) -> RecordingChainSampling:
        self.block_sizes.append(block_size)
        return self

    def workspace_size(self, batch, representation, *, length):
        self.workspace_sizes.append((batch, representation, length))
        return 64

    def probabilities(self, logits, t, k, p, *, request_indices, workspace, out_probs):
        self.calls.append((logits.clone(), t, k, p, request_indices))
        out_probs.fill_(1 / out_probs.shape[-1])

    def warm_up(self, **options) -> tuple[()]:
        self.warm_ups.append(options)
        return ()


@pytest.fixture
def sampling_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical]:
    sampling = RecordingCategoricalSampling()
    request_local = RecordingRequestLocalCategorical()
    monkeypatch.setattr(sample_module, "CategoricalSampling", lambda: sampling)
    monkeypatch.setattr(sample_module, "RequestLocalCategorical", lambda: request_local)
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda _name: nullcontext())
    return sampling, request_local


@pytest.fixture
def chain_sampling(monkeypatch: pytest.MonkeyPatch) -> RecordingChainSampling:
    operation = RecordingChainSampling()
    # K1, cti-kernels' request-local chain sampling.
    monkeypatch.setattr(ctikernels.sampling, "RequestLocalChainSampling", operation, raising=False)
    return operation


def _decode(
    slot: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    *,
    seed: int | None = None,
    position: int = 0,
    **law: object,
) -> tuple[Row, RequestSpec]:
    """A decode row and its request's sampling constants; ``law`` adds to its
    ``TokenSampling``."""

    sampling = TokenSampling(temperature, top_k, top_p, **law)
    return Row(slot, QueryKind.DECODE, 1, 2, 2, position), RequestSpec(slot, 1, 8, sampling, seed)


def _rows(*decodes: tuple[Row, RequestSpec]) -> tuple[tuple[Row, ...], tuple[RequestSpec, ...]]:
    rows, specs = zip(*decodes, strict=True)
    return rows, specs


def make_sampler(
    vocab_size: int,
    capability: DeviceCapability | None = None,
    *,
    request_local: bool = False,
) -> Sampler:
    device = torch.device("cpu")
    law = (
        RequestLocalSamplingLaw(device, vocab_size, 5, torch.bfloat16)
        if request_local
        else LegacySamplingLaw(capability)
    )
    return Sampler(
        device,
        vocab_size,
        law=law,
        processing=LogitProcessing(torch.zeros(1, 1, dtype=torch.int32), vocab_size),
        batch_capacity=16,
        logits_dtype=torch.bfloat16,
    )


@pytest.mark.parametrize(
    ("use_top_k", "use_top_p"),
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_sampler_forwards_non_greedy_sampling_arguments(
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
    use_top_k: bool,
    use_top_p: bool,
) -> None:
    sampling, request_local = sampling_runtime
    sampler = make_sampler(2, DeviceCapability(9, 0))
    top_k = torch.tensor([1, 1]) if use_top_k else None
    top_p = torch.tensor([0.8, 0.8]) if use_top_p else None
    logits = torch.tensor([[1.0, 2.0], [3.0, 1.0]], dtype=torch.float16)
    temperatures = torch.tensor([0.5, 0.7])

    result = sampler.sample(
        logits,
        BatchSamplingArgs(
            temperatures=temperatures,
            top_k=top_k,
            top_p=top_p,
        ),
    ).token_ids

    assert torch.equal(result, torch.tensor([1, 0]))
    assert len(sampling.softmax_calls) == 1
    actual_logits, actual_temperatures, enable_pdl, probabilities = sampling.softmax_calls[0]
    torch.testing.assert_close(actual_logits, logits.float())
    assert actual_temperatures is temperatures
    assert enable_pdl is True
    assert len(sampling.sample_calls) == 1
    actual_probabilities, actual_top_k, actual_top_p = sampling.sample_calls[0]
    assert actual_probabilities is probabilities
    assert actual_top_k is top_k
    assert actual_top_p is top_p
    assert request_local.calls == []


def test_sampler_greedy_path_skips_sampling_operations(
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
) -> None:
    sampling, request_local = sampling_runtime
    sampler = make_sampler(2, DeviceCapability(12, 0))

    result = sampler.sample(
        torch.tensor([[1.0, 2.0], [3.0, 1.0]]),
        BatchSamplingArgs(temperatures=None),
    ).token_ids

    assert torch.equal(result, torch.tensor([1, 0]))
    assert sampling.softmax_calls == []
    assert sampling.sample_calls == []
    assert request_local.calls == []


def test_sampler_prepares_rows_from_their_request_constants(
    monkeypatch: pytest.MonkeyPatch,
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
) -> None:
    monkeypatch.setattr(
        sample_module,
        "make_device_tensor",
        lambda data, dtype, _device: torch.tensor(data, dtype=dtype),
    )
    sampler = make_sampler(16, DeviceCapability(12, 0))
    greedy = _decode(0)
    sampled = _decode(1, 0.7, 4, 0.8)
    args = sampler.prepare(*_rows(greedy, sampled))

    torch.testing.assert_close(args.temperatures, torch.tensor([1e-6, 0.7]))
    assert torch.equal(args.top_k, torch.tensor([16, 4], dtype=torch.int32))
    torch.testing.assert_close(args.top_p, torch.tensor([1.0, 0.8]))


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        (None, False),
        (DeviceCapability(8, 9), False),
        (DeviceCapability(9, 0), True),
        (DeviceCapability(12, 0), True),
    ],
)
def test_legacy_law_resolves_pdl_from_device_capability(
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
    capability: DeviceCapability | None,
    expected: bool,
) -> None:
    assert LegacySamplingLaw(capability).enable_pdl is expected


def test_sampler_resolves_operations_and_pdl_only_at_construction(
    monkeypatch: pytest.MonkeyPatch,
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
) -> None:
    sampling, request_local = sampling_runtime

    class CapabilityProbe:
        calls = 0

        def is_at_least(self, major: int, minor: int) -> bool:
            assert (major, minor) == (9, 0)
            self.calls += 1
            return True

    capability = CapabilityProbe()
    sampler = make_sampler(2, capability)

    def unexpected_construction():
        pytest.fail("sampling operations must be constructed only once")

    monkeypatch.setattr(sample_module, "CategoricalSampling", unexpected_construction)
    monkeypatch.setattr(sample_module, "RequestLocalCategorical", unexpected_construction)

    args = BatchSamplingArgs(temperatures=torch.ones(2))
    sampler.sample(torch.ones((2, 2)), args)
    sampler.sample(torch.ones((2, 2)), args)
    seeded_args = BatchSamplingArgs(temperatures=torch.ones(2), seeds=(17, 23), positions=(0, 4))
    sampler.sample(torch.ones((2, 2)), seeded_args)
    sampler.sample(torch.ones((2, 2)), seeded_args)

    assert capability.calls == 1
    assert [call[2] for call in sampling.softmax_calls] == [True] * 4
    assert len(sampling.sample_calls) == 2
    assert len(request_local.calls) == 2


def test_sampler_preserves_request_rng_and_selected_logprob(
    monkeypatch: pytest.MonkeyPatch,
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
) -> None:
    sampling, request_local = sampling_runtime
    monkeypatch.setattr(
        sample_module,
        "make_device_tensor",
        lambda data, dtype, _device: torch.tensor(data, dtype=dtype),
    )
    sampler = make_sampler(3)
    args = sampler.prepare(
        *_rows(
            *(
                _decode(index, 0.5, 3, seed=seed, position=position, logprobs=LogprobMode.PROCESSED)
                for index, seed, position in ((0, 17, 0), (1, 23, 4))
            )
        )
    )
    logits = torch.tensor([[0.1, 0.7, -0.2], [1.2, 0.3, -0.4]])

    out = sampler.sample(logits, args)
    sampled, selected = out.token_ids, out.logprobs

    assert args.seeds == (17, 23)
    assert args.positions == (0, 4)
    assert len(sampling.softmax_calls) == 1
    # Processed rows redraw from the distribution their logprobs read, with their streams.
    assert len(request_local.calls) == 2
    probabilities, seeds, positions, top_k, top_p = request_local.calls[-1]
    torch.testing.assert_close(probabilities, sampling.softmax_calls[0][3])
    assert seeds == args.seeds
    assert positions == args.positions
    assert top_k is top_p is None
    assert sampling.sample_calls == []
    assert torch.equal(sampled, torch.tensor([2, 2]))
    expected = torch.log_softmax(logits / 0.5, dim=-1).gather(1, sampled[:, None]).squeeze(1)
    torch.testing.assert_close(selected, expected)


def test_an_unseeded_min_p_row_redraws_beside_a_seeded_row(
    monkeypatch: pytest.MonkeyPatch,
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
) -> None:
    """A seeded row and an unseeded min-p row in one step; the redraw holds no
    seeded row, so no request stream draws for it (an empty draw crashed the kernel)."""
    sampling, request_local = sampling_runtime
    monkeypatch.setattr(
        sample_module,
        "make_device_tensor",
        lambda data, dtype, _device: torch.tensor(data, dtype=dtype),
    )
    sampler = make_sampler(3)
    args = sampler.prepare(*_rows(_decode(0, 0.7, seed=7), _decode(1, 0.7, min_p=0.05)))

    sampler.sample(torch.tensor([[0.1, 0.7, -0.2], [1.2, 0.3, -0.4]]), args)

    assert args.seeds == (7, None) and args.redraw == (1,)
    assert [seeds for _, seeds, *_ in request_local.calls] == [(7,)]
    assert len(sampling.sample_calls) == 2  # the batch's shared draw, then the redraw's


def test_seeded_sampling_applies_its_filters_to_the_request_streams(
    monkeypatch: pytest.MonkeyPatch,
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
) -> None:
    _, request_local = sampling_runtime
    monkeypatch.setattr(
        sample_module,
        "make_device_tensor",
        lambda data, dtype, _device: torch.tensor(data, dtype=dtype),
    )
    sampler = make_sampler(4)
    args = sampler.prepare(*_rows(_decode(0, 0.7, 2, 0.8, seed=7)))

    sampler.sample(torch.zeros((1, 4)), args)

    (_, seeds, _, top_k, top_p) = request_local.calls[0]
    assert seeds == (7,)
    assert top_k is args.top_k and top_k.tolist() == [2]
    assert top_p is args.top_p


@pytest.mark.parametrize("filtered", [False, True])
def test_a_mixed_batch_draws_seeded_rows_from_their_own_streams(
    monkeypatch: pytest.MonkeyPatch,
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
    filtered: bool,
) -> None:
    sampling, request_local = sampling_runtime
    monkeypatch.setattr(
        sample_module,
        "make_device_tensor",
        lambda data, dtype, _device: torch.tensor(data, dtype=dtype),
    )
    sampler = make_sampler(3)
    top_k = 2 if filtered else None
    args = sampler.prepare(
        *_rows(
            *(
                _decode(row, 1.0, top_k, seed=seed, position=position)
                for row, seed, position in ((0, None, 3), (1, 17, 5), (2, None, 0), (3, 23, 9))
            )
        )
    )
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [1.0, 2.0, 0.0], [0.0, 2.0, 1.0]])

    sampled = sampler.sample(logits, args).token_ids

    assert args.seeds == (None, 17, None, 23)
    ((probabilities, seeds, positions, local_top_k, _),) = request_local.calls
    # Only the seeded rows reach their streams, with their own positions and filters.
    torch.testing.assert_close(probabilities, sampling.softmax_calls[0][3][[1, 3]])
    assert (seeds, positions) == ((17, 23), (5, 9))
    assert (local_top_k is None) is not filtered
    if filtered:
        assert local_top_k.tolist() == [2, 2]
    # The fakes: the shared generator takes the argmax, a request stream the argmin.
    assert sampled.tolist() == [0, 0, 1, 0]


def test_a_mixed_batch_keeps_per_row_filters_and_selected_logprobs(
    monkeypatch: pytest.MonkeyPatch,
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
) -> None:
    sampling, request_local = sampling_runtime
    monkeypatch.setattr(
        sample_module,
        "make_device_tensor",
        lambda data, dtype, _device: torch.tensor(data, dtype=dtype),
    )
    sampler = make_sampler(4)
    rows = ((None, 2, 0.9), (11, 3, 1.0), (None, 4, 0.7), (13, 2, 0.8))
    args = sampler.prepare(
        *_rows(
            *(
                _decode(row, 0.5 + row / 10, top_k, top_p, seed=seed, position=row)
                for row, (seed, top_k, top_p) in enumerate(rows)
            )
        )
    )

    sampler.sample(torch.zeros((4, 4)), args)

    # The seeded rows keep their own filters, gathered from the batch's.
    ((_, seeds, positions, top_k, top_p),) = request_local.calls
    assert (seeds, positions) == ((11, 13), (1, 3))
    assert top_k.tolist() == [3, 2]
    torch.testing.assert_close(top_p, torch.tensor([1.0, 0.8]))
    ((_, shared_top_k, shared_top_p),) = sampling.sample_calls
    assert shared_top_k.tolist() == [2, 3, 4, 2]
    torch.testing.assert_close(shared_top_p, torch.tensor([0.9, 1.0, 0.7, 0.8]))

    # Selected log-probabilities follow the mixed batch's final tokens.
    args = sampler.prepare(
        *_rows(
            *(
                _decode(row, 1.0, seed=seed, position=row, logprobs=LogprobMode.RAW)
                for row, seed in enumerate((None, 11, None, 13))
            )
        )
    )
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]] * 4)

    out = sampler.sample(logits, args)
    sampled, selected = out.token_ids, out.logprobs

    # The fakes: the shared generator takes the argmax, a request stream the argmin.
    assert sampled.tolist() == [0, 3, 0, 3]
    expected = torch.log_softmax(logits, dim=-1).gather(1, sampled[:, None]).squeeze(1)
    torch.testing.assert_close(selected, expected)


def test_round_filters_replace_the_provider_filters(
    monkeypatch: pytest.MonkeyPatch,
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
    chain_sampling: RecordingChainSampling,
) -> None:
    sampling, request_local = sampling_runtime
    monkeypatch.setattr(
        sample_module,
        "make_device_tensor",
        lambda data, dtype, _device: torch.tensor(data, dtype=dtype),
    )

    sampler = make_sampler(4, request_local=True)
    for seed in (7, None):
        args = sampler.prepare(*_rows(_decode(0, 0.7, 2, 0.8, seed=seed)))
        sampler.sample(torch.zeros((1, 4)), args)
    # Unfiltered: a greedy row keeps temperature 0 and a small one is not clamped.
    args = sampler.prepare(*_rows(_decode(0, 0.0), _decode(1, 1e-7)))
    sampler.sample(torch.zeros((2, 4)), args)

    calls = chain_sampling.calls
    assert [(k.tolist(), p.tolist()) for *_, k, p, _ in calls[:2]] == [
        ([2], [0.800000011920929])
    ] * 2
    _, t, k, p, _ = calls[2]
    assert (
        t.tolist() == [0.0, pytest.approx(1e-7)] and k.tolist() == [4, 4] and p.tolist() == [1, 1]
    )
    assert sampling.softmax_calls == []
    (probabilities, _, _, top_k, top_p) = request_local.calls[0]
    assert torch.equal(probabilities, torch.full((1, 4), 0.25)) and top_k is top_p is None
    assert sampling.sample_calls[0][1:] == (None, None)


def test_request_local_rows_run_unpadded_for_k1(
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
    chain_sampling: RecordingChainSampling,
) -> None:
    sampling, _ = sampling_runtime
    sampler = make_sampler(4, request_local=True)
    # K1 blocks hold 2 to 32 positions; the workspace holds the capacity's rows of one.
    assert chain_sampling.block_sizes == [2]
    assert chain_sampling.workspace_sizes == [(5, "probabilities", 1)]

    sampler.sample(
        torch.arange(12, dtype=torch.bfloat16).view(3, 4),
        BatchSamplingArgs(
            temperatures=torch.tensor([0.5, 1.0, 2.0]),
            top_k=torch.tensor([2, 4, 3], dtype=torch.int32),
        ),
    )

    ((logits, t, k, p, rows),) = chain_sampling.calls
    assert logits.shape == (3, 1, 4) and torch.equal(
        logits[:, 0].float(), torch.arange(12.0).view(3, 4)
    )
    # One row per request, no padding; an absent top-p is 1.
    assert rows.tolist() == [0, 1, 2]
    assert k.tolist() == [2, 4, 3] and p.tolist() == [1.0] * 3
    assert t.tolist() == [0.5, 1.0, 2.0]
    ((probabilities, top_k, top_p),) = sampling.sample_calls
    assert probabilities.shape == (3, 4) and top_k is top_p is None


def test_a_top_k_past_the_vocabulary_is_identity(
    monkeypatch: pytest.MonkeyPatch,
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
) -> None:
    monkeypatch.setattr(
        sample_module,
        "make_device_tensor",
        lambda data, dtype, _device: torch.tensor(data, dtype=dtype),
    )
    sampler = make_sampler(4)
    assert sampler.prepare(*_rows(_decode(0, 0.7, 2**31))).top_k is None


def test_warm_up_draws_through_every_reachable_path_at_each_bucket(
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
) -> None:
    sampling, request_local = sampling_runtime
    sampler = make_sampler(5)

    sampler.warm_up([1, 3])

    # Each bucket, without and with the top-k and top-p filters, then min-p's unfiltered draw.
    paths = [(size, k, p) for size in (1, 3) for k in (False, True) for p in (False, True)]
    paths = sorted(paths + [(1, False, False), (3, False, False)])
    shared = [(r.shape[0], k is not None, p is not None) for r, k, p in sampling.sample_calls]
    seeded = [(r.shape[0], k is not None, p is not None) for r, _, _, k, p in request_local.calls]
    assert sorted(shared) == paths
    assert sorted(seeded) == paths


def test_warm_up_processes_model_dtype_and_steered_logits_under_every_penalty(
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    apply = sample_module.LogitPenalties.apply

    def record(self, logits, rows, lookahead=None, *, out):
        penalized = bool(
            rows.presence.all() and rows.frequency.all() and (rows.repetition != 1).all()
        )
        calls.append((logits.dtype, logits.shape[0], penalized))
        apply(self, logits, rows, lookahead, out=out)

    monkeypatch.setattr(sample_module.LogitPenalties, "apply", record)
    make_sampler(5).warm_up([1, 3])

    # The largest bucket's model-dtype logits, then steered ones, which steering makes FP32.
    assert calls == [(torch.bfloat16, 3, True), (torch.float32, 3, True)]


def test_dflash_warm_up_builds_probabilities_from_model_dtype_and_processed_logits(
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
    chain_sampling: RecordingChainSampling,
) -> None:
    sampling, request_local = sampling_runtime
    sampler = make_sampler(5, request_local=True)

    sampler.warm_up([2])

    # The unfiltered path's draw at the batch size (the probabilities apply any filters in
    # the same kernels) on BF16 logits, then one K1 build per logits dtype, the model's and
    # processed FP32, covering every row count.
    assert {logits.dtype for logits, *_ in chain_sampling.calls} == {torch.bfloat16}
    calls = [
        (logits.shape[0], int(k[0]), float(p[0])) for logits, _, k, p, _ in chain_sampling.calls
    ]
    assert calls == [(2, 5, 1.0)] * 2
    assert chain_sampling.warm_ups == [
        {
            "batch_capacity": 5,
            "dtypes": (torch.bfloat16, torch.float32),
            "filters": ("auto",),
            "min_p": (False,),
            "modes": (),
            "strided_logits": False,
        }
    ]
    # The probabilities already hold the filters: the draws, min-p's too, are unfiltered.
    assert [(k, p) for *_, k, p in request_local.calls] == [(None, None)] * 2
    assert sampling.softmax_calls == [] and sampling.sample_calls == []


def test_greedy_selected_logprob_is_the_unscaled_model_logprob_of_the_argmax(
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
) -> None:
    sampling, request_local = sampling_runtime
    sampler = make_sampler(3)
    # Temperature zero and top-k one are both greedy.
    args = sampler.prepare(
        *_rows(_decode(0, logprobs=LogprobMode.RAW), _decode(1, 0.5, 1, logprobs=LogprobMode.RAW))
    )
    logits = torch.tensor([[0.1, 0.7, -0.2], [1.2, 0.3, 1.2]], dtype=torch.bfloat16)

    out = sampler.sample(logits, args)
    sampled, selected = out.token_ids, out.logprobs

    assert args.temperatures is None and args.logprobs is not None
    assert sampled.tolist() == [1, 0]
    expected = torch.log_softmax(logits.float(), dim=-1).gather(1, sampled[:, None]).squeeze(1)
    assert selected.dtype == torch.float32
    torch.testing.assert_close(selected, expected)
    assert sampling.softmax_calls == [] and request_local.calls == []


def test_selected_logprobs_read_raw_or_the_drawn_distribution(
    monkeypatch: pytest.MonkeyPatch,
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
) -> None:
    monkeypatch.setattr(
        sample_module,
        "make_device_tensor",
        lambda data, dtype, _device: torch.tensor(data, dtype=dtype),
    )
    sampler = make_sampler(4)
    args = sampler.prepare(
        *_rows(
            # Greedy rows read raw in either mode; a row that asks for none rides along.
            _decode(0, logprobs=LogprobMode.RAW),
            _decode(1, logprobs=LogprobMode.PROCESSED),
            _decode(2, 0.5, 2, logprobs=LogprobMode.RAW),
            _decode(3, 0.5, 2, 0.9, logprobs=LogprobMode.PROCESSED),
            _decode(4, 1.0),
        )
    )
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]] * 5)

    out = sampler.sample(logits, args)
    sampled, selected = out.token_ids, out.logprobs

    # The fake shared generator takes the argmax.
    assert sampled.tolist() == [0] * 5
    raw = torch.log_softmax(logits[0], -1)[0]
    # Top-2 at T=0.5 keeps p = (0.881, 0.119); top-0.9 then keeps both.
    processed = torch.log_softmax(logits[0, :2] / 0.5, -1)[0]
    torch.testing.assert_close(selected[:4], torch.stack([raw, raw, raw, processed]))


def test_top_p_keeps_the_smallest_prefix_reaching_its_mass() -> None:
    probabilities = torch.tensor([[0.5, 0.3, 0.15, 0.05], [0.25, 0.25, 0.25, 0.25]])

    filtered = SampledLogprobs.filter(probabilities, None, torch.tensor([0.8, 0.5]))

    torch.testing.assert_close(
        filtered, torch.tensor([[0.625, 0.375, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    )


def test_penalties_follow_vllm_and_count_each_positions_drafts() -> None:
    # Row 0's context: prompt [1, 2], output [2, 3]; row 1's: prompt [3], no output.
    table = torch.tensor([[1, 2, 2, 3, 0, 0], [3, 0, 0, 0, 0, 0]], dtype=torch.int32)
    rows = LogitRows(
        torch.tensor([0, 1]),
        torch.tensor([2, 1]),
        torch.tensor([4, 1]),
        torch.tensor([0.5, 0.0]),
        torch.tensor([0.25, 0.0]),
        torch.tensor([2.0, 2.0]),
    )
    logits = torch.tensor([[1.0, 2.0, -2.0, 4.0, 1.0], [1.0, 1.0, 1.0, -1.0, 1.0]])
    penalties = LogitProcessing(table, 5)

    penalized = penalties.apply(logits, rows)

    # Repetition halves positive and doubles negative logits of seen tokens; output token 2
    # (once) and 3 (once) also lose 0.25 per occurrence and 0.5 for appearing.
    expected = torch.tensor([[1.0, 1.0, -4.75, 1.25, 1.0], [1.0, 1.0, 1.0, -2.0, 1.0]])
    torch.testing.assert_close(penalized, expected)
    # A round's position i reads the context plus the drafts before it.
    drafts = torch.tensor([[4, 4], [0, 3]])
    per_position = penalties.apply(logits[:, None].repeat(1, 3, 1), rows, drafts)
    for i in range(3):
        extended = table.clone()
        for row in range(2):
            start = int(rows.lengths[row])
            extended[row, start : start + i] = drafts[row, :i].int()
        moved = LogitRows(
            *(getattr(rows, name) for name in ("rows", "prompt_lengths")),
            rows.lengths + i,
            rows.presence,
            rows.frequency,
            rows.repetition,
        )
        torch.testing.assert_close(
            per_position[:, i], LogitProcessing(extended, 5).apply(logits, moved)
        )


def test_min_tokens_masks_each_positions_stops_and_logit_bias_adds() -> None:
    # Row 0 has one output and min_tokens 3, so a round's positions 0 and 1 mask the model's
    # stop token 4 and its own 1.
    processing = LogitProcessing(torch.zeros(2, 4, dtype=torch.int32), 5, stop_token_ids=(4,))
    laws = (
        TokenSampling(min_tokens=3, stop_token_ids=(1,), logit_bias=((2, 5.0),)),
        TokenSampling(logit_bias=((0, -100.0),), stop_token_ids=(1,)),
    )
    rows = LogitRows(
        torch.tensor([0, 1]),
        torch.tensor([2, 1]),
        torch.tensor([3, 1]),
        torch.zeros(2),
        torch.zeros(2),
        torch.ones(2),
        processing.steering(laws, 2, pin_memory=False),
    )
    logits = torch.zeros(2, 3, 5)

    processed = processing.apply(logits, rows, torch.zeros(2, 2, dtype=torch.int64))

    masked, steered = [0.0, -math.inf, 5.0, 0.0, -math.inf], [0.0, 0.0, 5.0, 0.0, 0.0]
    expected = torch.tensor([[masked, masked, steered], [[-100.0, 0.0, 0.0, 0.0, 0.0]] * 3])
    torch.testing.assert_close(processed, expected)
    torch.testing.assert_close(processing.apply(logits[:, 0], rows), expected[:, 0])


def test_min_tokens_masks_the_models_stops_even_for_a_row_that_ignores_eos() -> None:
    # vLLM's: ignore_eos stops ending at EOS, not min_tokens' mask of it.
    processing = LogitProcessing(torch.zeros(1, 4, dtype=torch.int32), 5, stop_token_ids=(4,))
    law = TokenSampling(min_tokens=3, stop_token_ids=(1,), ignore_eos=True)
    logits = torch.zeros(1, 1, 5)

    processing.steering((law,), 1, pin_memory=False).apply_(
        logits, torch.zeros(1, dtype=torch.int64)
    )

    assert logits.isneginf().tolist() == [[[False, True, False, False, True]]]


def test_one_requests_positions_steer_each_after_its_outputs() -> None:
    # One request of a round's three positions: its rows to the op are three copies of
    # request 0 (contiguous, as the op requires), position i after i more outputs.
    processing = LogitProcessing(torch.zeros(1, 4, dtype=torch.int32), 5, stop_token_ids=(4,))
    law = TokenSampling(min_tokens=2, logit_bias=((1, 0.5),))
    logits = torch.zeros(1, 3, 5)

    processing.steering((law,), 1, pin_memory=False).apply_(
        logits, torch.zeros(1, dtype=torch.int64)
    )

    masked, steered = [0.0, 0.5, 0.0, 0.0, -math.inf], [0.0, 0.5, 0.0, 0.0, 0.0]
    assert logits[0].tolist() == [masked, masked, steered]


def test_requests_past_the_staged_ones_steer_nothing() -> None:
    # A round's graph steers its bucket's padding rows too.
    processing = LogitProcessing(torch.zeros(3, 4, dtype=torch.int32), 5, stop_token_ids=(4,))
    law = TokenSampling(min_tokens=2, logit_bias=((0, 1.5),))
    logits = torch.zeros(3, 2, 5)

    processing.steering((law,), 3, pin_memory=False).apply_(
        logits, torch.zeros(3, dtype=torch.int64)
    )

    assert logits[0].tolist() == [[1.5, 0.0, 0.0, 0.0, -math.inf]] * 2
    assert not logits[1:].any()


def test_min_p_keeps_tokens_near_the_likeliest() -> None:
    probabilities = torch.tensor([[0.5, 0.3, 0.15, 0.05]])

    filtered = SampledLogprobs.filter(probabilities, None, None, torch.tensor([0.4]))

    torch.testing.assert_close(filtered, torch.tensor([[0.625, 0.375, 0.0, 0.0]]))


def test_top_logprobs_rank_the_read_distribution(
    monkeypatch: pytest.MonkeyPatch,
    sampling_runtime: tuple[RecordingCategoricalSampling, RecordingRequestLocalCategorical],
) -> None:
    monkeypatch.setattr(
        sample_module,
        "make_device_tensor",
        lambda data, dtype, _device: torch.tensor(data, dtype=dtype),
    )
    sampler = make_sampler(4)
    logits = torch.tensor([[0.0, 3.0, 1.0, 2.0]] * 2)

    rows = _rows(
        _decode(0, logprobs=LogprobMode.RAW, top_logprobs=2), _decode(1, logprobs=LogprobMode.RAW)
    )
    out = sampler.sample(logits, sampler.prepare(*rows))

    raw = torch.log_softmax(logits[0], -1)
    assert out.top_ids.tolist() == [[1, 3], [1, 3]]
    torch.testing.assert_close(out.top_logprobs[0], raw[[1, 3]])
    torch.testing.assert_close(out.logprobs, raw[[1, 1]])


@pytest.mark.parametrize(
    ("law", "message"),
    [
        ({"temperature": -0.5}, "temperature must be >= 0"),
        ({"top_p": 1.5}, "top_p must be in"),
        ({"top_k": 0}, "top_k must be"),
        ({"temperature": 1.0, "top_k": -1}, "top_k must be None or a positive integer"),
        ({"min_p": 1.5}, "min_p must be in"),
        ({"temperature": 0.0, "top_k": 5}, "top_k must be None when temperature is 0"),
        ({"temperature": 0.0, "top_p": 0.9}, "top_p must be 1 when temperature is 0"),
    ],
)
def test_the_sampling_law_refuses_values_outside_its_ranges(law: dict, message: str) -> None:
    with pytest.raises(SamplingError, match=message):
        TokenSampling(**law)
    with pytest.raises(SamplingError, match=message):
        SamplingParams(max_new_tokens=8, **law).snapshot()


def test_sampling_params_require_max_new_tokens_and_default_to_temperature_one() -> None:
    with pytest.raises(TypeError, match="max_new_tokens"):
        SamplingParams(temperature=0)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="max_tokens"):
        SamplingParams(max_new_tokens=8, max_tokens=8)  # type: ignore[call-arg]
    law = SamplingParams(max_new_tokens=8).snapshot().law
    assert (law.temperature, law.top_k, law.top_p, law.stop_token_ids) == (1.0, None, 1.0, ())


@pytest.mark.parametrize(
    ("flags", "mode"),
    [
        ({}, None),
        ({"logprobs": True}, LogprobMode.RAW),
        ({"logprobs": True, "logprobs_processed": True}, LogprobMode.PROCESSED),
    ],
)
def test_logprob_flags_snapshot_to_their_mode(flags: dict, mode: LogprobMode | None) -> None:
    assert SamplingParams(max_new_tokens=1, **flags).snapshot().law.logprobs is mode


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ({"logprobs_processed": True}, "logprobs_processed requires logprobs"),
        ({"logprobs": "raw"}, "logprobs must be a boolean"),
        ({"logprobs": True, "logprobs_processed": 1}, "logprobs_processed must be a boolean"),
    ],
)
def test_logprob_flags_refuse_what_they_cannot_read(flags: dict, message: str) -> None:
    with pytest.raises(SamplingError, match=message):
        SamplingParams(max_new_tokens=1, **flags).snapshot()


def test_greedy_is_temperature_zero_or_top_k_one() -> None:
    assert TokenSampling(0.0).is_greedy and TokenSampling(0.7, 1).is_greedy
    assert not TokenSampling(1e-7).is_greedy
