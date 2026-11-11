"""A request's own stops: stop token ids and stop strings, decided where its tokens settle."""

from __future__ import annotations

import random

import pytest
import torch
from tokenizers import Tokenizer, decoders
from tokenizers.models import WordLevel, WordPiece
from transformers import PreTrainedTokenizerFast

from cti.inference._control._scheduler import InferenceScheduler
from cti.inference._control._scheduling.encoder_cache import EncoderCache
from cti.inference._control._scheduling.stops import RequestStops, StopHit
from cti.inference._control._scheduling.strategy import SchedulingLimits, TokenSteps
from cti.protocol import (
    DetokenizeMsg,
    RejectionKind,
    RequestEnd,
    SamplingParams,
    TokenUsage,
    UserMsg,
)
from cti.tokenizer.detokenize import IncrementalDetokenizer
from cti.tokenizer.text import TextStream, TokenizerTextDecoder
from tests.support.inference.cache import new_cache_manager
from tests.support.inference.scheduling import settle_forward

# Byte-level pieces, as a BPE vocabulary has them: 7 and 8 are the two bytes of "é".
_PIECES = {
    1: b"He",
    2: b"llo",
    3: b" wor",
    4: b"ld",
    5: b"\n",
    6: b"\n\n",
    7: "é".encode()[:1],
    8: "é".encode()[1:],
    9: b"!",
    # The second byte of one "é" and the first of the next: no token boundary is a
    # character boundary in 7, 11, 11, ...
    11: "é".encode()[1:] + "é".encode()[:1],
    # A byte no character contains: it decodes to U+FFFD for good.
    12: b"\xff",
    50: b"<sp>",
}
_SPECIAL = 50
_EOS = 99


class _ByteDecoder:
    prefix_stable = True

    def stream(self, *, skip_special_tokens: bool) -> TextStream:
        return TextStream(self, skip_special_tokens=skip_special_tokens)

    def __init__(self) -> None:
        self.windows: list[list[int]] = []
        self.calls = 0

    def renders(self, token_id: int, *, skip_special_tokens: bool) -> bool:
        return token_id in _PIECES and not (skip_special_tokens and token_id == _SPECIAL)

    def batch_decode(self, sequences: list[list[int]], *, skip_special_tokens: bool) -> list[str]:
        self.windows += sequences
        self.calls += 1
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in sequences]

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        shown = (t for t in ids if self.renders(t, skip_special_tokens=skip_special_tokens))
        return b"".join(_PIECES[t] for t in shown).decode("utf-8", errors="replace")


def _stops(decoder: _ByteDecoder | None = None, **params: object) -> RequestStops:
    return RequestStops(
        SamplingParams(max_new_tokens=1024, temperature=0, **params).snapshot(),
        decoder or _ByteDecoder(),
    )


def _feed(stops: RequestStops, ids: list[int], chunks: list[int]) -> tuple[int, str | None] | None:
    """The first hit feeding ``ids`` in chunks of the given sizes, as an index into ids."""

    start = 0
    for size in chunks:
        if (hit := stops.find(ids[start : start + size])) is not None:
            return start + hit.index, hit.stop_str
        start += size
    return None


def _reference(
    ids: list[int], stop: tuple[str, ...], stop_token_ids: tuple[int, ...], skip: bool
) -> tuple[int, str | None] | None:
    """The first token whose complete-character text holds a stop string, by full decodes."""

    for index, token in enumerate(ids):
        if token in stop_token_ids:
            return index, None
        text = _ByteDecoder().decode(ids[: index + 1], skip_special_tokens=skip).rstrip("\ufffd")
        found = [(text.find(s) + len(s), -len(s), s) for s in stop if s in text]
        if found:
            return index, min(found)[2]
    return None


def test_a_stop_string_ends_at_the_token_completing_it_across_tokens() -> None:
    assert _stops(stop=("lo w",)).find([1, 2, 3, 4]) == StopHit(2, "lo w")
    # Fed one token at a time, the earlier tokens hold only a prefix of it.
    assert _feed(_stops(stop=("lo w",)), [1, 2, 3, 4], [1, 1, 1, 1]) == (2, "lo w")


def test_a_character_split_across_tokens_matches_once_complete() -> None:
    stops = _stops(stop=("é!",))

    assert stops.find([7]) is None
    assert stops.find([8]) is None
    assert stops.find([9]) == StopHit(0, "é!")


def test_the_stop_ending_first_wins_and_the_longest_on_a_tie() -> None:
    # "Hello world": "lo" ends at 5, "Hello" at 5 too, "ello w" at 7.
    assert _stops(stop=("ello w", "lo", "Hello")).find([1, 2, 3]) == StopHit(1, "Hello")


def test_a_stop_token_id_ends_the_output_at_its_token() -> None:
    stops = RequestStops(
        SamplingParams(max_new_tokens=1024, temperature=0, stop_token_ids=(4,)).snapshot(), None
    )

    assert stops.find([1, 2]) is None
    assert stops.find([3, 4, 9]) == StopHit(1, None)
    assert _stops(stop=("ld",), stop_token_ids=(4,)).find([1, 2, 3, 4]) == StopHit(3, None)


def test_special_tokens_match_only_when_rendered() -> None:
    assert _stops(stop=("<sp>",)).find([1, _SPECIAL, 2]) is None
    assert _stops(stop=("<sp>",), skip_special_tokens=False).find([1, _SPECIAL]) == StopHit(
        1, "<sp>"
    )


def test_skipped_special_tokens_stay_out_of_the_decode_windows() -> None:
    decoder = _ByteDecoder()
    stops = RequestStops(
        SamplingParams(max_new_tokens=1024, temperature=0, stop=("x",)).snapshot(), decoder
    )
    for token in [1, *[_SPECIAL] * 20, 2]:
        assert stops.find([token]) is None

    assert max(map(len, decoder.windows)) <= 2


@pytest.mark.parametrize("seed", range(300))
def test_chunked_matching_agrees_with_full_decodes(seed: int) -> None:
    rng = random.Random(seed)
    # Runs of broken bytes long enough to settle and compress, as well as short ones.
    ids = [
        token
        for _ in range(rng.randint(1, 10))
        for token in [rng.choice(list(_PIECES))] * rng.choice([1, 1, 1, 6])
    ]
    stop = tuple(
        rng.choice(
            ["lo", "\n", "\n\n", "é", "é!", "wor", "ld\n", "<sp>", "He", "o w"]
            + ["!\ufffd", "\ufffd\ufffd", "\ufffd!", "é\ufffd", "\ufffd\ufffd!"]
        )
        for _ in range(rng.randint(1, 3))
    )
    stop_token_ids = tuple(rng.sample([4, 9], rng.randint(0, 1)))
    skip = rng.random() < 0.7
    chunks = []
    while sum(chunks) < len(ids):
        chunks.append(rng.randint(1, 4))
    params = {"stop": stop, "stop_token_ids": stop_token_ids, "skip_special_tokens": skip}

    assert _feed(_stops(**params), ids, chunks) == _reference(ids, stop, stop_token_ids, skip)


def test_stop_strings_need_a_decoder() -> None:
    assert (
        RequestStops.of(SamplingParams(max_new_tokens=1024, temperature=0).snapshot(), None) is None
    )
    assert (
        RequestStops.of(
            SamplingParams(max_new_tokens=1024, temperature=0, stop_token_ids=(3,)).snapshot(), None
        )
        is not None
    )
    with pytest.raises(ValueError, match="text decoder"):
        RequestStops.of(
            SamplingParams(max_new_tokens=1024, temperature=0, stop=("x",)).snapshot(), None
        )


def _scheduler(decoder: _ByteDecoder | None = None) -> InferenceScheduler:
    return InferenceScheduler(
        checks=True,
        limits=SchedulingLimits(
            max_running_requests=2,
            max_batch_tokens=16,
            max_sequence_length=32,
            max_prompt_tokens=16,
        ),
        cache_manager=new_cache_manager(
            64, 1, torch.full((2, 32), -1, dtype=torch.int32), reuse=False, checks=True
        ),
        stop_token_ids=(_EOS,),
        vocab_size=1 << 20,
        decode=TokenSteps(),
        encoder_cache=EncoderCache(0),
        text_decoder=decoder,
    )


def _message(uid: int, **sampling: object) -> UserMsg:
    return UserMsg(
        uid=uid,
        input_ids=torch.arange(10, 14, dtype=torch.int32),
        sampling=SamplingParams(max_new_tokens=8, temperature=0, **sampling).snapshot(),
    )


def _step(scheduler: InferenceScheduler, tokens: list[int]) -> list[DetokenizeMsg]:
    plan = scheduler.plan_next()
    scheduler.mark_submitted(plan)
    outcome = settle_forward(scheduler, plan, torch.tensor(tokens, dtype=torch.int32))
    return outcome.replies


def test_a_stop_string_finishes_its_request_at_the_completing_token() -> None:
    scheduler = _scheduler(_ByteDecoder())
    assert scheduler.handle_message(_message(0, stop=("wor",))) == ()

    replies = [reply for token in (1, 2, 3) for reply in _step(scheduler, [token])]

    *_, last = replies
    assert [reply.token_ids for reply in replies] == [(1,), (2,), (3,)]
    assert (last.end.is_stop_token, last.end.stop_str) == (False, "wor")
    assert last.end.finish_reason == "stop" and last.end.usage.completion_tokens == 3
    scheduler.require_quiescent()


def test_a_stop_token_id_ends_a_request_that_ignores_eos() -> None:
    scheduler = _scheduler()
    assert scheduler.handle_message(_message(0, ignore_eos=True, stop_token_ids=(4,))) == ()

    replies = [reply for token in (_EOS, 4) for reply in _step(scheduler, [token])]

    assert [reply.end for reply in replies[:1]] == [None]
    assert replies[1].end.is_stop_token
    scheduler.require_quiescent()


def test_stop_strings_without_a_decoder_reject_the_request() -> None:
    (rejection,) = _scheduler().handle_message(_message(0, stop=("x",)))

    assert rejection.kind is RejectionKind.INVALID_REQUEST


def _wordpiece_decoder() -> TokenizerTextDecoder:
    """A WordPiece tokenizer whose decode cleans up spaces, " n't" into "n't"."""

    vocab = ["[UNK]", "do", "n", "##'", "##t", "it", "'", "##s", ",", "x", "!"]
    backend = Tokenizer(WordPiece({piece: i for i, piece in enumerate(vocab)}, unk_token="[UNK]"))
    backend.decoder = decoders.WordPiece()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", clean_up_tokenization_spaces=True
    )
    assert tokenizer.decode([1, 2, 3, 4]) == "don't"
    return TokenizerTextDecoder(tokenizer)


def test_a_stop_string_matches_the_output_as_it_streams_not_a_cleaned_up_one() -> None:
    decoder = _wordpiece_decoder()
    params = SamplingParams(max_new_tokens=1024, temperature=0, stop=("do n't",)).snapshot()

    assert decoder.batch_decode([[1, 2, 3, 4]], skip_special_tokens=True) == ["do n't"]
    assert RequestStops(params, decoder).find([1, 2, 3, 4]) == StopHit(3, "do n't")
    assert _feed(RequestStops(params, decoder), [1, 2, 3, 4], [1, 1, 1, 1]) == (3, "do n't")


@pytest.mark.parametrize("seed", range(100))
def test_chunked_matching_agrees_with_full_decodes_of_a_real_tokenizer(seed: int) -> None:
    decoder = _wordpiece_decoder()
    rng = random.Random(seed)
    # Words whose cleanup would rewrite " n't", " 's", " ," and " !".
    words = [[1, 2, 3, 4], [5, 6, 7], [9, 8], [9, 10], [2, 3, 4], [1], [9], [6]]
    ids = [token for _ in range(rng.randint(1, 6)) for token in rng.choice(words)]
    candidates = ["n't", "don't", "do n't", "it's", "it 's", "x,", "x ,", "x!", "x !"]
    stop = tuple(rng.sample(candidates, rng.randint(1, 3)))
    chunks = []
    while sum(chunks) < len(ids):
        chunks.append(rng.randint(1, 4))
    expected = None
    for index in range(len(ids)):
        (text,) = decoder.batch_decode([ids[: index + 1]], skip_special_tokens=True)
        found = [(text.find(s) + len(s), -len(s), s) for s in stop if s in text]
        if found:
            expected = (index, min(found)[2])
            break

    assert (
        _feed(
            RequestStops(
                SamplingParams(max_new_tokens=1024, temperature=0, stop=stop).snapshot(), decoder
            ),
            ids,
            chunks,
        )
        == expected
    )


def test_matching_keeps_bounded_state_over_a_long_output() -> None:
    decoder = _ByteDecoder()
    stops = _stops(decoder, stop=("never", "\n\n\n"))

    for token in [1, 2, 3, 4, 5, 9, 7, 8] * 500:
        assert stops.find([token]) is None
    decoder.windows.clear()

    assert stops.find([1]) is None
    assert max(map(len, decoder.windows)) <= 3


@pytest.mark.parametrize("chunk", [1, 3])
def test_tokens_that_all_end_inside_a_character_still_match_in_place(chunk: int) -> None:
    ids = [7, *[11] * 40, 8, 9]  # "é" * 41 + "!"
    stop = ("é" * 20, "é!")

    # 11 at index 20 completes the 20th "é"; "é!" ends the text at index 42.
    assert _feed(_stops(stop=stop), ids, [chunk] * len(ids)) == (20, "é" * 20)
    assert _feed(_stops(stop=stop[1:]), ids, [chunk] * len(ids)) == (42, "é!")


def test_a_step_decodes_its_tokens_once_unless_a_stop_string_completes() -> None:
    decoder = _ByteDecoder()
    stops = RequestStops(
        SamplingParams(max_new_tokens=1024, temperature=0, stop=("ld!",)).snapshot(), decoder
    )

    assert stops.find([1, 2, 3, 5, 9, 1, 2, 3]) is None
    assert decoder.calls == 1
    # Only a match decodes the tokens one by one, to find the one completing it.
    assert stops.find([2, 4, 9, 5]) == StopHit(2, "ld!")
    assert decoder.calls == 3


def test_a_run_of_broken_bytes_is_committed_and_stays_bounded() -> None:
    decoder = _ByteDecoder()
    stops = RequestStops(
        SamplingParams(max_new_tokens=1024, temperature=0, stop=("!",)).snapshot(), decoder
    )

    for _ in range(4096):
        assert stops.find([12]) is None
    decoder.windows.clear()

    assert stops.find([9]) == StopHit(0, "!")
    assert max(map(len, decoder.windows)) <= 3 * TextStream.SETTLING_TOKENS


@pytest.mark.parametrize("chunk", [1, 2, 5])
def test_broken_bytes_before_a_stop_match_like_full_decodes(chunk: int) -> None:
    ids = [12] * 9 + [7, 12, 8, 1, 2, 12, 12, 9]
    stop = ("\ufffdHello",)

    assert _feed(_stops(stop=stop), ids, [chunk] * len(ids)) == _reference(ids, stop, (), True)


@pytest.mark.parametrize("chunk", [1, 3, 12])
def test_a_replacement_character_counts_once_a_complete_character_follows(chunk: int) -> None:
    ids = [9, *[12] * 10, 9]  # "!" + 10 broken bytes + "!"
    stop = ("!\ufffd", "\ufffd!")

    # However the steps group the tokens: "!\ufffd" completes only once "!" follows.
    assert _feed(_stops(stop=stop), ids, [chunk] * len(ids)) == (11, "!\ufffd")
    assert _reference(ids, stop, (), True) == (11, "!\ufffd")


def test_a_long_run_of_replacement_characters_is_kept_bounded() -> None:
    decoder = _ByteDecoder()
    stops = _stops(decoder, stop=("x\ufffd\ufffd!",))
    for token in [9, *[12] * 4096]:
        assert stops.find([token]) is None
    decoder.windows.clear()

    # A run longer than the stop string cannot complete it.
    assert stops.find([9]) is None
    assert max(map(len, decoder.windows)) <= 9


def _byte_fallback_decoder() -> TokenizerTextDecoder:
    """A byte-fallback tokenizer: a run of byte tokens decodes together, so a broken byte
    turns the whole run into replacement characters, an earlier complete one included."""

    vocab = {"<unk>": 0, "<0xC3>": 1, "<0xA9>": 2, "<0xFF>": 3, "!": 4}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    assert tokenizer.decode([1, 2]) == "é" and tokenizer.decode([1, 2, 3]) == "\ufffd" * 3
    return TokenizerTextDecoder(tokenizer)


def test_a_decoder_that_revises_text_is_matched_token_by_token() -> None:
    decoder = _byte_fallback_decoder()
    assert not decoder.prefix_stable

    # "é" shows once its second byte arrives, even if a broken byte in the same step
    # later turns the run into replacement characters.
    assert RequestStops(
        SamplingParams(max_new_tokens=1024, temperature=0, stop=("é",)).snapshot(), decoder
    ).find([1, 2, 3]) == StopHit(1, "é")
    assert RequestStops(
        SamplingParams(max_new_tokens=1024, temperature=0, stop=("\ufffd",)).snapshot(), decoder
    ).find([1, 2, 3, 4]) == StopHit(3, "\ufffd")
    # A run of byte tokens is never committed early, so later bytes still complete it.
    three = RequestStops(
        SamplingParams(max_new_tokens=1024, temperature=0, stop=("ééé",)).snapshot(), decoder
    )
    assert three.find([1, 2, 1, 2, 1]) is None
    assert three.find([2]) == StopHit(0, "ééé")


def _rendered(decoder: TokenizerTextDecoder, tokens: list[int], batches: list[int]) -> str:
    """The detokenizer's text of ``tokens``, delivered as steps of the given sizes, the last
    finishing the request."""

    manager = IncrementalDetokenizer(decoder)
    end = RequestEnd(TokenUsage(prompt_tokens=1, completion_tokens=len(tokens)), len(tokens))
    texts = []
    start = 0
    for size in batches:
        step = tuple(tokens[start : start + size])
        start += size
        if step:
            last = start >= len(tokens)
            texts += manager.detokenize(
                [DetokenizeMsg(uid=0, token_ids=step, end=end if last else None)]
            )
    return "".join(texts)


_REVISING_DECODERS = {
    # Byte fallback regroups a run of byte tokens.
    "byte-fallback": (
        decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()]),
        {"<0xC3>": 1, "<0xA9>": 2, "<0xFF>": 3, "!": 4},
    ),
    # A step after byte-level decoding hides U+FFFD.
    "byte-level-masked": (
        decoders.Sequence([decoders.ByteLevel(), decoders.Replace("\ufffd", "x")]),
        {"\u00c3": 1, "\u00a9": 2, "\u00ff": 3, "!": 4},
    ),
    # A multi-character replacement acts across tokens, revising text already shown.
    "cross-token-replace": (
        decoders.Sequence([decoders.Fuse(), decoders.Replace("A \ufffdB", "C")]),
        {"A \ufffd": 1, "B": 2, "\ufffd": 3, "!": 4},
    ),
    # A suffix inside a token reads differently whether or not the token is the last.
    "bpe-inner-suffix": (decoders.BPEDecoder(), {"a</w>b": 1, "c": 2, "é": 3, "!": 4}),
    # An empty token adds no byte, so token counts tell nothing about pending characters.
    "byte-level-empty-token": (
        decoders.ByteLevel(),
        {"\u00c3": 1, "": 2, "\u00a9": 3, "!": 4},
    ),
}


def _revising_decoder(name: str) -> TokenizerTextDecoder:
    decoder, vocab = _REVISING_DECODERS[name]
    backend = Tokenizer(WordLevel({"<unk>": 0, **vocab}, unk_token="<unk>"))
    backend.decoder = decoder
    text_decoder = TokenizerTextDecoder(
        PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    )
    assert not text_decoder.prefix_stable
    return text_decoder


@pytest.mark.parametrize(
    ("name", "tokens", "stop", "index"),
    [
        ("bpe-inner-suffix", [1, 2], "ab", 0),
        ("byte-level-empty-token", [1, 2, 2, 2, 2, 2, 3, 4], "é", 6),
    ],
)
def test_reported_decoder_cases_match_their_reference_for_any_grouping(
    name: str, tokens: list[int], stop: str, index: int
) -> None:
    decoder = _revising_decoder(name)

    for groups in ([1] * len(tokens), [3] * 3, [len(tokens)]):
        assert _feed(
            RequestStops(
                SamplingParams(max_new_tokens=1024, temperature=0, stop=(stop,)).snapshot(), decoder
            ),
            tokens,
            groups,
        ) == (
            index,
            stop,
        )
        assert _rendered(decoder, tokens, groups) == _rendered(decoder, tokens, [1] * len(tokens))
    assert stop in _rendered(decoder, tokens[: index + 1], [index + 1])


class _Recording:
    """A decoder that records the longest window it decodes."""

    def __init__(self, decoder: TokenizerTextDecoder) -> None:
        self._decoder = decoder
        self.prefix_stable = decoder.prefix_stable
        self.longest = 0

    def renders(self, token_id: int, *, skip_special_tokens: bool) -> bool:
        return self._decoder.renders(token_id, skip_special_tokens=skip_special_tokens)

    def stream(self, *, skip_special_tokens: bool) -> TextStream:
        return TextStream(self, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, sequences: list[list[int]], *, skip_special_tokens: bool) -> list[str]:
        self.longest = max([self.longest, *map(len, sequences)])
        return self._decoder.batch_decode(sequences, skip_special_tokens=skip_special_tokens)


@pytest.mark.parametrize("groups", [1, 3, 64], ids=["1", "3", "64"])
def test_a_revising_decoder_keeps_a_long_broken_run_bounded_and_agreeing(groups: int) -> None:
    decoder = _Recording(_revising_decoder("byte-fallback"))
    # A broken run of 60 bytes, "!" ending it, then "é" and "!": the run is committed in
    # bounded pieces. (Byte fallback decodes a run of byte tokens as one group.)
    tokens = [1, *[3] * 60, 4, 1, 2, 4]
    stops = RequestStops(
        SamplingParams(max_new_tokens=1024, temperature=0, stop=("é!",)).snapshot(), decoder
    )

    hit = _feed(stops, tokens, [groups] * len(tokens))

    rendered = _rendered(decoder, tokens, [groups] * len(tokens))
    assert rendered == _rendered(decoder, tokens, [1] * len(tokens))
    assert rendered == IncrementalDetokenizer.render_text(decoder, tokens, skip_special_tokens=True)
    assert hit == (len(tokens) - 1, "é!") and rendered.endswith("é!")
    assert decoder.longest <= 2 * 8 + 1


@pytest.mark.parametrize("groups", [1, 3, 64], ids=["1", "3", "64"])
def test_a_revising_decoder_keeps_a_long_run_of_empty_tokens_bounded_and_agreeing(
    groups: int,
) -> None:
    decoder = _Recording(_revising_decoder("byte-level-empty-token"))
    # "!", 100 tokens that show nothing, then "é!": the empty run is committed in bounded
    # pieces.
    tokens = [4, *[2] * 100, 1, 3, 4]
    stops = RequestStops(
        SamplingParams(max_new_tokens=1024, temperature=0, stop=("é!",)).snapshot(), decoder
    )

    hit = _feed(stops, tokens, [groups] * len(tokens))

    rendered = _rendered(decoder, tokens, [groups] * len(tokens))
    assert rendered == _rendered(decoder, tokens, [1] * len(tokens))
    assert (
        rendered
        == IncrementalDetokenizer.render_text(decoder, tokens, skip_special_tokens=True)
        == "!é!"
    )
    assert hit == (len(tokens) - 1, "é!")
    assert decoder.longest <= 2 * 8 + 1


def _streamed(decoder: TokenizerTextDecoder, tokens: list[int], batches: list[int]) -> str:
    """What the detokenizer streams of ``tokens`` before the request finishes."""

    manager = IncrementalDetokenizer(decoder)
    texts = []
    start = 0
    for size in batches:
        step = tuple(tokens[start : start + size])
        start += size
        if step:
            texts += manager.detokenize([DetokenizeMsg(uid=0, token_ids=step)])
    return "".join(texts)


@pytest.mark.parametrize("groups", [1, 3, 64], ids=["1", "3", "64"])
@pytest.mark.parametrize(
    ("stop", "index"),
    [("\ufffd", 7), ("\ufffd" * 10, 15), ("\ufffd!", 20)],
    ids=["one", "spanning", "then-text"],
)
def test_replacement_characters_count_once_a_revising_decoder_commits_them(
    groups: int, stop: str, index: int
) -> None:
    decoder = _revising_decoder("byte-fallback")
    # 20 broken bytes then "!": every 8 pending ids are committed, replacement characters
    # final and shown.
    tokens = [3] * 20 + [4]
    batches = [groups] * len(tokens)

    hit = _feed(
        RequestStops(
            SamplingParams(max_new_tokens=1024, temperature=0, stop=(stop,)).snapshot(), decoder
        ),
        tokens,
        batches,
    )

    assert hit == (index, stop)
    # Nothing streamed before the matched token holds the stop; offline text cuts the same.
    assert stop not in _streamed(decoder, tokens[:index], batches)
    online = _rendered(decoder, tokens[: index + 1], batches)
    assert online == IncrementalDetokenizer.render_text(
        decoder, tokens[: index + 1], skip_special_tokens=True
    )
    assert stop in online


@pytest.mark.parametrize("name", _REVISING_DECODERS)
@pytest.mark.parametrize("seed", range(80))
def test_matching_and_rendering_agree_however_tokens_are_grouped(name: str, seed: int) -> None:
    decoder = _revising_decoder(name)
    rng = random.Random(seed)
    tokens = [rng.choice([1, 2, 3, 4]) for _ in range(rng.randint(1, 12))]
    candidates = [
        "é",
        "é\ufffd",
        "ééé",
        "\ufffd!",
        "é!",
        "!é",
        "x",
        "C",
        "x!",
        "A ",
        "C!",
        "ab",
        "a ",
    ]
    stop = tuple(rng.sample(candidates, rng.randint(1, 3)))

    def groups(count: int) -> list[int]:
        sizes = []
        while sum(sizes) < count:
            sizes.append(rng.randint(1, 4))
        return sizes

    hit = _feed(
        RequestStops(
            SamplingParams(max_new_tokens=1024, temperature=0, stop=stop).snapshot(), decoder
        ),
        tokens,
        groups(len(tokens)),
    )

    # InferenceScheduler steps and detokenizer batches may group tokens differently.
    assert hit == _feed(
        RequestStops(
            SamplingParams(max_new_tokens=1024, temperature=0, stop=stop).snapshot(), decoder
        ),
        tokens,
        [1] * 12,
    )
    rendered = _rendered(decoder, tokens, [1] * len(tokens))
    assert _rendered(decoder, tokens, groups(len(tokens))) == rendered
    # Offline text renders as the detokenizer streams it.
    assert IncrementalDetokenizer.render_text(decoder, tokens, skip_special_tokens=True) == rendered
    if hit is not None:
        # What the client sees of the stopped request holds the stop string.
        index, stop_str = hit
        assert stop_str in _rendered(decoder, tokens[: index + 1], groups(index + 1))


@pytest.mark.parametrize("groups", [[1] * 8, [3, 3, 2], [8]], ids=["1", "3", "all"])
def test_ids_outside_the_vocabulary_do_not_settle_a_pending_character(groups: list[int]) -> None:
    alphabet = _byte_level_alphabet()
    vocab = {"<unk>": 0, alphabet[0xC3]: 1, alphabet[0xA9]: 2, "!": 3}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.decoder = decoders.ByteLevel()
    decoder = TokenizerTextDecoder(
        PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    )
    assert decoder.prefix_stable
    # 77 is no token, as a model's padded vocabulary rows are not: it renders nothing.
    tokens = [1, *[77] * 5, 2, 3]

    assert _feed(
        RequestStops(
            SamplingParams(max_new_tokens=1024, temperature=0, stop=("é",)).snapshot(), decoder
        ),
        tokens,
        groups,
    ) == (
        6,
        "é",
    )
    assert _rendered(decoder, tokens, groups) == "é!"


def _byte_level_alphabet() -> dict[int, str]:
    """Byte-level BPE's printable stand-in for each byte: printable Latin-1 bytes stand for
    themselves, the others for the characters from U+0100 on."""

    printable = [*range(ord("!"), ord("~") + 1), *range(0xA1, 0xAD), *range(0xAE, 0x100)]
    others = [byte for byte in range(256) if byte not in printable]
    return {**{b: chr(b) for b in printable}, **{b: chr(0x100 + n) for n, b in enumerate(others)}}


# Byte pieces: ASCII, whole and split multi-byte characters, and bytes no character holds.
_BYTE_PIECES = tuple(
    sorted(
        {
            b"a",
            b"!",
            b" ",
            b"ab",
            "é".encode(),
            "é".encode()[:1],
            "é".encode()[1:],
            "中".encode()[:2],
            "中".encode()[2:],
            *(bytes([byte]) for byte in "😀".encode()),
            "😀".encode()[1:3],
            b"\xff",
            "é".encode()[1:] + b"a",
        }
    )
)


def _byte_level_decoder(rng: random.Random) -> tuple[TokenizerTextDecoder, list[int]]:
    """A real byte-level BPE tokenizer over a random vocabulary of byte pieces."""

    alphabet = _byte_level_alphabet()
    pieces = rng.sample(_BYTE_PIECES, rng.randint(8, len(_BYTE_PIECES)))
    vocab = {
        "<unk>": 0,
        **{"".join(alphabet[b] for b in piece): i + 1 for i, piece in enumerate(pieces)},
    }
    assert len(vocab) == len(pieces) + 1  # every id a real token
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.decoder = decoders.ByteLevel()
    decoder = TokenizerTextDecoder(
        PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    )
    assert decoder.prefix_stable
    return decoder, list(range(1, len(vocab)))


@pytest.mark.parametrize("seed", range(400))
def test_byte_level_matching_follows_the_full_prefix_reference_for_any_grouping(seed: int) -> None:
    rng = random.Random(seed)
    decoder, ids = _byte_level_decoder(rng)
    # Ids the vocabulary does not assign, which a padded model vocabulary may sample.
    unassigned = [len(ids) + 1, len(ids) + 50]
    tokens = [
        token
        for _ in range(rng.randint(1, 16))
        for token in [rng.choice(ids + unassigned)] * rng.choice([1, 1, 1, 5])
    ]

    def text(prefix: list[int]) -> str:
        (decoded,) = decoder.batch_decode([prefix], skip_special_tokens=True)
        return decoded

    full = text(tokens)
    candidates = [full[i : i + n] for i in range(len(full)) for n in (1, 2, 3)]
    candidates += ["\ufffd\ufffd", "é\ufffd", "\ufffda", "!\ufffd"]
    stop = tuple({rng.choice(candidates) for _ in range(rng.randint(1, 3))} - {""})
    if not stop:
        return
    expected = None
    for index in range(len(tokens)):
        visible = text(tokens[: index + 1]).rstrip("\ufffd")
        found = [(visible.find(s) + len(s), -len(s), s) for s in stop if s in visible]
        if found:
            expected = (index, min(found)[2])
            break
    groups = []
    while sum(groups) < len(tokens):
        groups.append(rng.randint(1, 6))

    assert (
        _feed(
            RequestStops(
                SamplingParams(max_new_tokens=1024, temperature=0, stop=stop).snapshot(), decoder
            ),
            tokens,
            groups,
        )
        == expected
    )
    assert _rendered(decoder, tokens, groups) == _rendered(decoder, tokens, [1] * len(tokens))
