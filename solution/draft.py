"""
solution/draft.py
------------------
Implements the REAL streaming contract, confirmed from
docs/STREAMING_CONTRACT.md:

    def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
        # audio_buffer = ALL audio heard so far (PCM s16le, mono, 16kHz)
        # called repeatedly as audio arrives (is_final=False), once more
        # after the user stops (is_final=True)
        # returns (text_so_far, stable_chars)

    - text_so_far: best transcript of the audio so far. Must keep the
      Hindi-English code-switch faithful (never translate the mix).
    - stable_chars: length of the committed leading prefix of
      text_so_far. Must be NON-DECREASING across calls, and that
      prefix string may only be EXTENDED, never rewritten.

Design, matching how it's actually scored (docs/STREAMING_CONTRACT.md
section 3): final meaning (40) + critical facts (20) = 60 of 100 points,
judged ONCE on the final call only. End-to-final latency is 25 points
(target ~2s). TTFS (5) and revision churn (5) are the only things partial
calls affect. So: partials exist to protect the TTFS/churn/no-partial
hard-caps cheaply - they are NOT where quality points are won. The full
quality+latency budget goes into the final pass.

Two behaviors follow from that:

1. Partial calls only re-transcribe the *uncommitted tail* of the audio
   (not the whole growing buffer) with a cheap model, so cost stays
   roughly constant no matter how long the clip gets. A prefix is only
   committed once it has matched across two consecutive tail passes AND
   a trailing safety margin is held back - because per 3.2, committed
   tokens that don't survive into the FINAL count as churn even if we
   personally never "rewrote" them - so being trigger-happy about
   committing hurts even without technically breaking the
   never-rewrite rule.

2. is_final=True ignores all the incremental/tail machinery and
   re-transcribes the FULL buffer once with the strong, faithful
   pipeline (fast pass -> escalate to the Hindi-capable model if
   code-switched -> romanize Devanagari -> repetition guard) - the
   exact same logic solution/transcribe.py uses in "auto" mode, reused
   directly so both paths agree.

NOT CONFIRMED: docs/STREAMING_CONTRACT.md never mentions a
`draft_reset` function. Your earlier failure ("draft_reset import
issue") means solution/stream_server.py imports it - that file is
visible locally in your repo right now (it only gets swapped for the
official copy at scoring time). If you want to close this last gap,
run:
    findstr /i "draft_reset" solution\\stream_server.py
and paste me the line - it'll show the exact call signature. Built
defensively below (accepts any args/kwargs) so it should work either
way, but confirming it removes the one remaining guess in this file.
"""

from __future__ import annotations

import time
from typing import Any, List, Tuple

import numpy as np

from solution.transcribe import (
    FAST_MODEL_NAME,
    HINGLISH_MODEL_NAME,
    _get_model,
    _looks_code_switched,
    _romanize,
    _detect_repetition_loop,
    _dedupe_repetition,
)

SAMPLE_RATE = 16000
TAIL_WINDOW_MAX_S = 8.0          # safety cap on tail re-decode length
COMMIT_SAFETY_WORDS = 2          # never commit the trailing N words of a pass
MIN_NEW_AUDIO_S = 0.4            # throttle: need this much new audio to re-run
MIN_WALL_INTERVAL_S = 0.35       # throttle: don't re-run more often than this


class _StreamState:
    def __init__(self) -> None:
        self.committed_text: str = ""
        self.committed_samples: int = 0       # audio cursor for the tail window
        self.prev_tail_words: List[str] = []  # for the 2-pass stability check
        self.last_run_wall: float = 0.0
        self.finalized: bool = False


_state = _StreamState()


def draft_reset(*_args: Any, **_kwargs: Any) -> None:
    """Reset all per-stream state. Signature not confirmed against the
    real solution/stream_server.py - see module docstring. Accepts and
    ignores any args/kwargs defensively."""
    global _state
    _state = _StreamState()
    return None


def _bytes_to_f32(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(bytes(buf), dtype=np.int16)
    return arr.astype(np.float32) / 32768.0


def _run_tail_partial(tail_audio: np.ndarray) -> Tuple[str, List[Tuple[str, float]]]:
    """Cheap pass over just the uncommitted tail. Returns the text and,
    if available, (word, end_time_within_tail) pairs used to advance the
    audio cursor precisely at a word boundary."""
    model = _get_model(FAST_MODEL_NAME)
    segments, _info = model.transcribe(
        tail_audio,
        task="transcribe",
        beam_size=1,                       # cheap - runs frequently
        vad_filter=True,
        condition_on_previous_text=False,
        word_timestamps=True,
    )
    segments = list(segments)
    text = " ".join(s.text.strip() for s in segments).strip()
    words: List[Tuple[str, float]] = []
    for seg in segments:
        for w in (getattr(seg, "words", None) or []):
            words.append((w.word.strip(), w.end))
    return text, words


def _run_final(full_audio: np.ndarray) -> str:
    """Full-buffer, full-quality pass - same faithful logic as
    transcribe.py's auto mode. This is where the 60 quality points live,
    so it gets the full latency budget, not the cheap partial pass."""
    fast_model = _get_model(FAST_MODEL_NAME)
    segments, info = fast_model.transcribe(
        full_audio, task="transcribe", beam_size=5,
        vad_filter=True, condition_on_previous_text=False,
    )
    fast_text = " ".join(s.text.strip() for s in segments).strip()

    if _looks_code_switched(fast_text, getattr(info, "language", None),
                             getattr(info, "language_probability", None)):
        strong_model = _get_model(HINGLISH_MODEL_NAME)
        segments2, _info2 = strong_model.transcribe(
            full_audio, task="transcribe", beam_size=5,
            vad_filter=True, condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments2).strip()
    else:
        text = fast_text

    if _detect_repetition_loop(text):
        text = _dedupe_repetition(text)

    return _romanize(text).strip()


def draft(audio_buffer: bytes, is_final: bool) -> Tuple[str, int]:
    """See module docstring for the contract. Returns (text_so_far,
    stable_chars)."""
    global _state
    full_audio = _bytes_to_f32(audio_buffer)

    if is_final:
        final_text = _run_final(full_audio)
        _state.committed_text = final_text
        _state.finalized = True
        return final_text, len(final_text)

    total_samples = len(full_audio)
    new_samples = total_samples - _state.committed_samples
    now = time.perf_counter()

    # Throttle: not enough new audio, or too soon since the last pass -
    # return the unchanged committed state (cheap, and can't cause churn
    # since nothing changes).
    if (new_samples / SAMPLE_RATE) < MIN_NEW_AUDIO_S or \
       (now - _state.last_run_wall) < MIN_WALL_INTERVAL_S:
        return _state.committed_text, len(_state.committed_text)

    _state.last_run_wall = now
    tail_start = _state.committed_samples
    tail_audio = full_audio[tail_start:]

    max_tail_samples = int(TAIL_WINDOW_MAX_S * SAMPLE_RATE)
    if len(tail_audio) > max_tail_samples:
        # Safety fallback only - shouldn't normally trigger if commits
        # are progressing at a reasonable rate.
        tail_audio = tail_audio[-max_tail_samples:]

    tail_text, tail_words = _run_tail_partial(tail_audio)
    new_words = tail_text.split()

    # 2-pass stability: only trust a word if it also appeared at the same
    # position in the immediately preceding tail pass.
    stable_word_count = 0
    for a, b in zip(_state.prev_tail_words, new_words):
        if a != b:
            break
        stable_word_count += 1
    commit_word_count = max(0, stable_word_count - COMMIT_SAFETY_WORDS)

    if commit_word_count > 0:
        addition = " ".join(new_words[:commit_word_count])
        _state.committed_text = (
            f"{_state.committed_text} {addition}".strip()
            if _state.committed_text else addition
        )
        if len(tail_words) >= commit_word_count:
            end_time_in_tail = tail_words[commit_word_count - 1][1]
            _state.committed_samples = tail_start + int(end_time_in_tail * SAMPLE_RATE)
        else:
            # No word timestamps available - fall back to a proportional
            # estimate of where in the audio that many words likely ended.
            frac = commit_word_count / max(1, len(new_words))
            _state.committed_samples = tail_start + int(frac * len(tail_audio))

    _state.prev_tail_words = new_words

    trailing = " ".join(new_words[commit_word_count:]) if commit_word_count else tail_text
    text_so_far = f"{_state.committed_text} {trailing}".strip()
    return text_so_far, len(_state.committed_text)
