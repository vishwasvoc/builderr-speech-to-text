"""
solution/draft.py
------------------
Streaming contract, confirmed from docs/STREAMING_CONTRACT.md:

    def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
        # audio_buffer = ALL audio heard so far (PCM s16le, mono, 16kHz)
        # returns (text_so_far, stable_chars)

REWRITTEN FROM SCRATCH after direct organizer feedback (see
solution/transcribe.py's docstring for the full quote) - completeness
comes first: every clip must return SOME final transcript, reliably,
before anything else is tuned. Read transcribe.py's docstring first;
this file shares its backend/timeout machinery and the same philosophy.

Design summary:
    - draft_reset() and draft() can each NEVER raise past their own
      boundary. Every code path is wrapped so a bug degrades to "return
      something safe" instead of "crash the stream server."
    - Partial calls (is_final=False) are pure best-effort: cheap,
      throttled, and if anything about them fails, draft() just returns
      whatever was already known. Partials are confirmed (guidelines
      page + the actual GitHub README, matching wording both times) to
      not affect the score directly, so they get minimal engineering
      risk budget.
    - The final call (is_final=True) is where the 70/100 (or 40+20 on
      the batch scorecard) accuracy points live, so it gets the real
      effort - but every model call within it goes through the same
      _call_bounded() fresh-executor timeout used in transcribe.py, and
      there is only ONE attempt at the slower Hindi-capable model, ever.
      No retries of a call that already failed.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple

import numpy as np

from solution.transcribe import (
    FAST_MODEL_NAME,
    HINGLISH_MODEL_NAME,
    FAST_BEAM_SIZE,
    HINGLISH_BEAM_SIZE,
    _call_bounded,
    _looks_code_switched,
    _romanize,
    _detect_repetition_loop,
    _dedupe_repetition,
)

SAMPLE_RATE = 16000
MIN_NEW_AUDIO_S = 0.6      # throttle: need this much new audio before another partial pass
MIN_WALL_INTERVAL_S = 0.7  # throttle: don't re-run more often than this

# Deadlines for the FINAL call specifically. Generous enough that a
# normally-completing call finishes cleanly (cutting it too short would
# itself hurt completeness), bounded so nothing can hang forever.
FAST_FINAL_DEADLINE_S = 6.0
HINGLISH_FINAL_DEADLINE_S = 6.0
PARTIAL_DEADLINE_S = 1.5


class _StreamState:
    def __init__(self) -> None:
        self.committed_text: str = ""
        self.committed_samples: int = 0
        self.last_run_wall: float = 0.0


_state = _StreamState()


def draft_reset(*_args: Any, **_kwargs: Any) -> None:
    """Reset per-stream state. Exact signature not confirmed (not in
    docs/STREAMING_CONTRACT.md - only known to be needed because a
    missing draft_reset caused an import failure early on). Accepts
    anything defensively. Can never raise."""
    global _state
    try:
        _state = _StreamState()
    except Exception:
        pass
    return None


def _bytes_to_f32(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(bytes(buf), dtype=np.int16)
    return arr.astype(np.float32) / 32768.0


def _run_final(full_audio: np.ndarray) -> str:
    """Exactly one attempt at the fast model, and at most one attempt at
    the Hindi-capable model. No retries. Always returns a string (never
    raises) - worst case, an empty string, which is a known, bounded
    outcome rather than a hang or crash."""
    text = ""
    try:
        fast = _call_bounded(FAST_MODEL_NAME, full_audio, FAST_BEAM_SIZE, FAST_FINAL_DEADLINE_S)
        fast_text = fast.text if fast else ""
        text = fast_text

        if fast is not None and _looks_code_switched(fast_text, fast.language, fast.language_probability):
            strong = _call_bounded(HINGLISH_MODEL_NAME, full_audio, HINGLISH_BEAM_SIZE, HINGLISH_FINAL_DEADLINE_S)
            if strong is not None and strong.text.strip():
                text = strong.text
            # else: keep fast_text - no retry, no second Hindi attempt

        if _detect_repetition_loop(text):
            text = _dedupe_repetition(text)

        text = _romanize(text)
    except Exception:
        pass  # text keeps whatever it last held - even "" is a safe, valid return

    return (text or "").strip()


def _run_partial(tail_audio: np.ndarray) -> str:
    """Best-effort only. On any failure, returns "" and the caller just
    reuses the last known committed text - a partial going wrong must
    never affect the final."""
    try:
        result = _call_bounded(FAST_MODEL_NAME, tail_audio, 1, PARTIAL_DEADLINE_S)
        return result.text if result else ""
    except Exception:
        return ""


def draft(audio_buffer: bytes, is_final: bool) -> Tuple[str, int]:
    """See module docstring. Returns (text_so_far, stable_chars).
    Guaranteed to never raise and to always return a (str, int) tuple -
    the absolute floor is ("", 0), which is a valid, safe response even
    if everything else in this function fails."""
    global _state
    try:
        full_audio = _bytes_to_f32(audio_buffer)
    except Exception:
        return (_state.committed_text if _state else ""), len(_state.committed_text if _state else "")

    try:
        if is_final:
            final_text = _run_final(full_audio)
            if final_text:
                _state.committed_text = final_text
                return final_text, len(final_text)
            # Final pass produced nothing usable - fall back to whatever
            # partial text we already have rather than return truly blank.
            fallback = _state.committed_text
            return fallback, len(fallback)

        now = time.perf_counter()
        new_samples = len(full_audio) - _state.committed_samples
        if (new_samples / SAMPLE_RATE) < MIN_NEW_AUDIO_S or (now - _state.last_run_wall) < MIN_WALL_INTERVAL_S:
            return _state.committed_text, len(_state.committed_text)

        _state.last_run_wall = now
        partial_text = _run_partial(full_audio)
        if partial_text:
            _state.committed_text = partial_text
            _state.committed_samples = len(full_audio)
        return _state.committed_text, len(_state.committed_text)

    except Exception:
        safe_text = _state.committed_text if _state else ""
        return safe_text, len(safe_text)
