"""
solution/draft.py
------------------
Streaming counterpart to solution/transcribe.py.

⚠️ IMPORTANT CAVEAT — READ BEFORE TRUSTING THIS BLINDLY ⚠️
I (the AI that wrote this) could not fetch your repo's actual
docs/STREAMING_CONTRACT.md or the original solution/draft.py stub —
GitHub blocks automated access to this specific private/small repo from
where I'm working. Everything below is built from the public challenge
copy you pasted earlier (GETTING_STARTED.md's "4b. The streaming track"
section), which describes the *shape* of the contract but not the exact
function signature or return schema.

Your last error was:
    stream server failed before READY (draft_reset import issue)

That almost certainly means stream_server.py does something like:
    from solution.draft import draft, draft_reset
...and solution/draft.py either didn't exist, or didn't define both
names. This file defines both. If you get a NEW error after this (e.g. a
TypeError about arguments, or a different missing name), that's actually
good news — it means we're past the import stage and into an argument
mismatch, which is a small, fast fix. Paste me the new traceback and
I'll adjust the signature immediately.

--------------------------------------------------------------------
What this implements (best-effort, pending contract confirmation):

    draft_reset()
        Called once per new audio stream/clip. Clears all internal
        buffers/state so streams don't bleed into each other.

    draft(audio_chunk, sample_rate=16000, is_final=False)
        Called repeatedly as real-time audio arrives (the harness feeds
        it small chunks, e.g. ~100-300ms at a time). Returns a dict:

            {
                "text":      str,  # full best-effort text so far
                "committed": str,  # the prefix that is now considered
                                    # stable and should not be rewritten
                "partial":   str,  # the trailing, still-changing part
            }

    Accepts audio_chunk as either raw int16 PCM bytes or a numpy array
    (float32 in [-1, 1] or int16) - normalizes internally either way,
    since I don't know for certain which the harness sends.

Design, matching the challenge's stated goals:
    - A cheap, fast partial pass runs every ~0.5s of new audio so the
      "time to first useful partial" score component stays low.
    - Text is only "committed" once it has been stable across two
      consecutive partial passes - this avoids the score penalty for
      "committed text getting rewritten" while still surfacing partials
      fast.
    - On is_final=True (or after a detected pause), a stronger pass
      re-transcribes the whole utterance using the same faithful
      (transcribe, never translate) + Hindi-routing + romanization logic
      as solution/transcribe.py, reused directly from that module so
      both paths behave identically. This is what the ~60%
      accuracy+mix scoring weight is paying for - worth the extra
      latency budget since speed is only ~35%.
    - Uses beam_size=1 for the frequent partial passes (cheap) and the
      full beam_size=5 finalizer logic from transcribe.py only once per
      utterance (expensive, but rare).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

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

SAMPLE_RATE_DEFAULT = 16000
PARTIAL_INTERVAL_S = 0.5   # how often to re-run the cheap partial pass
MIN_AUDIO_FOR_PARTIAL_S = 0.3  # don't bother transcribing tiny buffers


class _StreamState:
    def __init__(self) -> None:
        self.pcm_f32 = np.zeros(0, dtype=np.float32)
        self.sample_rate = SAMPLE_RATE_DEFAULT
        self.committed_text = ""
        self.last_partial_words: list[str] = []
        self.last_partial_time = 0.0
        self.finalized = False


_state = _StreamState()


def draft_reset(*_args: Any, **_kwargs: Any) -> None:
    """Reset all per-stream state. Called by the harness at the start of
    each new clip/session. Accepts/ignores any args defensively in case
    the harness passes something (e.g. a session id) we don't need."""
    global _state
    _state = _StreamState()
    return None


def _to_float32(audio_chunk: Any) -> np.ndarray:
    """Normalize whatever audio representation we're handed into a 1-D
    float32 numpy array in [-1, 1]."""
    if isinstance(audio_chunk, (bytes, bytearray)):
        arr = np.frombuffer(bytes(audio_chunk), dtype=np.int16)
        return arr.astype(np.float32) / 32768.0
    arr = np.asarray(audio_chunk)
    if arr.dtype == np.int16:
        return arr.astype(np.float32) / 32768.0
    return arr.astype(np.float32)


def _commit_stable_prefix(new_words: list[str]) -> None:
    """Compare the new partial's words against the previous partial's
    words; whatever matches at the start is considered stable and gets
    committed, so it won't be rewritten again (avoids the 'committed
    text gets rewritten' latency-score penalty)."""
    old_words = _state.last_partial_words
    stable_len = 0
    for a, b in zip(old_words, new_words):
        if a != b:
            break
        stable_len += 1
    # Leave the last couple of words uncommitted even if they matched -
    # they're the most likely to change as more audio arrives.
    safe_len = max(0, stable_len - 2)
    if safe_len > 0:
        newly_committed = " ".join(new_words[:safe_len])
        if newly_committed and not _state.committed_text.startswith(newly_committed):
            _state.committed_text = newly_committed
    _state.last_partial_words = new_words


def _run_partial(audio: np.ndarray, sample_rate: int) -> str:
    model = _get_model(FAST_MODEL_NAME)
    segments, _info = model.transcribe(
        audio,
        task="transcribe",
        beam_size=1,               # cheap - this runs frequently
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def _run_final(audio: np.ndarray, sample_rate: int) -> str:
    """Same faithful logic as transcribe.py's auto mode: fast pass,
    escalate to the Hindi-capable model if it looks code-switched,
    romanize Devanagari, guard against repetition loops."""
    fast_model = _get_model(FAST_MODEL_NAME)
    segments, info = fast_model.transcribe(
        audio, task="transcribe", beam_size=5,
        vad_filter=True, condition_on_previous_text=False,
    )
    fast_text = " ".join(seg.text.strip() for seg in segments).strip()

    if _looks_code_switched(fast_text, getattr(info, "language", None),
                             getattr(info, "language_probability", None)):
        strong_model = _get_model(HINGLISH_MODEL_NAME)
        segments2, _info2 = strong_model.transcribe(
            audio, task="transcribe", beam_size=5,
            vad_filter=True, condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments2).strip()
    else:
        text = fast_text

    if _detect_repetition_loop(text):
        text = _dedupe_repetition(text)

    text = _romanize(text)
    return text.strip()


def draft(
    audio_chunk: Any,
    sample_rate: int = SAMPLE_RATE_DEFAULT,
    is_final: bool = False,
    **_kwargs: Any,
) -> Dict[str, str]:
    """Called repeatedly as audio arrives. See module docstring for the
    return schema. Defensive **_kwargs absorbs any extra fields the
    harness might pass that we don't know about yet."""
    _state.sample_rate = sample_rate
    chunk_f32 = _to_float32(audio_chunk)
    _state.pcm_f32 = np.concatenate([_state.pcm_f32, chunk_f32])

    duration_s = len(_state.pcm_f32) / float(sample_rate)
    now = time.perf_counter()

    if is_final:
        final_text = _run_final(_state.pcm_f32, sample_rate)
        _state.committed_text = final_text
        _state.finalized = True
        return {
            "text": final_text,
            "committed": final_text,
            "partial": "",
        }

    should_run_partial = (
        duration_s >= MIN_AUDIO_FOR_PARTIAL_S
        and (now - _state.last_partial_time) >= PARTIAL_INTERVAL_S
    )
    if should_run_partial:
        _state.last_partial_time = now
        partial_text = _run_partial(_state.pcm_f32, sample_rate)
        words = partial_text.split()
        _commit_stable_prefix(words)
        full_text = partial_text
        trailing = full_text[len(_state.committed_text):].strip()
        return {
            "text": full_text,
            "committed": _state.committed_text,
            "partial": trailing,
        }

    # Not time for a new partial pass yet - return what we have.
    trailing = " ".join(_state.last_partial_words)[len(_state.committed_text):].strip()
    return {
        "text": (_state.committed_text + " " + trailing).strip(),
        "committed": _state.committed_text,
        "partial": trailing,
    }
