"""
solution/draft.py
------------------
Streaming contract, confirmed from docs/STREAMING_CONTRACT.md:

    def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
        # audio_buffer = ALL audio heard so far (PCM s16le, mono, 16kHz)
        # returns (text_so_far, stable_chars)
"""

from __future__ import annotations

import threading
import time
from typing import Any, Tuple

import numpy as np

from solution.transcribe import (
    FAST_MODEL_NAME,
    HINGLISH_MODEL_NAME,
    FALLBACK_MODEL_NAME,
    FAST_BEAM_SIZE,
    HINGLISH_BEAM_SIZE,
    HINGLISH_CALL_DEADLINE_S,
    FALLBACK_CALL_DEADLINE_S,
    _call_bounded,
    _looks_code_switched,
    _detect_repetition_loop,
    _dedupe_repetition,
    normalize_final,
)

SAMPLE_RATE = 16000
MIN_NEW_AUDIO_S = 0.6
MIN_WALL_INTERVAL_S = 0.7

TOTAL_FINAL_BUDGET_S = 1.8
PARTIAL_DEADLINE_S = 0.8


class StreamingState:
    def __init__(self) -> None:
        self.audio_buffer: list[np.ndarray] = []
        self.committed_samples: int = 0
        self.speech_active: bool = False
        self.silence_start: float | None = None
        self.final_text: str | None = None
        self.live_draft: str = ""
        self.is_hinglish: bool = False
        self.last_run_wall: float = 0.0


state = StreamingState()


def draft_reset(*_args: Any, **_kwargs: Any) -> None:
    global state
    try:
        state = StreamingState()
    except Exception:
        pass
    return None


def _bytes_to_f32(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(bytes(buf), dtype=np.int16)
    return arr.astype(np.float32) / 32768.0


def _is_near_silent(audio: np.ndarray, rms_threshold: float = 0.004) -> bool:
    try:
        if audio.size == 0:
            return True
        rms = float(np.sqrt(np.mean(np.square(audio))))
        return rms < rms_threshold
    except Exception:
        return False


def get_final_hinglish(audio: np.ndarray, draft_text_latest: str) -> str:
    result = None
    exception = None

    def run_small():
        nonlocal result, exception
        try:
            result = _call_bounded(
                HINGLISH_MODEL_NAME, audio, HINGLISH_BEAM_SIZE, HINGLISH_CALL_DEADLINE_S, is_hinglish=True
            )
        except Exception as e:
            exception = e

    thread = threading.Thread(target=run_small, daemon=True)
    thread.start()
    thread.join(timeout=TOTAL_FINAL_BUDGET_S)

    if thread.is_alive() or exception or result is None or not result.text.strip():
        # Fallback Tier 1: Multilingual tiny model
        try:
            fallback_res = _call_bounded(
                FALLBACK_MODEL_NAME, audio, 1, FALLBACK_CALL_DEADLINE_S, is_hinglish=True
            )
            if fallback_res and fallback_res.text.strip():
                return normalize_final(fallback_res.text)
        except Exception:
            pass
        # Fallback Tier 2: Return latest draft text
        return normalize_final(draft_text_latest)

    return normalize_final(result.text)


def draft(audio_buffer: bytes, is_final: bool) -> Tuple[str, int]:
    """Streaming API contract function."""
    global state
    try:
        full_audio = _bytes_to_f32(audio_buffer)
    except Exception:
        return (state.final_text or state.live_draft), len(state.final_text or state.live_draft)

    try:
        if is_final:
            if state.final_text:
                return state.final_text, len(state.final_text)

            if _is_near_silent(full_audio):
                state.final_text = ""
                return "", 0

            # Determine route
            if state.is_hinglish or _looks_code_switched(state.live_draft, "hi", 1.0):
                final_res = get_final_hinglish(full_audio, state.live_draft)
            else:
                eng_res = _call_bounded(FAST_MODEL_NAME, full_audio, FAST_BEAM_SIZE, 0.6)
                if eng_res and eng_res.text.strip():
                    if _looks_code_switched(eng_res.text, eng_res.language, eng_res.language_probability):
                        final_res = get_final_hinglish(full_audio, state.live_draft)
                    else:
                        final_res = normalize_final(eng_res.text)
                else:
                    final_res = get_final_hinglish(full_audio, state.live_draft)

            if _detect_repetition_loop(final_res):
                final_res = _dedupe_repetition(final_res)

            state.final_text = final_res
            return state.final_text, len(state.final_text)

        if state.final_text:
            return state.final_text, len(state.final_text)

        now = time.perf_counter()
        new_samples = len(full_audio) - state.committed_samples
        if (new_samples / SAMPLE_RATE) < MIN_NEW_AUDIO_S or (now - state.last_run_wall) < MIN_WALL_INTERVAL_S:
            return state.live_draft, len(state.live_draft)

        state.last_run_wall = now
        res = _call_bounded(FALLBACK_MODEL_NAME, full_audio, 1, PARTIAL_DEADLINE_S, is_hinglish=True)
        if res and res.text.strip():
            state.live_draft = res.text.strip()
            state.committed_samples = len(full_audio)
            if _looks_code_switched(state.live_draft, res.language, res.language_probability):
                state.is_hinglish = True

        return state.live_draft, len(state.live_draft)

    except Exception:
        safe_text = state.final_text or state.live_draft
        return safe_text, len(safe_text)