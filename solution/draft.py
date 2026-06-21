"""The ONE function you implement for the STREAMING dictation track.

You do NOT build a server. The sealed harness (solution/stream_server.py) handles
the WebSocket, the real-time audio feed, and emitting events. You write `draft()`.

    draft(audio_buffer, is_final) -> (text_so_far, stable_chars)

The harness calls draft() repeatedly as audio arrives (is_final=False) and once
after the user stops (is_final=True). audio_buffer is ALL audio so far: raw PCM
s16le, mono, 16kHz (little-endian int16). Return:

  - text_so_far : your best transcript of the audio heard so far. Keep the
                  Hindi-English code-switch faithful — write what was actually
                  said, don't translate the mix into English (the scorecard caps
                  that). On is_final=True, return your best full transcript.
  - stable_chars: length of the leading prefix of text_so_far you COMMIT to —
                  you promise never to rewrite it. Must be non-decreasing across
                  calls. Rewriting committed text counts as revision churn.

Tips that match how the reference engine (RambleFix) does it:
  - Re-decode the rolling prefix; commit the longest common prefix with your
    previous draft (that part has stopped changing — safe to lock).
  - Don't translate to chase a meaning score; it kills faithfulness and is capped.
  - Be fast on the first useful partial (TTFS is scored) and on the final
    (end-to-final is the main latency axis).
  - Never return a blank, a loop, or hang — degrade to your best partial instead.

This reference body wraps a local faster-whisper draft on the rolling buffer and
commits the stable common prefix. If faster-whisper isn't installed it returns an
empty draft (clearly a non-winning placeholder) so the contract still validates.
Replace the body with your own router + Hindi-capable model + finalizer.
"""
from __future__ import annotations

import re

_SR = 16000
_MIN_AUDIO_BYTES = int(_SR * 0.75) * 2  # ~0.75s before the first draft (2 bytes/sample)

# per-clip state (the harness calls draft_reset() between clips)
_prev_text: str = ""
_committed: str = ""
_model = None
_np = None


def draft_reset() -> None:
    """Called by the sealed harness at the start of each clip. Clear per-clip state."""
    global _prev_text, _committed
    _prev_text = ""
    _committed = ""


def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    global _prev_text, _committed
    if not is_final and len(audio_buffer) < _MIN_AUDIO_BYTES:
        return (_committed, len(_committed))

    text = _transcribe_pcm(audio_buffer)
    if not text:
        # never blank-out a committed prefix; hold what we have
        return (_committed, len(_committed))

    # commit the longest common WORD prefix with the previous draft — that part
    # has stabilized across two decodes, so it's safe to lock.
    stable_text = _common_word_prefix(_prev_text, text)
    if len(stable_text) >= len(_committed):
        _committed = stable_text
    _prev_text = text

    if is_final:
        # final: everything is committed; return the full transcript
        _committed = text
        return (text, len(text))

    return (text, len(_committed))


def _transcribe_pcm(audio_buffer: bytes) -> str:
    """Local, offline ASR on the rolling PCM prefix. Reference uses faster-whisper."""
    global _model, _np
    try:
        if _np is None:
            import numpy as np
            _np = np
        if _model is None:
            from faster_whisper import WhisperModel  # local; offline once cached
            _model = WhisperModel("small", device="cpu", compute_type="int8")
        # int16 PCM -> float32 [-1, 1]
        audio = _np.frombuffer(audio_buffer, dtype=_np.int16).astype(_np.float32) / 32768.0
        if audio.size == 0:
            return ""
        segments, _info = _model.transcribe(audio, language=None, task="transcribe")
        return " ".join(s.text for s in segments).strip()
    except Exception:  # noqa: BLE001 - no model installed yet, or transient decode error
        return ""


def _common_word_prefix(left: str, right: str) -> str:
    lw, rw = _words(left), _words(right)
    out: list[str] = []
    for a, b in zip(lw, rw):
        if a.lower() != b.lower():
            break
        out.append(b)
    return " ".join(out)


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'.-]+", text, flags=re.UNICODE)
