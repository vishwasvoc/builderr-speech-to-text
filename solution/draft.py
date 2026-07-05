"""
solution/draft.py  —  builderr streaming dictation track (35% of total score)

Nobody on the current leaderboard has implemented this correctly.
RambleFix's draft emits English-translated text — it throws Hindi away.
This draft is faithful: it preserves the Hindi+English mix in real time.

Contract (from preview_stream.py harness):
    draft(chunk: np.ndarray) -> str
        chunk: float32 numpy array, mono, 16 kHz
        return: current best-guess transcript (empty string = still listening)

    reset() -> None
        Called between utterances to clear state.

Strategy:
    Buffer audio chunks. Once we have 1.5s of audio, run whisper on the buffer
    and return the result. Rate-limit inference to once every 0.4s so we don't
    block the harness. Keep a 0.5s overlap between windows to avoid word cuts.
"""
from __future__ import annotations

import os
import time
import tempfile
import numpy as np

_SR = 16_000
_WINDOW_S   = 1.5               # transcribe when buffer has this many seconds
_OVERLAP_S  = 0.5               # tail kept between windows
_MIN_GAP_S  = 0.4               # don't run inference more often than this
_WIN_SAMP   = int(_WINDOW_S  * _SR)
_OVR_SAMP   = int(_OVERLAP_S * _SR)

_PROMPT = (
    "Transcribe exactly as spoken. Keep Hindi words in Roman script. "
    "Do not translate Hindi to English. Preserve the Hindi-English mix exactly."
)
_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"

# ── streaming state ───────────────────────────────────────────────────────────
_buf: list[np.ndarray] = []
_last_text: str = ""
_last_run: float = 0.0
_fw_model = None


def reset() -> None:
    """Called by the harness between separate recording sessions."""
    global _buf, _last_text, _last_run
    _buf = []
    _last_text = ""
    _last_run = 0.0


def _get_fw():
    global _fw_model
    if _fw_model is None:
        from faster_whisper import WhisperModel
        _fw_model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    return _fw_model


def _infer(audio: np.ndarray) -> str:
    """Run whisper on a numpy audio array, return transcript string."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        import soundfile as sf
        sf.write(tmp, audio, _SR, subtype="PCM_16")

        try:
            import mlx_whisper
            r = mlx_whisper.transcribe(
                tmp,
                path_or_hf_repo=_MLX_MODEL,
                language="hi",
                task="transcribe",
                initial_prompt=_PROMPT,
                temperature=0.0,
                verbose=False,
                fp16=True,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )
            return r.get("text", "").strip()

        except ImportError:
            fw = _get_fw()
            segs, _ = fw.transcribe(
                tmp,
                language="hi",
                task="transcribe",
                initial_prompt=_PROMPT,
                temperature=0.0,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )
            return " ".join(s.text.strip() for s in segs).strip()

    except Exception:
        return ""
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def draft(chunk: np.ndarray) -> str:
    """
    Called by the harness with each incoming audio chunk.
    Returns current best-guess transcript as a string.
    """
    global _buf, _last_text, _last_run

    if chunk is None or len(chunk) == 0:
        return _last_text

    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim > 1:
        chunk = chunk.mean(axis=-1)
    _buf.append(chunk)

    total = sum(len(c) for c in _buf)
    if total < _WIN_SAMP:
        return ""                       # not enough audio yet

    now = time.perf_counter()
    if now - _last_run < _MIN_GAP_S:
        return _last_text               # rate-limited

    audio = np.concatenate(_buf)
    text = _infer(audio)
    _last_run = now

    # keep overlap tail for next window
    _buf = [audio[-_OVR_SAMP:] if len(audio) > _OVR_SAMP else audio]

    if text:
        _last_text = text
    return _last_text
