"""
solution/draft.py — builderr streaming dictation track

Correct signature (from the harness, confirmed from repo diff):

    draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]

Args:
    audio_buffer : bytes
        ALL audio so far for this utterance, as raw PCM.
        Format: s16le (signed 16-bit little-endian), mono, 16 kHz.
        The harness sends the FULL accumulated buffer each call, not just
        the latest chunk — so this is always the complete audio heard so far.

    is_final : bool
        False while the user is still speaking (mid-utterance partials).
        True once the user has stopped — this is your last chance to return
        the best full transcript. Return your highest-quality result here.

Returns:
    (text_so_far, stable_chars) : tuple[str, int]

    text_so_far  : str
        Your current best transcript of everything heard so far.
        Keep Hindi words in Roman script. Do NOT translate to English.

    stable_chars : int
        Length of the LEADING PREFIX of text_so_far that you COMMIT to —
        you promise never to shorten or change these characters in future calls.
        Must be non-decreasing across calls (never go backwards).
        The harness uses this to lock in text that won't change, giving the
        "live dictation feel" where early words appear and stay put.

        Strategy: commit words only when you're confident they're stable.
        A safe heuristic: commit everything except the last word,
        since the last word is most likely to be revised as more audio arrives.

Scoring notes (from leaderboard analysis):
    - RambleFix's draft TRANSLATES Hindi — it scores low on faithfulness here.
    - A faithful draft that keeps Hindi in Roman script will beat it.
    - Time-to-first-stable-text (TTFS) matters — emit something early.
    - stable_chars being too aggressive (over-committing) causes revision
      churn penalty. Be conservative — commit only clearly-finished words.
"""
from __future__ import annotations

import io
import os
import re
import tempfile
import time

import numpy as np

_SR = 16_000
_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
_MIN_AUDIO_S = 0.8          # don't attempt transcription under this duration
_MIN_AUDIO_BYTES = int(_MIN_AUDIO_S * _SR * 2)   # s16le = 2 bytes per sample
_RATE_LIMIT_S = 0.35        # don't run inference more often than this (mid-utterance)

_PROMPT = (
    "Transcribe exactly as spoken. Keep Hindi words in Roman script. "
    "Do not translate Hindi to English. Preserve the Hindi-English mix exactly. "
    "Keep technical terms, numbers, and names as spoken."
)

# state across calls within one utterance
_last_text: str = ""
_last_stable: int = 0
_last_run_at: float = 0.0
_fw_model = None


def _get_fw():
    global _fw_model
    if _fw_model is None:
        from faster_whisper import WhisperModel
        _fw_model = WhisperModel(
            "large-v3-turbo", device="cpu", compute_type="int8"
        )
    return _fw_model


def _pcm_to_float(audio_buffer: bytes) -> np.ndarray:
    """Convert raw s16le bytes to float32 numpy array in [-1, 1]."""
    arr = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32)
    return arr / 32768.0


def _write_wav(audio_f32: np.ndarray) -> str:
    """Write float32 audio to a temp WAV file, return path."""
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    sf.write(tmp, audio_f32, _SR, subtype="PCM_16")
    return tmp


def _infer(audio_buffer: bytes) -> str:
    """Run whisper on the full audio buffer, return transcript string."""
    audio_f32 = _pcm_to_float(audio_buffer)
    tmp = _write_wav(audio_f32)
    try:
        try:
            import mlx_whisper
            result = mlx_whisper.transcribe(
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
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )
            text = result.get("text", "").strip()
        except ImportError:
            fw = _get_fw()
            segs, _ = fw.transcribe(
                tmp,
                language="hi",
                task="transcribe",
                initial_prompt=_PROMPT,
                temperature=0.0,
                beam_size=1,
                vad_filter=False,       # don't skip anything on partial buffers
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )
            text = " ".join(s.text.strip() for s in segs).strip()
        return text
    except Exception:
        return ""
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _compute_stable_chars(text: str, is_final: bool) -> int:
    """
    Compute how many leading characters of text to commit.

    Strategy:
        - On is_final=True: commit everything (full transcript is stable).
        - Mid-utterance: commit everything up to (but not including)
          the last word — last word may still be revised as more audio arrives.
        - Never return a value less than the previous stable_chars
          (non-decreasing invariant).
    """
    global _last_stable

    if is_final:
        stable = len(text)
    else:
        # find the last word boundary
        stripped = text.rstrip()
        last_space = stripped.rfind(" ")
        if last_space == -1:
            # only one word so far — don't commit anything yet
            stable = 0
        else:
            # commit up to and including the space after the last complete word
            stable = last_space + 1

    # non-decreasing invariant
    stable = max(stable, _last_stable)
    # never exceed text length
    stable = min(stable, len(text))
    _last_stable = stable
    return stable


def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    """
    Called repeatedly by the harness as audio arrives.

    Args:
        audio_buffer: ALL audio so far, raw s16le bytes, mono 16 kHz.
        is_final:     True when user has stopped speaking.

    Returns:
        (text_so_far, stable_chars)
    """
    global _last_text, _last_stable, _last_run_at

    # Reset state at the start of a new utterance (empty buffer resets us).
    if len(audio_buffer) == 0:
        _last_text = ""
        _last_stable = 0
        _last_run_at = 0.0
        return ("", 0)

    # Not enough audio yet for a meaningful partial.
    if len(audio_buffer) < _MIN_AUDIO_BYTES and not is_final:
        return (_last_text, _last_stable)

    now = time.perf_counter()

    # Rate-limit mid-utterance inference to avoid hammering the model.
    if not is_final and (now - _last_run_at) < _RATE_LIMIT_S:
        return (_last_text, _last_stable)

    # Run inference.
    text = _infer(audio_buffer)
    _last_run_at = now

    if text:
        _last_text = text

    stable = _compute_stable_chars(_last_text, is_final)
    return (_last_text, stable)
