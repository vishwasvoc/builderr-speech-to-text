"""
solution/transcribe.py — builderr Dual-Language Speech-to-Text Challenge

Signature (from harness):
    transcribe(audio_path: str) -> str

Returns the transcript as a plain string, preserving Hindi+English mix.

Stack:
    Primary:  mlx-whisper + whisper-large-v3-turbo (Apple M1 GPU, MIT license)
    Fallback: faster-whisper CPU (Windows dev / non-Apple hardware)

Key faithfulness choices:
    - language="hi"               forces Hindi+English code-switching mode
    - task="transcribe"           NOT "translate" — critical, prevents Hindi→English
    - initial_prompt              anchors decoder to Roman-script Hinglish
    - temperature=0.0             greedy decoding, fastest + most consistent
    - condition_on_previous_text=False   prevents error snowballing
"""
from __future__ import annotations

import os
import re
import time

_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"

_PROMPT = (
    "Transcribe exactly as spoken. Keep Hindi words in Roman script. "
    "Do not translate Hindi to English. Preserve the Hindi-English mix exactly. "
    "Keep technical terms, numbers, and names as spoken. "
    "Example: 'rollback abhi mat karo, pehle p95 check karlo'"
)

# lazy singleton for CPU fallback
_fw_model = None


def _get_fw():
    global _fw_model
    if _fw_model is None:
        from faster_whisper import WhisperModel
        _fw_model = WhisperModel(
            "large-v3-turbo", device="cpu", compute_type="int8"
        )
    return _fw_model


def _clean(text: str) -> str:
    if not text:
        return ""
    _junk = {
        "thank you for watching", "thanks for watching",
        "please subscribe", "subtitles by",
        "[music]", "(music)", "[ music ]",
    }
    if text.lower().strip() in _junk:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _run_mlx(audio_path: str) -> str:
    import mlx_whisper
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=_MLX_MODEL,
        language="hi",
        task="transcribe",
        initial_prompt=_PROMPT,
        temperature=0.0,
        word_timestamps=False,
        verbose=False,
        fp16=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )
    return _clean(result.get("text", "").strip())


def _run_fw(audio_path: str) -> str:
    model = _get_fw()
    segs, _ = model.transcribe(
        audio_path,
        language="hi",
        task="transcribe",
        initial_prompt=_PROMPT,
        temperature=0.0,
        word_timestamps=False,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        beam_size=1,
    )
    return _clean(" ".join(s.text.strip() for s in segs))


def transcribe(audio_path: str) -> str:
    """
    Transcribe a Hindi+English audio file.

    Args:
        audio_path: path to audio file (WAV / MP3 / M4A / FLAC)

    Returns:
        Transcript as a plain string, preserving Hindi+English mix exactly.
        Does NOT translate Hindi to English.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        import mlx_whisper  # noqa — available on Apple Silicon
        return _run_mlx(audio_path)
    except ImportError:
        return _run_fw(audio_path)
