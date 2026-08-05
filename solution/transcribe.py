"""
solution/transcribe.py
-----------------------
Local, offline dual-language (Hindi+English) speech-to-text engine.
Batch contract: python -m solution.transcribe --input clip.wav --mode auto --output result.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FAST_MODEL_NAME = os.environ.get("STT_FAST_MODEL", "base.en")
HINGLISH_MODEL_NAME = os.environ.get("STT_HINGLISH_MODEL", "small")
FALLBACK_MODEL_NAME = os.environ.get("STT_FALLBACK_MODEL", "tiny")

DEVICE = os.environ.get("STT_DEVICE", "cpu")               # faster-whisper fallback only
COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", "int8")   # faster-whisper fallback only

SAMPLE_RATE = 16000
SILENCE_PAD_S = 0.3  # 300ms speech_pad_ms

FAST_BEAM_SIZE = int(os.environ.get("STT_FAST_BEAM_SIZE", "1"))        # English path: base.en, greedy
HINGLISH_BEAM_SIZE = int(os.environ.get("STT_HINGLISH_BEAM_SIZE", "5"))  # Hinglish path: small, beam 5

FAST_CALL_DEADLINE_S = float(os.environ.get("STT_FAST_DEADLINE_S", "0.6"))
HINGLISH_CALL_DEADLINE_S = float(os.environ.get("STT_HINGLISH_DEADLINE_S", "1.8"))
FALLBACK_CALL_DEADLINE_S = float(os.environ.get("STT_FALLBACK_DEADLINE_S", "0.5"))

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
HINDI_PROB_THRESHOLD = 0.10
EN_CONFIDENCE_THRESHOLD = 0.75

PROMPT_HINT = (
    "Hinglish mixed transcription. Hindi in Devanagari, English in Latin script. "
    "Example: नमस्ते, how are you? मैं ठीक हूँ। "
    "Example: क्या हाल है? मैं office जा रहा हूँ। "
    "Example: Please wait, मैं अभी आता हूँ।"
)

HINGLISH_HINT_WORDS = {
    "hai", "hain", "kar", "karo", "kya", "nahi", "nahin", "ka", "ki", "ke",
    "yeh", "woh", "mein", "aur", "bhi", "toh", "kro", "acha", "theek", "hoon",
    "baat", "sahi", "kuch", "sab", "apna", "kaise", "kab", "kahan", "kyun",
    "sirf", "lekin", "magar", "phir", "pehle", "baad", "saath", "liye", "bhai",
    "samajh", "aaj", "kal", "abhi", "kabhi", "chal", "rha", "rhi", "rhe",
    "hu", "gya", "gyi", "gye", "wale", "wali", "wala"
}
REPEAT_NGRAM = 4
REPEAT_MIN_RUNS = 4

_MLX_REPO_CANDIDATES: Dict[str, List[str]] = {
    "tiny": ["mlx-community/whisper-tiny-mlx", "mlx-community/whisper-tiny"],
    "base": ["mlx-community/whisper-base-mlx", "mlx-community/whisper-base"],
    "base.en": ["mlx-community/whisper-base.en-mlx", "mlx-community/whisper-base.en"],
    "small": ["mlx-community/whisper-small-mlx", "mlx-community/whisper-small"],
}

# ---------------------------------------------------------------------------
# Backend Selection & Lazy Loading
# ---------------------------------------------------------------------------
_mlx_whisper = None
_WhisperModel = None
_BACKEND: Optional[str] = None
_model_cache: Dict[str, Any] = {}
_resolved_mlx_repo: Dict[str, str] = {}


def _lazy_imports() -> None:
    global _mlx_whisper, _WhisperModel, _BACKEND
    if _BACKEND is None:
        try:
            import mlx_whisper as _mw  # type: ignore
            _mlx_whisper = _mw
            _BACKEND = "mlx"
        except Exception:
            _BACKEND = "faster_whisper"
    if _BACKEND == "faster_whisper" and _WhisperModel is None:
        from faster_whisper import WhisperModel  # type: ignore
        _WhisperModel = WhisperModel


@dataclass
class EngineResult:
    text: str = ""
    language: Optional[str] = None
    language_probability: Optional[float] = None
    engine_name: str = "none"


def _get_faster_whisper_model(name: str):
    if name not in _model_cache:
        _model_cache[name] = _WhisperModel(name, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model_cache[name]


def _mlx_transcribe_once(size: str, audio: Any, beam_size: int, is_hinglish: bool = False,
                         logprob_override: Optional[float] = None) -> EngineResult:
    candidates = [_resolved_mlx_repo[size]] if size in _resolved_mlx_repo else _MLX_REPO_CANDIDATES.get(size, [f"mlx-community/whisper-{size}-mlx"])
    last_err: Optional[Exception] = None
    for repo in candidates:
        try:
            if is_hinglish or size == FALLBACK_MODEL_NAME:
                extra: Dict[str, Any] = {
                    "language": "hi",
                    "beam_size": beam_size,
                    "best_of": beam_size,
                    "temperature": 0.0,
                    "no_speech_threshold": 0.99,
                    "compression_ratio_threshold": 2.8,
                    "logprob_threshold": logprob_override if logprob_override is not None else -2.0,
                    "initial_prompt": PROMPT_HINT,
                }
            else:
                extra = {
                    "language": "en",
                    "beam_size": 1,
                    "best_of": 1,
                    "temperature": 0.0,
                    "no_speech_threshold": 0.6,
                    "compression_ratio_threshold": 2.4,
                    "logprob_threshold": -1.0,
                }

            try:
                result = _mlx_whisper.transcribe(audio, path_or_hf_repo=repo, task="transcribe", **extra)
            except TypeError:
                extra.pop("initial_prompt", None)
                result = _mlx_whisper.transcribe(audio, path_or_hf_repo=repo, task="transcribe", **extra)

            _resolved_mlx_repo[size] = repo
            return EngineResult(
                text=(result.get("text") or "").strip(),
                language=result.get("language", "hi" if is_hinglish else "en"),
                engine_name=f"mlx-whisper-{size}",
            )
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"mlx-whisper: no working repo for size={size}: {last_err}")


def _faster_whisper_transcribe_once(size: str, audio: Any, beam_size: int, is_hinglish: bool = False,
                                     logprob_override: Optional[float] = None) -> EngineResult:
    model = _get_faster_whisper_model(size)
    use_vad = not is_hinglish
    if is_hinglish or size == FALLBACK_MODEL_NAME:
        extra: Dict[str, Any] = {
            "language": "hi",
            "best_of": beam_size,
            "temperature": 0.0,
            "no_speech_threshold": 0.99,
            "compression_ratio_threshold": 2.8,
            "log_prob_threshold": logprob_override if logprob_override is not None else -2.0,
            "initial_prompt": PROMPT_HINT,
            "vad_parameters": dict(threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=400, speech_pad_ms=300),
        }
    else:
        extra = {
            "language": "en",
            "best_of": 1,
            "temperature": 0.0,
            "no_speech_threshold": 0.6,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
        }

    segments, info = model.transcribe(
        audio, task="transcribe", beam_size=beam_size if is_hinglish else 1,
        vad_filter=use_vad, condition_on_previous_text=False,
        **extra,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return EngineResult(
        text=text,
        language=getattr(info, "language", None),
        language_probability=getattr(info, "language_probability", None),
        engine_name=f"faster-whisper-{size}",
    )


def _pad_with_silence(audio: np.ndarray, sr: int, pad_s: float = SILENCE_PAD_S) -> np.ndarray:
    try:
        if audio is None or audio.size == 0:
            return audio
        pad = np.zeros(int(sr * pad_s), dtype=np.float32)
        return np.concatenate([pad, audio.astype(np.float32), pad])
    except Exception:
        return audio


def _load_wav_as_array(path: str) -> Optional[Tuple[np.ndarray, int]]:
    try:
        import wave
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()
            channels = wf.getnchannels()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        if sampwidth != 2 or n_frames == 0:
            return None
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return data, sr
    except Exception:
        return None


def _transcribe_uncatchable(model_size: str, audio: Any, beam_size: int, is_hinglish: bool = False,
                            logprob_override: Optional[float] = None) -> EngineResult:
    _lazy_imports()

    padded_audio = audio
    try:
        if isinstance(audio, np.ndarray):
            padded_audio = _pad_with_silence(audio, SAMPLE_RATE)
        elif isinstance(audio, str):
            loaded = _load_wav_as_array(audio)
            if loaded is not None:
                arr, sr = loaded
                padded_audio = _pad_with_silence(arr, sr)
    except Exception:
        padded_audio = audio

    if _BACKEND == "mlx":
        try:
            return _mlx_transcribe_once(model_size, padded_audio, beam_size, is_hinglish=is_hinglish, logprob_override=logprob_override)
        except Exception:
            pass
    if _WhisperModel is None:
        from faster_whisper import WhisperModel  # type: ignore
        globals()["_WhisperModel"] = WhisperModel
    return _faster_whisper_transcribe_once(model_size, padded_audio, beam_size, is_hinglish=is_hinglish, logprob_override=logprob_override)


def _call_bounded_ex(model_size: str, audio: Any, beam_size: int, deadline_s: float,
                      is_hinglish: bool = False, logprob_override: Optional[float] = None
                      ) -> Tuple[Optional[threading.Thread], Optional[EngineResult]]:
    """Like `_call_bounded`, but also returns the worker thread.

    The underlying inference call cannot be cancelled once started — on timeout
    the thread just keeps running in the background. Callers that fire these
    calls on a repeating cadence (e.g. streaming partials) should hold onto the
    returned thread and check `.is_alive()` before starting another call on the
    same resource, instead of relying only on a wall-clock throttle, so a slow
    call can't have a second (and third...) call stacked on top of it.
    """
    result_box: Dict[str, Any] = {}

    def _worker() -> None:
        try:
            result_box["value"] = _transcribe_uncatchable(
                model_size, audio, beam_size, is_hinglish=is_hinglish, logprob_override=logprob_override
            )
        except Exception as e:
            result_box["error"] = e

    try:
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=max(0.05, deadline_s))
        if t.is_alive():
            return t, None
        return t, result_box.get("value")
    except Exception:
        return None, None


def _call_bounded(model_size: str, audio: Any, beam_size: int, deadline_s: float,
                   is_hinglish: bool = False, logprob_override: Optional[float] = None) -> Optional[EngineResult]:
    _, result = _call_bounded_ex(
        model_size, audio, beam_size, deadline_s, is_hinglish=is_hinglish, logprob_override=logprob_override
    )
    return result


# ---------------------------------------------------------------------------
# Post-Processing
# ---------------------------------------------------------------------------
def _looks_code_switched(text: str, language: Optional[str], lang_prob: Optional[float]) -> bool:
    try:
        if DEVANAGARI_RE.search(text):
            return True
        if language and language != "en" and (lang_prob or 0.0) >= HINDI_PROB_THRESHOLD:
            return True
        if language == "en" and lang_prob is not None and lang_prob < EN_CONFIDENCE_THRESHOLD:
            return True
        lowered = set(re.findall(r"[a-z']+", text.lower()))
        return len(lowered & HINGLISH_HINT_WORDS) >= 1
    except Exception:
        return False


def _looks_code_switched_text(text: str) -> bool:
    """Text-only code-switch heuristic: Devanagari script or Hinglish hint words.

    Use this (instead of `_looks_code_switched`) when no ASR-reported
    `language`/`language_probability` is available for the text being checked —
    e.g. checking a rolling live draft rather than a fresh model result.
    Passing a fabricated `language` into `_looks_code_switched` to reuse it here
    is wrong: `_looks_code_switched` short-circuits to True whenever
    `language != "en"` and `lang_prob >= HINDI_PROB_THRESHOLD`, independent of
    the text, so a call like `_looks_code_switched(text, "hi", 1.0)` returns
    True unconditionally.
    """
    try:
        if DEVANAGARI_RE.search(text):
            return True
        lowered = set(re.findall(r"[a-z']+", text.lower()))
        return len(lowered & HINGLISH_HINT_WORDS) >= 1
    except Exception:
        return False


def normalize_final(text: str) -> str:
    try:
        if not text:
            return text
        out = unicodedata.normalize("NFC", text)
        out = re.sub(r"\s+([?.!,:;])", r"\1", out)
        out = re.sub(r"\s+", " ", out).strip()
        return out
    except Exception:
        return text


def _detect_repetition_loop(text: str) -> bool:
    try:
        words = text.split()
        if len(words) < REPEAT_NGRAM * REPEAT_MIN_RUNS:
            return False
        for i in range(len(words) - REPEAT_NGRAM * REPEAT_MIN_RUNS + 1):
            window = tuple(words[i:i + REPEAT_NGRAM])
            if all(tuple(words[i + r * REPEAT_NGRAM: i + (r + 1) * REPEAT_NGRAM]) == window
                   for r in range(1, REPEAT_MIN_RUNS)):
                return True
        return False
    except Exception:
        return False


def _dedupe_repetition(text: str) -> str:
    try:
        words = text.split()
        n = REPEAT_NGRAM
        for i in range(len(words) - n * 2 + 1):
            if tuple(words[i:i + n]) == tuple(words[i + n:i + 2 * n]):
                return " ".join(words[:i + n])
        return text
    except Exception:
        return text


# ---------------------------------------------------------------------------
# Core Batch API
# ---------------------------------------------------------------------------
def transcribe(input_path: str, mode: str = "auto", dictionary_path: Optional[str] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    raw_candidates: List[Dict[str, str]] = []
    model_ids: List[str] = []
    warnings: List[str] = []
    final_text = ""
    language_guess = "unknown"

    try:
        if mode not in ("auto", "fast", "hinglish", "verbatim"):
            mode = "auto"

        fast = _call_bounded(FAST_MODEL_NAME, input_path, FAST_BEAM_SIZE, FAST_CALL_DEADLINE_S)
        if fast is not None:
            raw_candidates.append({"engine": fast.engine_name, "text": fast.text})
            model_ids.append(fast.engine_name)
            final_text = fast.text
            language_guess = fast.language or "en"
        else:
            warnings.append("fast_pass_failed_or_timed_out")

        need_hinglish = mode in ("hinglish", "verbatim") or (
            mode == "auto" and (
                fast is None or not fast.text.strip() or
                _looks_code_switched(fast.text, fast.language, fast.language_probability)
            )
        )

        if need_hinglish:
            # Primary Hinglish Model (small, language="hi", beam_size=5, deadline=1.8s)
            strong = _call_bounded(
                HINGLISH_MODEL_NAME, input_path, HINGLISH_BEAM_SIZE, HINGLISH_CALL_DEADLINE_S, is_hinglish=True
            )
            if strong is not None and strong.text.strip():
                raw_candidates.append({"engine": strong.engine_name, "text": strong.text})
                model_ids.append(strong.engine_name)
                final_text = strong.text
                language_guess = "hi"
            else:
                # Fallback Tier 1: Tiny Multilingual Model (language="hi", beam_size=1, deadline=0.5s)
                warnings.append("hinglish_primary_failed_using_tiny_fallback")
                fb1 = _call_bounded(
                    FALLBACK_MODEL_NAME, input_path, 1, FALLBACK_CALL_DEADLINE_S, is_hinglish=True
                )
                if fb1 is not None and fb1.text.strip():
                    raw_candidates.append({"engine": fb1.engine_name, "text": fb1.text})
                    model_ids.append(fb1.engine_name)
                    final_text = fb1.text
                    language_guess = "hi"

        if _detect_repetition_loop(final_text):
            warnings.append("repetition_loop_detected_and_trimmed")
            final_text = _dedupe_repetition(final_text)

        if dictionary_path:
            final_text, matched = _apply_dictionary(final_text, dictionary_path)
            if matched:
                warnings.append(f"dictionary_terms_applied:{','.join(matched)}")

        final_text = normalize_final(final_text)

    except Exception as e:
        warnings.append(f"unexpected_error:{type(e).__name__}")

    if not final_text:
        warnings.append("blank_output")

    total_ms = (time.perf_counter() - t0) * 1000.0
    result = {
        "text": final_text,
        "mode_used": mode,
        "language_guess": language_guess,
        "timings_ms": {"total": round(total_ms, 1)},
        "raw_candidates": raw_candidates,
        "model_ids": model_ids,
        "local_only": True,
    }
    if warnings:
        result["warnings"] = warnings
    return result


def _apply_dictionary(text: str, dictionary_path: str) -> Tuple[str, List[str]]:
    try:
        if not os.path.exists(dictionary_path):
            return text, []
        with open(dictionary_path, "r", encoding="utf-8") as f:
            terms = json.load(f)
        matched = []
        for wrong, right in terms.items():
            pattern = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
            if pattern.search(text):
                text = pattern.sub(right, text)
                matched.append(right)
        return text, matched
    except Exception:
        return text, []


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Local, offline dual-language (Hindi+English) speech-to-text.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--mode", default="auto", choices=["auto", "fast", "hinglish", "verbatim"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--dictionary", default=None)
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        result = {
            "text": "", "mode_used": args.mode, "language_guess": "unknown",
            "timings_ms": {"total": 0.0}, "raw_candidates": [], "model_ids": [],
            "local_only": True, "warnings": ["input_file_not_found"],
        }
    else:
        result = transcribe(args.input, mode=args.mode, dictionary_path=args.dictionary)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())