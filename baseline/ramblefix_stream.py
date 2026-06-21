"""RambleFix streaming baseline — the benchmark line for the streaming track.

RambleFix is the engine entrants are trying to beat. It is a BENCHMARK LINE ONLY:
it can never take the $500. Its published streaming numbers are pre-registered in
docs/STREAMING_CONTRACT.md (the latency curve knees were committed BEFORE these
numbers were computed).

This file is a CLEAN, self-contained wrapper that exposes RambleFix's streaming
behaviour through the SAME public `draft()` contract entrants implement, so the
benchmark runs through the public evaluator.py with nothing private embedded.

What RambleFix's draft loop does (the shape, reproduced here without its source):
  - warm local whisper.cpp small server drafts the rolling audio prefix,
  - commit the longest common word-prefix across consecutive drafts (stabilizes),
  - on key-up, paste the latest draft immediately,
  - an async Hinglish finalizer (Srota/Qwen-class, code-switch-faithful) replaces
    the final when it lands.

Model dependency (NOT shipped here — the public can substitute equivalents):
  - a local whisper.cpp server with a small multilingual model (draft path),
  - a local code-switch-capable finalizer (Hinglish path).
Both must be commercial-friendly to qualify as a publishable baseline. If you do
not have these locally, this wrapper degrades to an empty draft (so the contract
still validates) and you should rely on the PUBLISHED numbers in the contract doc.

NO secrets, no API keys, no cloud, no private recordings. Fully local or nothing.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import wave

_SR = 16000
_MIN_AUDIO_BYTES = int(_SR * 0.75) * 2

# Point these at your LOCAL models. Defaults are unset -> wrapper degrades cleanly.
WHISPER_CPP_SERVER = os.environ.get("RAMBLEFIX_WHISPER_CPP_SERVER", "")  # e.g. http://127.0.0.1:8080/inference
HINGLISH_FINALIZER_CMD = os.environ.get("RAMBLEFIX_HINGLISH_FINALIZER", "")  # local CLI: reads a wav, prints text

_prev_text = ""
_committed = ""


def draft_reset() -> None:
    global _prev_text, _committed
    _prev_text = ""
    _committed = ""


def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    """Same contract as solution/draft.py. Draft path while streaming; Hinglish
    finalizer on is_final (faithful to the code-switch, never translated away)."""
    global _prev_text, _committed
    if not is_final and len(audio_buffer) < _MIN_AUDIO_BYTES:
        return (_committed, len(_committed))

    if is_final and HINGLISH_FINALIZER_CMD:
        final = _finalize(audio_buffer)
        if final:
            _committed = final
            return (final, len(final))

    text = _whisper_cpp_prefix(audio_buffer)
    if not text:
        return (_committed, len(_committed))

    stable = _common_word_prefix(_prev_text, text)
    if len(stable) >= len(_committed):
        _committed = stable
    _prev_text = text
    if is_final:
        _committed = text
        return (text, len(text))
    return (text, len(_committed))


def _pcm_to_wav(audio_buffer: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="ramblefix-stream-")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SR)
        w.writeframes(audio_buffer)
    return path


def _whisper_cpp_prefix(audio_buffer: bytes) -> str:
    """Draft path: local whisper.cpp server on the rolling prefix. Loopback only."""
    if not WHISPER_CPP_SERVER:
        return ""
    path = _pcm_to_wav(audio_buffer)
    try:
        import urllib.request
        with open(path, "rb") as fh:
            data = fh.read()
        req = urllib.request.Request(WHISPER_CPP_SERVER, data=data,
                                     headers={"Content-Type": "audio/wav"})
        with urllib.request.urlopen(req, timeout=10) as resp:  # local server only
            import json
            body = resp.read().decode("utf-8", "replace")
        try:
            return _clean(json.loads(body).get("text", ""))
        except Exception:
            return _clean(body)
    except Exception:  # noqa: BLE001 - degrade, never crash the run
        return ""
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _finalize(audio_buffer: bytes) -> str:
    """Hinglish finalizer: a local CLI that reads a wav path and prints text."""
    if not HINGLISH_FINALIZER_CMD or not shutil.which(HINGLISH_FINALIZER_CMD.split()[0]):
        return ""
    path = _pcm_to_wav(audio_buffer)
    try:
        out = subprocess.run(HINGLISH_FINALIZER_CMD.split() + [path],
                             capture_output=True, text=True, timeout=20)
        return _clean(out.stdout)
    except Exception:  # noqa: BLE001
        return ""
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _clean(text: str) -> str:
    s = (text or "").strip()
    if re.fullmatch(r"\[(?:BLANK_AUDIO|MUSIC|NOISE|SILENCE|INAUDIBLE)\]", s, re.I):
        return ""
    return s


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
