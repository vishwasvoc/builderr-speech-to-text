#!/usr/bin/env python3
"""Public Hinglish finalizer for the RambleFix baseline — the faithful final path.

RambleFix's private engine uses a Srota/Qwen-class code-switch model for the final.
This is a PUBLIC, commercial-friendly stand-in so anyone can run the baseline
end-to-end with nothing private. It loads a Hindi-English-faithful Whisper model
(keeps English tech terms in English, romanizes Hindi) and prints the transcript
for a single wav path — exactly the CLI shape ramblefix_stream.py shells out to:

    RAMBLEFIX_HINGLISH_FINALIZER="python baseline/finalizer_hinglish.py" \
        python -m evaluator ...

Model (override with RAMBLEFIX_FINALIZER_MODEL):
  - Oriserve/Whisper-Hindi2Hinglish-Swift  (default; faithful romanized Hinglish)
  - moorlee/qwen3-asr-0.6b-hinglish        (alt; also code-switch-faithful)

Runs on Apple GPU (MPS) when available, else CPU. First run downloads the model
from the Hugging Face Hub; subsequent runs are warm. No API keys, no cloud.
"""
from __future__ import annotations

import os
import sys

MODEL = os.environ.get("RAMBLEFIX_FINALIZER_MODEL", "Oriserve/Whisper-Hindi2Hinglish-Swift")

_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from transformers import pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    _pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL,
        device=device,
        torch_dtype=torch.float32,
    )
    return _pipe


def transcribe(wav_path: str) -> str:
    out = _get_pipe()(
        wav_path,
        generate_kwargs={"task": "transcribe"},
        chunk_length_s=30,
    )
    return (out.get("text") or "").strip()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: finalizer_hinglish.py <wav_path>", file=sys.stderr)
        sys.exit(2)
    try:
        print(transcribe(sys.argv[1]))
    except Exception as exc:  # noqa: BLE001 - finalizer must never crash the run
        print("", end="")
        print(f"finalizer error: {exc}", file=sys.stderr)
        sys.exit(0)
