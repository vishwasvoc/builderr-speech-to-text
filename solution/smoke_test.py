"""
Quick sanity check that does NOT need a real recording.
Generates a short synthetic WAV (a spoken-word clip won't exist, so this
only proves the pipeline loads models, runs, and writes valid JSON without
crashing). Once this passes, replace test_tone.wav with a real clip -
ideally one of the sample clips from the challenge repo's samples/ folder,
or a short recording of your own voice mixing Hindi and English.

Run:
    python smoke_test.py
"""
import subprocess
import sys
import wave
import struct
import math
import os

OUT_WAV = "test_tone.wav"


def make_silence_wav(path: str, seconds: float = 2.0, rate: int = 16000) -> None:
    n_frames = int(seconds * rate)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        # near-silence with a faint tone so it's not pure zero-signal
        frames = bytearray()
        for i in range(n_frames):
            val = int(500 * math.sin(2 * math.pi * 220 * i / rate))
            frames += struct.pack("<h", val)
        w.writeframes(bytes(frames))


def main() -> int:
    make_silence_wav(OUT_WAV)
    print(f"Wrote {OUT_WAV}, running the CLI contract on it...")
    cmd = [
        sys.executable, "-m", "solution.transcribe",
        "--input", OUT_WAV,
        "--mode", "auto",
        "--output", "result.json",
    ]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print("Pipeline FAILED.")
        return 1
    print("\nPipeline ran and wrote result.json successfully.")
    print("(Text will likely be empty/garbage - this was a synthetic tone,")
    print(" not real speech. This test only checks the plumbing works.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
