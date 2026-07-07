import argparse
import json
import time
import numpy as np
from faster_whisper import WhisperModel

def main():
    parser = argparse.ArgumentParser(description="Builderr Hinglish ASR Engine")
    parser.add_argument("--input", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "fast", "hinglish", "verbatim"])
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON")
    args = parser.parse_args()

    start_time = time.time()

    model_id = "small"
    model = WhisperModel(model_id, device="cpu", compute_type="int8")
    
    hinglish_prompt = "Transcribe the following mixed Hindi and English speech exactly as spoken using Latin script. Do not translate. Ensure dates and numbers are written out."

    segments, info = model.transcribe(
        args.input,
        language="hi",
        initial_prompt=hinglish_prompt,
        beam_size=5,
        word_timestamps=False
    )
    
    transcribed_text = " ".join([segment.text for segment in segments]).strip()
    
    end_time = time.time()
    latency_ms = int((end_time - start_time) * 1000)

    result = {
        "text": transcribed_text,
        "mode_used": args.mode,
        "language_guess": info.language,
        "timings_ms": {
            "total": latency_ms
        },
        "raw_candidates": [transcribed_text],
        "model_ids": [f"faster-whisper-{model_id}"],
        "local_only": True
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
