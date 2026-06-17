# builderr · Dual-Language Speech-to-Text Challenge

Build a **local, dual-language** speech-to-text engine — **Hindi + English** to start —
that beats the best free local tools on real code-switched work speech, **with no cloud
calls during scoring**. Prize: **$500**. Full brief: [`AGENT_BRIEF.md`](AGENT_BRIEF.md).

> Dual-language = two languages mixed in one utterance (code-switching). We open with
> Hindi+English; the same harness extends to other pairs (e.g. Chinese+English) later.

## The contract you implement

```bash
python -m solution.transcribe --input clip.wav --mode auto --output result.json
```

Emit the JSON in [`solution/transcribe.py`](solution/transcribe.py) (`text`,
`mode_used`, `language_guess`, `timings_ms`, `raw_candidates`, `model_ids`,
`local_only`). Modes: `auto` / `fast` / `hinglish` / `verbatim`.

## Run the local preview (scores you exactly like admission, offline)

```bash
pip install -r requirements.txt   # install a local ASR engine (faster-whisper, etc.)
python preview.py
```

## How you're scored (`scorecard.py`, out of 100)

| Area | Weight |
| --- | ---: |
| Meaning accuracy | 40 |
| Critical facts & terms (dates, numbers, negation, acronyms, names) | 25 |
| Latency & reliability (p50/p95, blanks, hangs) | 20 |
| Local-only proof (network blocked after warmup) | 10 |
| Auditability (raw candidates, route, timings, model IDs) | 5 |

**Hard caps:** a critical flip caps a clip at 50; a blank at 20; a repetition loop
at 30; output unrelated to the audio at 20. **Any network call during scoring
fails. Hardcoded phrases / hidden answer maps fail.**

## Rules

- Fully local; the official run blocks outbound network after warmup (loopback to a
  local ASR server is fine — see `offline_guard.py`).
- **Declare your models and their licenses — they must be commercial-friendly.**
- p95 under 5s, hang/error rate under 1%, no hardcoded phrase fixes.

## Submit

Email your repo to **submit@builderr.ai**. We clone it, run it offline on the hidden
set on a Linux box (CPU+GPU), and you land on the board.

## Tests

```bash
python tests/test_scorecard.py     # scoring is fair + un-gameable
python tests/test_no_network.py    # offline enforcement works
```
