# Build skill — dual-language speech-to-text engine

Drop this into your AI assistant (Claude / Cursor / Codex) as context for building your entry.
It's **direction, not a recipe** — the high-level shape that works, based on the reference engine
(RambleFix). The details are yours to invent.

## The goal
A speech-to-text engine that runs **locally and offline**, is **fast**, and is **faithful on
Hindi+English code-switch** — it writes what was actually said, it doesn't translate the mix into
English. Match the best free engines on English; **beat the benchmark on the mix**.

## High-level architecture (the shape that works)
It's not one model — it's a small pipeline:

1. **Fast foreground recognizer** (always warm) — handles the common English / Indian-English clip
   in well under a second.
2. **Router** — per clip, decide: plain English (use the fast path) or Hindi-mixed / risky
   (escalate). This is the crux: don't run the heavy model on *every* clip (latency dies), and
   don't miss the mix (accuracy dies).
3. **Stronger Hindi-capable path** — a code-switch-capable model, run **only on the escalated
   clips**.
4. **Finalizer** — keep the code-switch faithful, fix the things that flip meaning (numbers, dates,
   "not", names, work terms), and guard against loops / blanks / hallucination. If all else fails,
   return a plain transcript rather than nothing.

Throughout, keep the raw candidates + timings + which model ran (auditability is scored).

## Design guidelines (general direction)
- **Don't translate the mix to English** to chase a meaning score — it kills faithfulness, and the
  scorecard caps it.
- **English is near-solved** — grab a good local model and move on; spend your time on the mix and
  the router.
- **The router is where you win or lose.**
- **Reliability counts as much as accuracy** — a tool that hangs or loops doesn't get used.
- **Pick small, permissive, CPU-runnable models** — so the winning engine is shippable as a free
  product and clears the license gate.
- **Stay offline** — the scored run blocks outbound network after warmup (loopback to a local ASR
  server is fine).

## The streaming track (where the $500 is)
The prize is decided on **live dictation**, not batch transcription. The same pipeline shape —
fast draft on the rolling audio, router, Hinglish-capable escalation, faithful finalizer — *is*
the streaming track, just emitted incrementally: draft text **as audio arrives**, commit what
won't change, finalize fast after the user stops. You implement **one function**, `draft()` in
[`solution/draft.py`](solution/draft.py); the streaming server and real-time feed are a sealed
harness we provide (you don't build a server). Full contract, scoring, caps, and the RambleFix
benchmark line live in [`docs/STREAMING_CONTRACT.md`](docs/STREAMING_CONTRACT.md) — read it, don't
guess the weights.

## What to build
Implement `solution.transcribe` (audio → text + JSON) for the batch contract, and `solution.draft`
for the streaming track. Run `python preview.py` (batch) or `python preview_stream.py` (streaming)
to score yourself on the **sample clips in `samples/`**, offline. Submit the repo to
`submit@builderr.ai`.

## 🔒 Hard Learnings
- **TTFS / end-to-final latency / streaming-compliance are gates and caps, never standalone
  weighted prizes.** They fold into the ONE combined streaming score (~35% live feel) and into the
  hard caps — do not reintroduce them as separate scored weights or a second prize.
- **One prize, one combined score.** No "Best on the Mix" second award. The mix is the heart of the
  ~60% accuracy weight, not a separate track.
- **Never translate the mix to win meaning** — it's capped. Keep the code-switch faithful.

## Pointers
- **Rules, contract, scoring:** [`AGENT_BRIEF.md`](AGENT_BRIEF.md)
- **Step-by-step:** [`GETTING_STARTED.md`](GETTING_STARTED.md)
- **How the benchmark is built + the bar to beat:** [`docs/REFERENCE_BOT.md`](docs/REFERENCE_BOT.md)
- **Sample clips to test on:** [`samples/`](samples/) (English + Hindi+English, with reference transcripts)
