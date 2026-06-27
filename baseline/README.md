# RambleFix baseline — the line you're trying to beat

RambleFix is the benchmark engine for the streaming dictation track. It is a
**benchmark line only — it can't win the $500.** It exists so you have a real,
runnable target to beat, and so you can see exactly *where* it's weak and aim there.

This folder is a clean wrapper around RambleFix's streaming behaviour, exposed
through the **same `draft()` contract you implement** in `solution/draft.py`, so it
runs through the public `evaluator.py` with nothing private inside. No API keys, no
cloud, no private recordings — fully local or nothing.

## What it does (the shape)

Two models, one fast and one faithful, racing:

1. **Draft path** — a warm local `whisper.cpp` small server transcribes the rolling
   audio prefix every chunk. Fast, but general: it tends to *translate* Hindi-English
   speech into plain English (you lose the code-switch).
2. **Commit** — it keeps the longest common word-prefix across consecutive drafts, so
   committed text stops flickering as more audio arrives.
3. **On key-up** — it pastes the latest draft immediately (so something lands fast),
   then an **async Hinglish finalizer** (a code-switch-faithful model) replaces the
   final when it's ready — keeping English terms in English, romanizing the Hindi.

You can watch both halves disagree on one clip:

| path | output on the same hi-en clip |
|------|-------------------------------|
| draft (whisper.cpp small, fast) | `In this tutorial, we will learn about the running of the Impress window and how to insert the slide and copy it.` |
| final (faithful) | `Is tutorial mein ham impress window ke bhaagon ke baare mein sikhenge aur kaise slide insert karen aur copy karen.` |

The fast one threw the mix away. The faithful one kept it but is slower. That tension
*is* the whole challenge.

## Pull and run it (public models, ~2 commands)

RambleFix's private engine uses a Srota/Qwen-class finalizer. Here we substitute
**public, commercial-friendly** models so you can run the exact same shape yourself:

- draft path → `whisper.cpp` with a small multilingual ggml model
- faithful final → `Oriserve/Whisper-Hindi2Hinglish-Swift` (or `moorlee/qwen3-asr-0.6b-hinglish`)

```bash
# 0) deps
pip install -r ../requirements-streaming.txt        # transformers, torch
brew install whisper-cpp                             # or build from ggml-org/whisper.cpp

# 1) draft server — warm, loopback only (downloads ggml-small the first time)
#    grab a model:  bash <(curl -s https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/models/download-ggml-model.sh) small
whisper-server -m /path/to/ggml-small.bin --host 127.0.0.1 --port 8089 &

# 2) point the wrapper at both local models, then run the public evaluator
export RAMBLEFIX_WHISPER_CPP_SERVER="http://127.0.0.1:8089/inference"
export RAMBLEFIX_HINGLISH_FINALIZER="python baseline/finalizer_hinglish.py"

# the wrapper now satisfies the same draft() contract as solution/draft.py
```

`finalizer_hinglish.py` is a ~50-line public reference: reads a wav path, prints
faithful romanized Hinglish, runs on Apple GPU (MPS) when present. Swap the model with
`RAMBLEFIX_FINALIZER_MODEL=...`. If you don't wire the env vars, the wrapper degrades
to an empty draft (so the contract still validates) — it never crashes the run.

> Note: these public substitutes are *RambleFix-class*, not byte-identical to the
> engine that produced the official board number. The canonical figure to beat is the
> one published on the leaderboard, measured on the frozen M1 Pro box.

## Known weaknesses — where it's beatable (aim here)

Measured on the locked 60-clip set (22 English / 25 Hindi / 13 hi-en), one scorer,
accelerator on. RambleFix lands **meaning ≈ 0.89, engine latency median ≈ 3s.**
Faithful — but **not under the 2s bar.** The gaps, ranked by how winnable they are:

1. **Latency is the headline gap (~3s vs the 2s target).** The async faithful
   finalizer is the bottleneck — Oriserve/Srota-class models via torch are accurate but
   heavy. **This is the #1 place to win:** get a faithful final under 2s. Levers that
   work — a smaller/quantized faithful model, mlx instead of torch, keeping it resident
   and warm, and overlapping decode with the stream instead of waiting for key-up.

2. **The pre-final window shows translated text.** Until the faithful final lands, the
   user sees the fast draft — which translates the mix to English (see the table
   above). If your final is late, that's what they read. Committing a *faithful* partial
   early beats this.

3. **Hindi is the weak language (≈0.83 vs English ≈0.97).** Heavier Hindi or fast
   code-switching is where it slips most. English-dominant clips it nearly nails.

4. **English term-flips.** On some clips it drops or mangles required terms/names — and
   the scorer caps hard on a flipped required term or number. A model that holds proper
   nouns and numbers exactly gains free points the generic-fast engines leave behind.

5. **Two-model overhead.** Running a draft model *and* a finalizer costs memory and
   warmup. A single fast+faithful model, if you can find one, sidesteps the whole race.

**The bar in one line:** nobody is *fast AND faithful under 2s* yet. The fastest entry
(~0.5s) holds only ~0.54 meaning; RambleFix holds 0.89 but needs ~3s. Close that gap
and the $500 is yours.

## Files

- `ramblefix_stream.py` — the streaming wrapper (same `draft()` contract as yours)
- `finalizer_hinglish.py` — public faithful-final reference (Oriserve / qwen3-hinglish)
- `stream_server.py` — the sealed harness shape (do-not-edit; builderr provides this)
