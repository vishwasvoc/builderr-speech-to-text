# Getting started — your first dual-language transcriber

No PhD required. If you can run Python and call a local model, you can enter. The goal:
turn a clip of mixed Hindi+English speech into clean text, on your own machine, fast.

## The whole path in one line

**fork → install a local model → run `preview.py` → improve the mix → submit the repo.**

---

## 1. Fork and install (5 minutes)

```bash
git clone https://github.com/builderr-ai/builderr-speech-to-text
cd builderr-speech-to-text
pip install -r requirements.txt        # pulls a local ASR engine (faster-whisper, etc.)
```

Everything runs on your laptop. No API keys, no cloud account, no audio leaves your machine.

## 2. Run the preview — see how the starter scores

```bash
python preview.py
```

This runs the example engine over the public dev clips and prints the **same scorecard the
real judging uses** (meaning, critical facts, latency, offline, caps). The starter is a thin
wrapper — it'll score low on the mix. That's your starting line, not your finish.

## 3. The one thing you implement

Open [`solution/transcribe.py`](solution/transcribe.py). You implement **one function**:
audio file in → text out, plus a small JSON (what you heard, how long it took, which models
you used). That's the whole contract. Run it like this:

```bash
python -m solution.transcribe --input clip.wav --mode auto --output result.json
```

## 4. The part that actually wins

Plain English is basically solved — wrapping an off-the-shelf model gets you there, and that's
fine, but it **won't win**. The prize is the **mix**: when someone says *"rollback abhi mat karo,
pehle p95 check karlo,"* your tool has to write down **what they actually said** (not translate it
to English) and keep the meaning — fast, and offline.

How the reference engine does it (read [`docs/REFERENCE_BOT.md`](docs/REFERENCE_BOT.md) for the
full story): a **fast recognizer** for the common English case, a **router** that spots the
Hindi-mixed clips, a **stronger Hindi-capable model** for just those, and a **finalizer** that
keeps it faithful without hanging or repeating. You don't have to copy it — but that shape works.

## 4b. The streaming track — where the $500 is decided

The prize is decided on **live dictation**: drafting text *as you speak* and finalizing fast, not
just transcribing a finished clip. You write **one more function** — `draft()` in
[`solution/draft.py`](solution/draft.py) — that emits text as audio arrives and commits what won't
change. The server and the real-time audio feed are a **sealed harness we provide**; you do not
build a server.

```bash
pip install -r requirements.txt -r requirements-streaming.txt
python preview_stream.py        # streams samples/ through your draft(), prints the streaming score
```

The contract, scoring, caps, frozen-CPU bar, and RambleFix's published numbers are the single
source of truth: [`docs/STREAMING_CONTRACT.md`](docs/STREAMING_CONTRACT.md). One combined score,
one $500 prize, RambleFix is the benchmark to beat.

## 5. Check yourself before you submit

```bash
python tests/test_scorecard.py     # how scoring works (and why it's hard to game)
python tests/test_no_network.py    # the run is offline — make sure yours works with the net off
```

Targets to aim for are in the [README](README.md) ("what it takes to win"). Short version:
match the best free tools on English, be **clearly the best on the Hindi+English mix**, stay fast,
stay local, ship-able license.

## 6. Submit

Email your repo to **submit@builderr.ai**. We clone it, run it offline on the hidden set, and you
land on the board. **Declare the models you used and their licenses** — they must be free to use
in a real product (so the winning tool can actually be released).

---

**Don't overthink version one.** Get *something* on the board, then iterate — the mix is where the
ranking is won, so spend your time there. And whatever you build here is genuinely useful: a free,
private, mixed-language dictation tool is a great thing to have on your GitHub, win or not.
