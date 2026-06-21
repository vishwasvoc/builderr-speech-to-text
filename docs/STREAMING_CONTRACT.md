# Streaming dictation track — the contract (single source of truth)

This is the one document that defines the streaming track. README, SKILL, and
GETTING_STARTED link here and never restate the weights. If anything elsewhere
disagrees with this file, this file wins.

There is **ONE prize: $500**, ranked on **ONE combined weighted score** out of
100. RambleFix is the **benchmark line** you are trying to beat — it can never
take the prize.

---

## 1. What you build: ONE function

You do **not** build a server. You write one function, `draft()`, in
[`solution/draft.py`](../solution/draft.py). The streaming server, the real-time
audio feed, and the event wire are a **sealed harness** builderr provides
([`solution/stream_server.py`](../solution/stream_server.py) — header says
DO NOT EDIT; it is replaced with the official copy at scoring time).

```python
def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    """audio_buffer = ALL audio so far (PCM s16le, mono, 16kHz).
    Called repeatedly as audio arrives (is_final=False), once after the user
    stops (is_final=True). Return (text_so_far, stable_chars)."""
```

- `text_so_far`: your best transcript of the audio heard so far. **Keep the
  Hindi-English code-switch faithful** — write what was actually said; do **not**
  translate the mix into English (the scorecard caps that).
- `stable_chars`: length of the leading prefix of `text_so_far` you **commit** to
  (promise never to rewrite). Must be **non-decreasing** across calls, and the
  committed prefix string may only be **extended**. Rewriting committed text is
  counted as **revision churn**.

Preview yourself, offline, exactly like admission:

```bash
pip install -r requirements.txt -r requirements-streaming.txt
python preview_stream.py
```

---

## 2. Wire protocol (sealed harness — for reference only)

One loopback WebSocket per clip. Server prints `READY port=<PORT>` on warm; the
harness blocks on that line.

**Evaluator → solution:**
```jsonc
{"type":"start","sample_rate":16000,"format":"pcm_s16le","channels":1,"clip_id":"<opaque>"}
// audio: raw BINARY ws frames, 20ms each (640 bytes PCM s16le @16kHz), in order.
//        Binary is the ONLY audio path.
{"type":"end"}
```
**Solution → evaluator:**
```jsonc
{"type":"partial","text":"rollback abhi mat","stable_chars":11}  // zero or more, during audio
{"type":"final","text":"rollback abhi mat karo, pehle p95 check karlo"}  // exactly one, after end
{"type":"meta","model_ids":["..."],"local_only":true}            // optional, unscored audit
```

There is **no `t_ms`, no `seq`, no `pcm_b64`**. Every scored timing is the
evaluator's own monotonic **receive** clock — you cannot self-report latency.

---

## 3. Scoring — one combined score, 100 points

| Metric | Weight | Definition |
|---|---:|---|
| Final meaning / fidelity | 40 | `judge_meaning(gold, final)` (reused from `scorecard.py`) — on the Hindi-English mix |
| Critical facts & terms | 20 | `critical_flip(gold, final, must_have)` (reused) — numbers / "not" / names / required terms |
| End-to-final latency | 25 | median over 5 runs of `final_recv − last_audio_sent` |
| Stable-partial latency (TTFS) | 5 | median over 5 runs of time to the first *useful committed* partial |
| Revision churn | 5 | `5 · (1 − min(1, revision_churn))` |
| Streaming reliability | 5 | no blank final / no loop / no drop / no hang across all 5 runs |

Roughly: **~60% accuracy + Hindi-English-mix correctness** (meaning + facts),
**~35% live dictation feel** (end-to-final is the main latency axis, plus TTFS
and churn), **~5% reliability**.

**Quality (meaning + facts) is judged ONCE**, on the final of the
**median-latency run** (the run whose end-to-final equals the median). Only the
latency axes (end-to-final, TTFS) use the 5-run median.

The implementation in [`streaming_scorecard.py`](../streaming_scorecard.py) **is**
this spec — it reuses all quality logic from `scorecard.py` and adds only the
streaming machinery (churn, TTFS usefulness, latency curves, caps).

### 3.1 Latency → points (pre-registered independent of RambleFix)

**The goal:** come close to Wispr Flow's *feel* — a good final within **~2 seconds** of you
stopping. ~2s sits inside Wispr's real-world felt latency (1–2s) but is achievable locally and
offline; today's best local engines (RambleFix included) sit around ~3.5s, so ~2s is a real,
meaningful cut, not a vanity bar. We do NOT claim Wispr's ~700ms cloud lab number — sub-1s is the
stretch ceiling, not the bar.

End-to-final (25 pts):
```
<= 1000ms        -> 25   (Wispr-class stretch ceiling)
1000–2000ms      -> linear 25 → 20   (~2s = the realistic target: strong score)
2000–3500ms      -> linear 20 → 10   (~3.5s ≈ today's best local — middling, beatable)
3500–5000ms      -> linear 10 → 3
> 5000ms         -> 0
```
TTFS (5 pts): `<=1000ms → 5`; `1000–2500ms → linear 5→2`; `>2500ms or undefined → 0`.
Churn (5 pts): `5 · (1 − min(1, revision_churn))`.

These knees were committed **before** RambleFix's streaming numbers were
computed (pre-registration; see §6).

### 3.2 Revision churn (length-stable)

Committed tokens are `normalize(text[:stable_chars])` — the **same tokenizer** as
the batch scorecard, so casing never counts as a change. Churn counts committed
tokens that were later dropped/changed (including those that didn't survive into
the final), **normalized by total committed tokens emitted** (not by final
length). A long "ramble" clip and a short clip with identical absolute thrash get
the same churn — you can't dilute it with a long final. A solution that never
commits (`stable_chars:0` always) gets churn 0.0 but earns 0 TTFS and trips the
no-useful-partial cap — silence is not a dodge.

### 3.3 TTFS "useful" partial

A committed prefix `C = normalize(text[:stable_chars])` is **useful** when
`len(C) >= 3` tokens **and** the token edit distance between `C` and the matching
gold prefix is `<= 1`. Junk early commits (`"the"`, `stable_chars:0`) are not
useful — they neither satisfy TTFS nor dodge the no-useful-partial cap. TTFS is
the receive time of the first useful partial minus the run's audio-start; median
over the 5 runs. If a run has no useful partial, its TTFS is undefined; if ≥ half
the runs are undefined, the metric is 0 and the cap fires.

### 3.4 Hard caps (per clip; clip score = `min(base, cap)`, most-severe wins)

| Condition | Cap |
|---|---:|
| No partial before `end` (any run) | 70 |
| No useful committed partial ever (TTFS undefined on ≥ half the runs) | 70 |
| Median end-to-final > 4000ms (slower than today's best local) | 80 |
| Median end-to-final > 6000ms | 50 |
| Critical fact flip on the final | 50 |
| Blank final / repetition loop / WER > 0.9 | 20 / 30 / 20 (reused from `scorecard.py`) |
| `revision_churn > 0.5` | 60 |
| Connection drop / hang on a timed clip | that clip = 0 |
| Non-loopback socket during the timed run | FAIL (`offline_guard`) |

Run score = mean over clips.

---

## 4. The harness (how you're fed)

[`evaluator.py`](../evaluator.py) launches your sealed server on a free loopback
port, warms up on one unscored clip, calls `block_network()` (cloud blocked;
loopback stays open), then per clip:

1. Feeds the WAV at **1x real time** as 20ms binary PCM frames, absolute-time
   paced so jitter never accumulates.
2. Captures partial/final events on a monotonic receive clock.
3. Repeats **5 times** with **per-run anti-replay jitter** (±0.5 dB gain, ±0.3%
   resample, deterministic per (clip, run)) — defeats fingerprint memoization
   across the 5 warm serial runs without changing the words.
4. **Pace-sanity floor:** total send time must be within ±5% of the clip
   duration; an out-of-band run is discarded and re-run, never scored.
5. Latency/TTFS = 5-run median; quality = the median-latency run's final.

There is **no partial-causality / foreknowledge detector** — buffering is already
non-profitable (audio arrives only at 1x; end-to-final is measured from the last
frame), and "batch in costume" is handled by the 70 cap, not a detector.

---

## 5. Frozen box (pre-registered)

The official run is on a **single frozen machine** — every entry (and the RambleFix
benchmark line) is timed on the same box under identical conditions, so speed is
about the code, not whose laptop is faster. It is a **representative work laptop**:
because the goal is a good final within ~2s (Wispr-class *feel*, locally), the
machine's on-device accelerator (GPU / Apple Neural Engine) is available to the
scored process — outbound **network is blocked** after model warmup. The exact
machine is pinned here at launch and does not change for the round:

> **Frozen box (pinned):** MacBook Pro 14-inch (2021) · Apple M1 Pro · 8-core CPU
> (6P + 2E) + GPU + 16-core Neural Engine · 32 GB RAM · macOS Tahoe 26.3.1 ·
> accelerator on, **network blocked** after model warmup. The §3.1 latency knees are
> calibrated to this machine.

---

## 6. RambleFix — the benchmark line (not a prize competitor)

RambleFix is the engine you're trying to beat. It is a **benchmark line only** and
is **ineligible for the $500**. Its streaming shape: a warm local `whisper.cpp`
small server drafts the rolling audio prefix, commits the stable common
word-prefix, pastes the latest draft immediately at key-up, and an async
code-switch-faithful Hinglish finalizer replaces the final when it lands. The
clean wrapper that runs it through this exact public harness is
[`baseline/ramblefix_stream.py`](../baseline/ramblefix_stream.py).

**Published RambleFix streaming numbers** (real measured, on a local Mac / Apple Silicon,
preliminary — first streaming approximation; the §3.1 curve knees were committed **before**
these were computed):

| Axis | RambleFix (measured) | Notes |
|---|---|---|
| Release-to-paste (latest draft) | **~50ms** | paste happens immediately after key-up |
| Release-to-final (async Hinglish finalizer) | **~3.3–3.5s** (p50), ~3.6s p95 | the finalizer, not on the paste path |
| Time-to-first-partial | **~1.5–2.5s** | the next bottleneck — first useful draft is still too late |
| Hindi-English mix (final) | WER ~**0.12**, meaning ~**0.84** | ~8× more faithful than cloud/free tools, which translate the mix away |

These were measured with the RambleFix streaming lab on real-time WAV feeds
(release-to-paste ~50ms; release-to-final ~3.3–3.5s p50; first-partial ~1.5–2.5s).
They are the **bar to beat**: push the **mix past meaning 0.90** while staying
faithful, and get the **first useful partial under ~1s** and end-to-final fast —
that's where the $500 is won.

> **Order constraint (pre-registration proof):** the §3.1 curve knees are finalized
> and committed first; RambleFix's streaming numbers are computed and published
> second, in the same release.
