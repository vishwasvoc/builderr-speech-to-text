"""builderr · STREAMING dictation track — scoring engine (ONE combined score).

This scores a *streaming* dictation run: the entrant's `draft()` emits text as
audio arrives (partials with a committed prefix), then one final after the user
stops. We score on the live-dictation experience, not just the final string.

ONE $500 prize, ONE combined weighted score out of 100:

    Final meaning / fidelity (Hindi-English mix)   40
    Critical facts & terms (numbers / "not" / names) 20   --> ~60% accuracy + mix
    End-to-final latency (release -> final)         25
    Stable-partial latency (TTFS)                    5    --> ~35% live feel
    Revision churn (did committed text get rewritten) 5
    Streaming reliability (no blank / loop / drop)   5    --> ~5% reliability

QUALITY (meaning + facts) is judged ONCE, on the final of the median-latency run.
LATENCY axes (end-to-final, TTFS) use the median over the 5 anti-replay runs.

All quality logic is REUSED from scorecard.py — this module never reimplements
WER / meaning / term / normalize. It adds only the streaming-specific machinery:
the churn metric, the TTFS "useful partial" judgement, the latency->points curves,
and the per-clip hard caps.

Hard caps (per clip; clip score = min(base, cap)):
    no partial before end ......................... 70
    no useful committed partial ever (TTFS gone) .. 70
    median end-to-final > 4000ms .................. 80   (slower than today's best local)
    median end-to-final > 6000ms .................. 50
    critical fact flip on the final .............. 50
    blank final / repetition loop / WER > 0.9 .... 20 / 30 / 20  (reused from scorecard)
    revision_churn > 0.5 ......................... 60
    connection drop / hang on a timed clip ....... 0  (that whole clip)

Pure standard library. No t_ms / seq / pcm_b64 anywhere — every timing comes from
the evaluator's monotonic receive clock (see evaluator.py / RunCapture).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

# REUSE the batch quality engine verbatim — do not reimplement any of this.
from scorecard import (
    judge_meaning,
    critical_flip,
    has_repetition_loop,
    wer,
    normalize,
    LATENCY_BAR_MS,  # shared 5s reliability bar (see scorecard.py)
)

# --- composite weights (one prize, one combined score) -------------------
W_MEANING = 40.0     # final meaning / fidelity on the mix
W_FACTS = 20.0       # numbers / negation / required terms on the final
W_END_TO_FINAL = 25.0
W_TTFS = 5.0
W_CHURN = 5.0
W_RELIABILITY = 5.0
assert (W_MEANING + W_FACTS + W_END_TO_FINAL + W_TTFS + W_CHURN + W_RELIABILITY) == 100.0

# --- caps ----------------------------------------------------------------
CAP_NO_PARTIAL = 70.0
CAP_NO_USEFUL_PARTIAL = 70.0
CAP_SLOW_FINAL = 80.0          # median end-to-final > 4s (slower than today's best local)
CAP_VERY_SLOW_FINAL = 50.0     # median end-to-final > 6s (sluggish)
CAP_FACT_FLIP = 50.0
CAP_BLANK = 20.0
CAP_LOOP = 30.0
CAP_UNRELATED = 20.0
CAP_HIGH_CHURN = 60.0
CHURN_CAP_THRESHOLD = 0.5


# =========================================================================
# revision churn — single runnable definition (normalized tokens, normalized
# by committed-token volume so it is length-stable across short and long clips)
# =========================================================================

def _committed_tokens(text: str, stable_chars: int) -> list[str]:
    """Normalized tokens of the prefix the solution COMMITTED to (promised not to
    rewrite). Uses scorecard.normalize so casing never counts as a change."""
    return normalize(text[: max(0, stable_chars)])


def _common_prefix_len(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def revision_churn(partials: list[tuple[float, str, int]], final_text: str) -> float:
    """0.0 = the solution never retracted a token it had committed.

    Normalized by the TOTAL committed tokens emitted (not by final length), so a
    long 'ramble' clip and a short clip with identical absolute thrash score the
    same — you can't dilute churn by producing a long final.

    A solution that never commits (stable_chars==0 always) gets churn 0.0 but
    earns 0 TTFS and trips the no-useful-partial cap — it can't dodge by silence.
    """
    history = [_committed_tokens(t, sc) for (_, t, sc) in partials]
    final_toks = normalize(final_text)
    retracted = 0
    total_committed = 0
    prev: list[str] = []
    for cur in history:
        lcp = _common_prefix_len(prev, cur)
        retracted += len(prev) - lcp          # tokens prev had committed that cur dropped/changed
        total_committed += max(0, len(cur) - lcp)  # newly committed tokens this step
        prev = cur
    # committed tokens that did not survive into the final
    retracted += len(prev) - _common_prefix_len(prev, final_toks)
    denom = max(1, total_committed)
    return retracted / denom


# =========================================================================
# TTFS — time to first *useful* committed partial (reuses normalize)
# =========================================================================

def _token_edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def is_useful_partial(text: str, stable_chars: int, gold: str) -> bool:
    """A committed prefix is 'useful' when it is real, committed, and right:
      - >= 3 committed tokens, AND
      - token edit distance to the matching gold prefix is <= 1.
    Junk early commits ("the", stable_chars=0) are NOT useful, so they can't
    satisfy TTFS or dodge the no-useful-partial cap."""
    committed = _committed_tokens(text, stable_chars)
    if len(committed) < 3:
        return False
    gold_toks = normalize(gold)
    n = min(len(committed), len(gold_toks)) if gold_toks else 0
    ref = gold_toks[: len(committed)] if len(committed) <= len(gold_toks) else gold_toks
    # if committed longer than gold by >1 it can't be within edit distance 1
    if len(committed) > len(gold_toks) + 1:
        return False
    return _token_edit_distance(committed, ref) <= 1


def first_useful_ttfs(run, gold: str) -> float | None:
    """Receive-time of the first useful partial minus the run's audio start.
    Returns None (undefined) if no useful partial appeared in this run."""
    t_start = run.get("t_start")
    for (recv, text, sc) in run.get("partials", []):
        if is_useful_partial(text, sc, gold):
            return None if t_start is None else max(0.0, (recv - t_start) * 1000.0)
    return None


def end_to_final_ms(run) -> float | None:
    """ms from last audio frame sent to the final being received. None if dropped."""
    final = run.get("final")
    if not final or run.get("dropped"):
        return None
    recv, _text = final
    t_end = run.get("t_end_audio")
    if t_end is None:
        return None
    return max(0.0, (recv - t_end) * 1000.0)


# =========================================================================
# latency -> points curves (CPU-calibrated; pre-registered, RambleFix-independent)
# =========================================================================

def end_to_final_points(median_ms: float | None) -> float:
    if median_ms is None:
        return 0.0
    if median_ms <= 1000:
        return W_END_TO_FINAL                            # 25 — Wispr-class stretch ceiling
    if median_ms <= 2000:
        return _lerp(median_ms, 1000, 2000, 25.0, 20.0)  # ~2s target band: strong, and realistic
    if median_ms <= 3500:
        return _lerp(median_ms, 2000, 3500, 20.0, 10.0)  # toward today's best-local baseline (~3.5s)
    if median_ms <= 5000:
        return _lerp(median_ms, 3500, 5000, 10.0, 3.0)
    return 0.0


def ttfs_points(median_ms: float | None) -> float:
    if median_ms is None:
        return 0.0
    if median_ms <= 1000:
        return W_TTFS                                # 5
    if median_ms <= 2500:
        return _lerp(median_ms, 1000, 2500, 5.0, 2.0)
    return 0.0


def churn_points(churn: float) -> float:
    return W_CHURN * (1.0 - min(1.0, churn))


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return y1
    frac = (x - x0) / (x1 - x0)
    return y0 + (y1 - y0) * max(0.0, min(1.0, frac))


# =========================================================================
# per-clip + run scoring
# =========================================================================

@dataclass
class StreamClipResult:
    clip_id: str
    score: float
    capped_at: float | None
    reasons: list[str] = field(default_factory=list)
    # diagnostics
    meaning: float = 0.0
    wer: float = 0.0
    median_end_to_final_ms: float | None = None
    median_ttfs_ms: float | None = None
    churn: float = 0.0
    reliability_ok: bool = True
    components: dict = field(default_factory=dict)


def _median(vals: list[float]) -> float | None:
    return None if not vals else float(statistics.median(vals))


def score_stream_clip(clip) -> StreamClipResult:
    """clip: {clip_id, gold, must_have?, runs:[RunCapture,...]} (5 runs).

    Each run is a dict with keys:
        t_start (float|None)      monotonic when 'start' was sent
        t_end_audio (float|None)  monotonic when last audio frame was sent
        partials [(recv,text,stable_chars), ...]
        final (recv, text) | None
        dropped bool
    """
    clip_id = clip.get("clip_id", "")
    gold = clip.get("gold", "")
    must_have = clip.get("must_have") or []
    runs = clip.get("runs") or []
    reasons: list[str] = []

    # ---- per-run latency captures -------------------------------------
    e2f = [v for v in (end_to_final_ms(r) for r in runs) if v is not None]
    ttfs_vals = [v for v in (first_useful_ttfs(r, gold) for r in runs) if v is not None]
    med_e2f = _median(e2f)
    med_ttfs = _median(ttfs_vals)

    # ---- choose the median-latency run for QUALITY --------------------
    median_run = _median_latency_run(runs)
    final_text = ""
    if median_run is not None and median_run.get("final") and not median_run.get("dropped"):
        final_text = median_run["final"][1] or ""

    # ---- churn (computed on the median-latency run) -------------------
    churn = (
        revision_churn(median_run.get("partials", []), final_text)
        if median_run is not None
        else 0.0
    )

    # ---- quality (reused scorecard logic) -----------------------------
    m = judge_meaning(gold, final_text)
    w = wer(gold, final_text)
    flipped, fr = critical_flip(gold, final_text, must_have)
    reasons += fr

    meaning_pts = W_MEANING * m
    facts_pts = 0.0 if flipped else W_FACTS
    e2f_pts = end_to_final_points(med_e2f)
    ttfs_pts = ttfs_points(med_ttfs)
    churn_pts = churn_points(churn)

    # ---- reliability across ALL runs ----------------------------------
    any_dropped = any(r.get("dropped") for r in runs)
    blank_final = not normalize(final_text)
    loop = has_repetition_loop(final_text)
    no_partial_any = any(not r.get("partials") for r in runs)
    no_useful = (sum(1 for r in runs if first_useful_ttfs(r, gold) is None) >= max(1, (len(runs) + 1) // 2))
    reliability_ok = not (any_dropped or blank_final or loop or no_partial_any)
    rel_pts = W_RELIABILITY if reliability_ok else 0.0
    if any_dropped:
        reasons.append("connection drop/hang on a timed run")
    if no_partial_any:
        reasons.append("no partial before end (some run)")
    if no_useful:
        reasons.append("no useful committed partial (TTFS undefined on >=half of runs)")

    base = meaning_pts + facts_pts + e2f_pts + ttfs_pts + churn_pts + rel_pts

    # ---- hard caps (most-severe wins) ---------------------------------
    cap = None

    def _apply(c, why):
        nonlocal cap
        if c is not None:
            cap = c if cap is None else min(cap, c)
            reasons.append(why)

    if any_dropped:
        _apply(0.0, "clip=0: connection drop/hang")
    if blank_final:
        _apply(CAP_BLANK, "blank final")
    elif loop:
        _apply(CAP_LOOP, "repetition loop")
    elif w > 0.9:
        _apply(CAP_UNRELATED, f"final unrelated to audio (WER {w:.2f})")
    if flipped:
        _apply(CAP_FACT_FLIP, "critical fact flip on final")
    if no_partial_any:
        _apply(CAP_NO_PARTIAL, "no-partial cap")
    if no_useful:
        _apply(CAP_NO_USEFUL_PARTIAL, "no-useful-partial cap")
    if med_e2f is not None and med_e2f > 6000:
        _apply(CAP_VERY_SLOW_FINAL, f"median end-to-final {med_e2f:.0f}ms > 6000ms")
    elif med_e2f is not None and med_e2f > 4000:
        _apply(CAP_SLOW_FINAL, f"median end-to-final {med_e2f:.0f}ms > 4000ms")
    if churn > CHURN_CAP_THRESHOLD:
        _apply(CAP_HIGH_CHURN, f"revision churn {churn:.2f} > {CHURN_CAP_THRESHOLD}")

    score = min(base, cap) if cap is not None else base

    return StreamClipResult(
        clip_id=clip_id,
        score=round(score, 2),
        capped_at=cap,
        reasons=reasons,
        meaning=round(m, 3),
        wer=round(w, 3),
        median_end_to_final_ms=None if med_e2f is None else round(med_e2f, 1),
        median_ttfs_ms=None if med_ttfs is None else round(med_ttfs, 1),
        churn=round(churn, 3),
        reliability_ok=reliability_ok,
        components={
            "meaning": round(meaning_pts, 2),
            "facts": round(facts_pts, 2),
            "end_to_final": round(e2f_pts, 2),
            "ttfs": round(ttfs_pts, 2),
            "churn": round(churn_pts, 2),
            "reliability": round(rel_pts, 2),
        },
    )


def _median_latency_run(runs):
    """The run whose end-to-final equals the median (lower-middle on ties).
    Falls back to the first run with a final if no run has measurable latency."""
    scored = [(end_to_final_ms(r), r) for r in runs]
    measurable = [(v, r) for (v, r) in scored if v is not None]
    if not measurable:
        for r in runs:
            if r.get("final") and not r.get("dropped"):
                return r
        return runs[0] if runs else None
    measurable.sort(key=lambda vr: vr[0])
    return measurable[(len(measurable) - 1) // 2][1]


def score_stream_run(clips) -> dict:
    """clips: list of clip dicts (see score_stream_clip). Returns overall /100
    (mean over clips) plus per-clip diagnostics — same aggregation as batch."""
    results = [score_stream_clip(c) for c in clips]
    n = len(results) or 1

    def _avg(getter):
        vals = [getter(r) for r in results if getter(r) is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    return {
        "overall_score": round(sum(r.score for r in results) / n, 2),
        "meaning_mean": round(sum(r.meaning for r in results) / n, 3),
        "wer_mean": round(sum(r.wer for r in results) / n, 3),
        "median_end_to_final_ms": _avg(lambda r: r.median_end_to_final_ms),
        "median_ttfs_ms": _avg(lambda r: r.median_ttfs_ms),
        "churn_mean": round(sum(r.churn for r in results) / n, 3),
        "reliability_ok_rate": round(sum(1 for r in results if r.reliability_ok) / n, 3),
        "clips_capped": sum(1 for r in results if r.capped_at is not None),
        "n": n,
        "clips": [r.__dict__ for r in results],
    }


if __name__ == "__main__":
    import json, sys
    clips = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else []
    print(json.dumps(score_stream_run(clips), indent=2))
