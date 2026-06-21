"""builderr · local dictation challenge — scoring engine.

Scores a predicted transcript against human gold on the rubric in AGENT_BRIEF.md:

    Meaning accuracy        40   (LLM judge in production; deterministic proxy here)
    Critical facts & terms  25   (dates / numbers / negation / acronyms / names)
    Latency & reliability   20   (p50/p95 total ms, blanks, hangs)
    Local-only proof        10   (network blocked after warmup; local_only flag)
    Auditability             5   (raw candidates / route / timings / model ids)

Hard caps (per clip): a critical flip caps at 50; a blank at 20; a repetition
loop at 30; output unrelated to the audio (WER > 0.9) at 20.

Pure standard library so it runs anywhere with the network off. WER is exact;
the meaning score here is a deterministic token-F1 PROXY — production swaps in a
locked-prompt LLM judge (see judge_meaning()).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-z0-9']+")
_NEG = {"no", "not", "n't", "never", "without", "dont", "don't", "cannot", "can't", "nahi", "mat"}

# Shared latency bar (ms): batch counts a clip as a hang past this; the streaming
# track caps end-to-final past this. One bar, imported by streaming_scorecard.py.
LATENCY_BAR_MS = 5000


def normalize(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def wer(gold: str, pred: str) -> float:
    """Word error rate via token-level edit distance. 0.0 = perfect."""
    g, p = normalize(gold), normalize(pred)
    if not g:
        return 0.0 if not p else 1.0
    # Levenshtein on token lists
    prev = list(range(len(p) + 1))
    for i, gw in enumerate(g, 1):
        cur = [i]
        for j, pw in enumerate(p, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (gw != pw)))
        prev = cur
    return prev[-1] / len(g)


def token_f1(gold: str, pred: str) -> float:
    g, p = normalize(gold), normalize(pred)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    from collections import Counter
    cg, cp = Counter(g), Counter(p)
    overlap = sum((cg & cp).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def judge_meaning(gold: str, pred: str) -> float:
    """Meaning preservation in [0,1].

    PROXY: token-F1 (deterministic, offline, good enough to wire the harness and
    write tests). PRODUCTION: replace with a locked-prompt LLM judge that compares
    intent, run with the network blocked or on a cached judge. The judge prompt
    must be fixed and published; never let a transcript instruct the judge.
    """
    return token_f1(gold, pred)


# --- critical facts -------------------------------------------------------

_NUM = re.compile(r"\b\d[\d,.:/-]*\b")
# distinctive ALL-CAPS / mixed tokens like PRD, API, p95, Jira, Codex, names
_ENTITY = re.compile(r"\b([A-Z][a-zA-Z0-9]*[A-Z0-9][a-zA-Z0-9]*|[A-Z][a-z]+|p\d{2,})\b")


def extract_facts(text: str) -> dict[str, set]:
    t = text or ""
    nums = set(re.sub(r"[,]", "", n) for n in _NUM.findall(t))
    ents = set(m.group(0).lower() for m in _ENTITY.finditer(t))
    negs = set(w for w in normalize(t) if w in _NEG)
    return {"numbers": nums, "entities": ents, "negations": negs}


def critical_flip(gold: str, pred: str, must_have: list[str] | None = None) -> tuple[bool, list[str]]:
    """A 'flip' = a number, a required entity/term, or a negation present in gold
    but lost/changed in pred. Returns (flipped, reasons)."""
    gf, pf = extract_facts(gold), extract_facts(pred)
    reasons = []
    missing_nums = gf["numbers"] - pf["numbers"]
    if missing_nums:
        reasons.append(f"number changed/dropped: {sorted(missing_nums)}")
    # negation polarity: gold negated but pred isn't (or vice versa)
    if bool(gf["negations"]) != bool(pf["negations"]):
        reasons.append("negation polarity flipped")
    for term in (must_have or []):
        if term.lower() not in (pred or "").lower():
            reasons.append(f"required term missing: {term!r}")
    return (len(reasons) > 0, reasons)


def has_repetition_loop(pred: str, n: int = 3, k: int = 4) -> bool:
    """Detect degenerate n-gram loops (same n-gram repeated >= k times)."""
    toks = normalize(pred)
    if len(toks) < n * k:
        return False
    from collections import Counter
    grams = Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return max(grams.values(), default=0) >= k


# --- per-clip + run scoring ----------------------------------------------

@dataclass
class ClipResult:
    clip_id: str
    score: float            # 0..100 for this clip
    capped_at: float | None
    reasons: list[str] = field(default_factory=list)
    wer: float = 0.0
    meaning: float = 0.0


def score_clip(gold: str, pred: str, must_have: list[str] | None,
               timings_ms: dict | None, local_only: bool, audit_fields: dict | None,
               clip_id: str = "") -> ClipResult:
    reasons: list[str] = []
    w = wer(gold, pred)
    m = judge_meaning(gold, pred)

    # component points (clip-level, same weights as the run rubric)
    meaning_pts = 40 * m
    flipped, fr = critical_flip(gold, pred, must_have)
    facts_pts = 0.0 if flipped else 25.0
    reasons += fr
    # latency/reliability + local + audit are scored per-clip as full credit if
    # well-formed; the run aggregate re-weights latency across all clips.
    local_pts = 10.0 if local_only else 0.0
    if not local_only:
        reasons.append("local_only flag false")
    audit_pts = 5.0 if audit_fields and all(audit_fields.get(k) for k in ("model_ids",)) else 2.5
    latency_pts = 20.0  # placeholder; real latency credit comes from score_run

    base = meaning_pts + facts_pts + latency_pts + local_pts + audit_pts

    # hard caps
    cap = None
    pred_blank = not normalize(pred)
    if pred_blank:
        cap = 20.0; reasons.append("blank output")
    elif has_repetition_loop(pred):
        cap = 30.0; reasons.append("repetition loop")
    elif w > 0.9:
        cap = 20.0; reasons.append(f"output unrelated to audio (WER {w:.2f})")
    elif flipped:
        cap = 50.0
    score = min(base, cap) if cap is not None else base
    return ClipResult(clip_id, round(score, 2), cap, reasons, round(w, 3), round(m, 3))


def percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals); i = min(len(s) - 1, int(q * (len(s) - 1) + 0.5))
    return s[i]


def score_run(rows: list[dict]) -> dict:
    """rows: each {clip_id, gold, pred, must_have?, timings_ms?, local_only?, audit?}.
    Returns the aggregate /100 plus reliability + latency diagnostics."""
    clips = []
    lat = []
    blanks = hangs = 0
    for r in rows:
        t = r.get("timings_ms") or {}
        total = t.get("total")
        if total is not None:
            lat.append(total)
        if total is not None and total > LATENCY_BAR_MS:
            hangs += 1
        cr = score_clip(r["gold"], r.get("pred", ""), r.get("must_have"),
                        t, bool(r.get("local_only")), r.get("audit"), r.get("clip_id", ""))
        if cr.capped_at == 20.0 and "blank output" in cr.reasons:
            blanks += 1
        clips.append(cr)

    n = len(clips) or 1
    # latency credit: full 20 if p95<=5s, linear down to 0 at p95>=10s
    p95 = percentile(lat, 0.95)
    p50 = percentile(lat, 0.50)
    lat_credit = 20.0 if p95 <= 5000 else max(0.0, 20.0 * (10000 - p95) / 5000)

    # recompute each clip's latency component into the average (clips used 20 placeholder)
    avg_no_lat = sum(c.score for c in clips) / n  # already includes 20 placeholder
    overall = round(avg_no_lat - 20.0 + lat_credit, 2)
    return {
        "overall_score": overall,
        "useful_mean": round(sum(c.meaning for c in clips) / n, 3),
        "wer_mean": round(sum(c.wer for c in clips) / n, 3),
        "p50_ms": p50, "p95_ms": p95,
        "blank_rate": round(blanks / n, 3),
        "hang_rate": round(hangs / n, 3),
        "clips_capped": sum(1 for c in clips if c.capped_at is not None),
        "n": n,
        "clips": [c.__dict__ for c in clips],
    }


if __name__ == "__main__":
    import json, sys
    rows = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else []
    print(json.dumps(score_run(rows), indent=2))
