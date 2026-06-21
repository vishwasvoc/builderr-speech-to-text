"""Proves the STREAMING scoring engine is fair, length-stable, and un-gameable.

Pure-function tests only (no server, no models). Run:
    pytest tests/test_streaming_scorecard.py
or  python tests/test_streaming_scorecard.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming_scorecard import (  # noqa: E402
    revision_churn,
    is_useful_partial,
    first_useful_ttfs,
    end_to_final_points,
    ttfs_points,
    churn_points,
    score_stream_clip,
    CAP_NO_PARTIAL,
    CAP_NO_USEFUL_PARTIAL,
    CAP_SLOW_FINAL,
    CAP_VERY_SLOW_FINAL,
    CAP_HIGH_CHURN,
)

GOLD = "rollback abhi mat karo pehle p95 check karlo"


def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    assert cond, name


def _good_run(e2f_ms, final_text, t_start=0.0):
    """A run with a useful early partial + a final at the given end-to-final ms."""
    t_end = 10.0
    return {
        "t_start": t_start,
        "t_end_audio": t_end,
        "partials": [(t_start + 0.5, "rollback abhi mat", 17)],
        "final": (t_end + e2f_ms / 1000.0, final_text),
        "dropped": False,
    }


def _clip(runs, gold=GOLD, must=None):
    return score_stream_clip({"clip_id": "c", "gold": gold, "must_have": must or [], "runs": runs})


def test_churn():
    # extend-only commits -> 0.0 churn
    p = [(0.0, "rollback abhi", 13), (0.1, "rollback abhi mat", 17),
         (0.2, "rollback abhi mat karo", 22)]
    check(f"extend-only churn == 0 ({revision_churn(p, 'rollback abhi mat karo')})",
          revision_churn(p, "rollback abhi mat karo") == 0.0)

    # case-only change is NOT churn (normalize lowercases)
    p_case = [(0.0, "rollback", 8)]
    check("case-only change is not churn",
          revision_churn(p_case, "Rollback abhi mat karo") == 0.0)

    # rewriting a committed token -> churn > 0
    p_rewrite = [(0.0, "rollback abhi", 13), (0.1, "rollback nahi", 13)]
    churn_rw = revision_churn(p_rewrite, "rollback nahi mat karo")
    check(f"rewriting committed token -> churn > 0 ({churn_rw})", churn_rw > 0)

    # length stability: identical absolute thrash, different final length => equal churn
    short_final = "alpha beta gamma"
    long_final = "alpha beta gamma " + " ".join(f"w{i}" for i in range(40))
    thrash = [(0.0, "alpha beta", 10), (0.1, "alpha ZZZZ", 10)]  # 1 committed token retracted
    c_short = revision_churn(thrash, short_final)
    c_long = revision_churn(thrash, long_final)
    check(f"churn length-stable across final length ({c_short} == {c_long})",
          abs(c_short - c_long) < 1e-9)

    # never-commit (stable_chars=0) -> churn 0 (but loses TTFS / trips cap, tested below)
    p_silent = [(0.0, "rollback abhi mat", 0), (0.1, "rollback abhi mat karo", 0)]
    check("never-commit churn == 0", revision_churn(p_silent, "rollback abhi mat karo") == 0.0)

    check("churn>1 clamps to 5*0 points", churn_points(2.0) == 0.0)
    check("churn 0 -> full 5 churn points", churn_points(0.0) == 5.0)


def test_ttfs_usefulness():
    # junk early commit: 1 token, stable_chars 0 -> not useful
    check("junk early partial not useful", not is_useful_partial("the", 0, GOLD))
    # a real committed 3-token prefix matching gold -> useful
    check("real committed prefix useful", is_useful_partial("rollback abhi mat", 17, GOLD))
    # committed but wrong (edit distance > 1) -> not useful
    check("wrong committed prefix not useful", not is_useful_partial("the weather today", 17, GOLD))

    # first_useful_ttfs picks first useful, in ms from t_start
    run = {"t_start": 0.0, "partials": [(0.4, "the", 0), (0.9, "rollback abhi mat", 17)]}
    check(f"ttfs = 900ms ({first_useful_ttfs(run, GOLD)})",
          abs(first_useful_ttfs(run, GOLD) - 900.0) < 1e-6)
    check("ttfs undefined when no useful partial",
          first_useful_ttfs({"t_start": 0.0, "partials": [(0.1, "x", 0)]}, GOLD) is None)


def test_latency_curves():
    check("e2f <=1000ms -> 25", end_to_final_points(800) == 25.0)
    check("e2f 1500ms -> between 20 and 25", 20.0 < end_to_final_points(1500) < 25.0)
    check("e2f exactly 2000ms -> 20 (target band)", abs(end_to_final_points(2000) - 20.0) < 1e-9)
    check("e2f exactly 3500ms -> 10 (~today's baseline)", abs(end_to_final_points(3500) - 10.0) < 1e-9)
    check("e2f exactly 5000ms -> 3", abs(end_to_final_points(5000) - 3.0) < 1e-9)
    check("e2f >5000ms -> 0", end_to_final_points(6000) == 0.0)
    check("ttfs <=1000ms -> 5", ttfs_points(700) == 5.0)
    check("ttfs >2500ms -> 0", ttfs_points(3000) == 0.0)


def test_caps():
    # perfect-ish fast clip scores high, no cap
    r = _clip([_good_run(800, GOLD) for _ in range(5)])
    check(f"fast faithful clip high, uncapped ({r.score})", r.capped_at is None and r.score >= 85)

    # no partial before end in some run -> cap 70
    runs = [_good_run(800, GOLD) for _ in range(5)]
    runs[0]["partials"] = []
    r = _clip(runs)
    check(f"no-partial cap 70 ({r.capped_at})", r.capped_at == CAP_NO_PARTIAL)

    # no useful partial on >= half the runs -> cap 70
    runs = [_good_run(800, GOLD) for _ in range(5)]
    for i in range(3):
        runs[i]["partials"] = [(0.4, "the", 0)]  # junk, not useful
    r = _clip(runs)
    check(f"no-useful-partial cap 70 ({r.capped_at})", r.capped_at == CAP_NO_USEFUL_PARTIAL)

    # median end-to-final > 4000ms (slower than today's best local) -> cap 80
    r = _clip([_good_run(4500, GOLD) for _ in range(5)])
    check(f"slow-final >4s cap 80 ({r.capped_at}, e2f {r.median_end_to_final_ms})",
          r.capped_at == CAP_SLOW_FINAL)

    # median end-to-final > 6000ms -> cap 50
    r = _clip([_good_run(6500, GOLD) for _ in range(5)])
    check(f"slow-final >6s cap 50 ({r.capped_at})", r.capped_at == CAP_VERY_SLOW_FINAL)

    # blank final -> cap 20
    r = _clip([_good_run(800, "") for _ in range(5)])
    check(f"blank final cap 20 ({r.capped_at})", r.capped_at == 20.0)

    # high churn ( > 0.5 ) -> cap 60
    thrash_run = _good_run(800, "rollback nahi karo")
    thrash_run["partials"] = [
        (0.5, "rollback abhi mat", 17),       # commits 3 tokens
        (0.7, "totally different text", 22),  # retracts all 3, commits 3 new
    ]
    r = _clip([dict(thrash_run) for _ in range(5)])
    check(f"high churn cap 60 ({r.capped_at}, churn {r.churn})",
          r.capped_at == CAP_HIGH_CHURN and r.churn > 0.5)

    # dropped connection -> clip score 0
    runs = [_good_run(800, GOLD) for _ in range(5)]
    runs[2]["dropped"] = True
    r = _clip(runs)
    check(f"dropped run -> clip 0 ({r.score})", r.score == 0.0)


def _run_all():
    test_churn()
    test_ttfs_usefulness()
    test_latency_curves()
    test_caps()
    print("\nALL STREAMING SCORECARD TESTS PASSED")


if __name__ == "__main__":
    _run_all()
