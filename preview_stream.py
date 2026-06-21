"""Local dev preview for the STREAMING dictation track — scores YOUR draft()
exactly like official admission, fully offline. Analogue of preview.py.

    pip install -r requirements.txt -r requirements-streaming.txt
    python preview_stream.py            # streams samples/ through solution.stream_server

It launches the sealed server (which calls your solution/draft.py), feeds the
sample clips at 1x real time, and prints the same streaming scorecard the hidden
set uses. The official run is identical but on hidden clips on a frozen CPU box.

See docs/STREAMING_CONTRACT.md for the contract, scoring, caps, and the published
RambleFix benchmark line.
"""
from __future__ import annotations
import asyncio
import os

HERE = os.path.dirname(os.path.abspath(__file__))
from evaluator import _evaluate  # noqa: E402


def main():
    manifest = os.path.join(HERE, "samples/manifest.json")
    res = asyncio.run(_evaluate(manifest, "solution.stream_server", runs=5,
                                enforce_offline=True))
    print(f"\n  streaming score   {res['overall_score']}/100")
    print(f"  meaning {res['meaning_mean']}   WER {res['wer_mean']}   churn {res['churn_mean']}")
    print(f"  median end-to-final {res['median_end_to_final_ms']}ms   median TTFS {res['median_ttfs_ms']}ms")
    print(f"  reliability-ok {res['reliability_ok_rate']}   clips capped {res['clips_capped']}/{res['n']}")
    for c in res["clips"]:
        flag = f"  capped@{c['capped_at']}" if c["capped_at"] else ""
        print(f"    {c['clip_id'][:28]:28s} score {c['score']:6}  e2f {c['median_end_to_final_ms']}ms"
              f"  ttfs {c['median_ttfs_ms']}ms{flag}  {';'.join(c['reasons'][:2])}")
    print("\n  (sample numbers are illustrative; the hidden set + your latency on the "
          "frozen CPU box rank you. The starter draft() scores low — that's your start line.)")


if __name__ == "__main__":
    main()
