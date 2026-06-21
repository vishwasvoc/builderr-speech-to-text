"""Runs the RambleFix baseline draft() through the SAME sealed wire protocol as
solution/stream_server.py, so the benchmark line is measured by the public
evaluator.py with no special-casing.

    python -m baseline.stream_server --host 127.0.0.1 --port <PORT>
    python evaluator.py --server-module baseline.stream_server --manifest samples/manifest.json

This reuses the sealed server's connection handler and just swaps in the baseline
draft function — it is NOT a place to put engine logic (that lives in
baseline/ramblefix_stream.py). No secrets, loopback only.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

# reuse the sealed harness internals, point them at the baseline draft
import solution.stream_server as harness
from baseline import ramblefix_stream

harness.draft = ramblefix_stream.draft
harness.draft_reset = ramblefix_stream.draft_reset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    try:
        asyncio.run(harness._serve(args.host, args.port))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
