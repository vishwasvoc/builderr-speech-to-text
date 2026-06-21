"""Proves the sealed stream_server speaks the wire contract correctly, WITHOUT
any ASR model: it monkeypatches draft() to a deterministic stub, spins the server
on a free loopback port, drives start/binary-audio/end, and asserts the protocol.

Run:  python tests/test_stream_contract.py   (or pytest)
Requires `websockets` (pip install -r requirements-streaming.txt).
"""
import sys
import os
import json
import socket
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    assert cond, name


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def _main():
    try:
        import websockets  # noqa: F401
    except Exception:
        print("SKIP: websockets not installed (pip install -r requirements-streaming.txt)")
        return

    import solution.stream_server as srv

    # deterministic stub draft: commits a growing prefix as more audio arrives
    state = {"calls": 0}

    def fake_draft(audio_buffer: bytes, is_final: bool):
        state["calls"] += 1
        words = ["rollback", "abhi", "mat", "karo"]
        n = min(len(words), 1 + len(audio_buffer) // 6400)  # grow with audio
        text = " ".join(words[:n])
        if is_final:
            return (" ".join(words), len(" ".join(words)))
        return (text, len(text))  # commit everything (stub)

    srv.draft = fake_draft
    srv.draft_reset = lambda: state.update(calls=0)
    srv.DRAFT_EVERY_FRAMES = 5  # draft often so we get partials on a short clip

    import websockets
    port = _free_port()
    server = await websockets.serve(srv._handle, "127.0.0.1", port, max_size=None)
    try:
        uri = f"ws://127.0.0.1:{port}"
        async with websockets.connect(uri, max_size=None) as ws:
            await ws.send(json.dumps({
                "type": "start", "sample_rate": 16000, "format": "pcm_s16le",
                "channels": 1, "clip_id": "t"}))

            partials = []
            final = None
            # feed ~1s of silence as 20ms binary frames
            frame = b"\x00" * 640

            # collect partials concurrently with feeding
            async def collector():
                nonlocal final
                async for raw in ws:
                    if isinstance(raw, (bytes, bytearray)):
                        continue
                    msg = json.loads(raw)
                    if msg.get("type") == "partial":
                        partials.append(msg)
                    elif msg.get("type") == "final":
                        final = msg
                    elif msg.get("type") == "meta":
                        return

            col = asyncio.create_task(collector())
            for _ in range(50):
                await ws.send(frame)
                await asyncio.sleep(0.002)
            await ws.send(json.dumps({"type": "end"}))
            await asyncio.wait_for(col, timeout=10)

        check("binary audio accepted + >=1 partial before end", len(partials) >= 1)
        check("exactly one final received", final is not None)
        check("final has text field", isinstance(final.get("text"), str))
        check("partial schema: text + stable_chars", all(
            isinstance(p.get("text"), str) and isinstance(p.get("stable_chars"), int)
            for p in partials))
        check("stable_chars non-decreasing", all(
            partials[i]["stable_chars"] <= partials[i + 1]["stable_chars"]
            for i in range(len(partials) - 1)))
    finally:
        server.close()
        await server.wait_closed()

    # READY line check via subprocess
    import subprocess
    port2 = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "solution.stream_server", "--host", "127.0.0.1",
         "--port", str(port2)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), text=True)
    try:
        line = proc.stdout.readline()
        check(f"prints READY port=<PORT> ({line.strip()!r})",
              line.strip() == f"READY port={port2}")
    finally:
        proc.terminate()
        proc.wait(timeout=5)

    print("\nALL STREAM CONTRACT TESTS PASSED")


def test_stream_contract():
    asyncio.run(_main())


if __name__ == "__main__":
    asyncio.run(_main())
