#!/usr/bin/env python3
"""Drive the reverse-proxied StreamDiffusion server through an orchestrator.

The app speaks no Livepeer protocol — the orchestrator reverse-proxies its own
HTTP/WS endpoints — so all the integration is here, and it is three calls:

  1. reserve_session()      — discover the runner and reserve a (metered) session
  2. ws_connect()           — drive the app's native protocol over the proxied app_url
  3. stop_runner_session()  — release the session (settles payment on-chain)

Everything else is the app's own protocol, reached at `session.app_url`:

  WS  /api/ws/{uuid}      input : control messages + JPEG frames
  GET /api/stream/{uuid}  output: MJPEG; opening it builds the pipeline and
                                  drives the server's per-frame pump
  POST /api/blending      set the prompt (not a per-frame field here)

MJPEG in on stdin, MJPEG out on stdout, so ffmpeg does the capture and playback:

  ffmpeg ... -f image2pipe -c:v mjpeg - | uv run client.py | ffplay -f mjpeg -i -
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from contextlib import suppress

import aiohttp

import prompts
from livepeer_gateway.errors import LivepeerGatewayError
from livepeer_gateway.live_runner import stop_runner_session
from livepeer_gateway.selection import reserve_session

DEFAULT_DISCOVERY = "https://localhost:8935/discovery"
APP_ID = "livepeer/streamdiffusion"  # keep in step with runners.json
# Per-frame params the server expects; prompt/seed/steps are set over REST instead.
FRAME_PARAMS = {"resolution": "512x512 (1:1)", "width": 512, "height": 512}
SOI, EOI = b"\xff\xd8", b"\xff\xd9"  # JPEG start/end-of-image markers

log = logging.getLogger("streamdiffusion-client")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restyle an MJPEG stream (stdin -> stdout) on the Livepeer network."
    )
    parser.add_argument("--discovery", default=DEFAULT_DISCOVERY)
    parser.add_argument(
        "--prompt", default="", help="Pin one prompt (disables auto-cycling)."
    )
    parser.add_argument(
        "--prompt-interval",
        type=float,
        default=60.0,
        help="Seconds between auto-cycled prompts.",
    )
    parser.add_argument(
        "--output", default="-", help="MJPEG output file, or - for stdout."
    )
    parser.add_argument(
        "--signer", default="", help="Remote signer base URL (on-chain/paid path)."
    )
    return parser.parse_args()


def _split_jpegs(buf: bytearray) -> list[bytes]:
    """Pull every complete JPEG out of buf, leaving the partial tail behind."""
    frames: list[bytes] = []
    while True:
        start = buf.find(SOI)
        if start < 0:
            break
        end = buf.find(EOI, start + 2)
        if end < 0:
            del buf[:start]  # drop bytes before the SOI; keep the partial frame
            break
        frames.append(bytes(buf[start : end + 2]))
        del buf[: end + 2]
    return frames


async def _stdin_reader() -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_running_loop().connect_read_pipe(
        lambda: protocol, sys.stdin.buffer
    )
    return reader


async def _read_input(reader: asyncio.StreamReader, state: dict) -> None:
    # Keep only the LATEST input frame; the WS loop sends it on demand. Dropping to
    # latest is what keeps a slow diffuser from building an input backlog.
    buf = bytearray()
    while chunk := await reader.read(65536):
        buf += chunk
        for frame in _split_jpegs(buf):
            state["latest"] = frame


async def _set_prompt(http: aiohttp.ClientSession, base: str, prompt: str) -> None:
    with suppress(Exception):
        await http.post(f"{base}/api/blending", json={"prompt_list": [[prompt, 1.0]]})
        log.info("prompt -> %r", prompt)


async def _cycle_prompts(
    http: aiohttp.ClientSession, base: str, interval: float
) -> None:
    """Hands-free art-style slideshow: rotate through the curated bank."""
    prompt = None
    while True:
        await asyncio.sleep(interval)
        prompt = prompts.random_prompt(prompt)
        await _set_prompt(http, base, prompt)


async def _publish(ws: aiohttp.ClientWebSocketResponse, state: dict) -> None:
    # The server drives the cadence: to each "send_frame" (from the output stream's
    # pump) we answer {"status":"next_frame"} -> params -> the latest input JPEG.
    # Empty bytes when there is none yet: the server simply asks again.
    async for msg in ws:
        if msg.type != aiohttp.WSMsgType.TEXT:
            continue
        status = msg.json().get("status")
        if status == "send_frame":
            await ws.send_json({"status": "next_frame"})
            await ws.send_json(FRAME_PARAMS)
            await ws.send_bytes(state.get("latest") or b"")
        elif status in ("timeout", "error"):
            log.info("server ended the stream: %s", msg.data)
            break


async def _read_output(http: aiohttp.ClientSession, base: str, uid: str, write) -> None:
    # Opening the stream lazily builds the pipeline (the first run compiles TensorRT
    # engines, which takes minutes), then drives the frame pump. Hence no read timeout.
    url = f"{base}/api/stream/{uid}"
    async with http.get(
        url, timeout=aiohttp.ClientTimeout(total=None, sock_read=None)
    ) as response:
        log.info("output stream open: %s (%s)", url, response.status)
        buf = bytearray()
        async for chunk in response.content.iter_chunked(65536):
            buf += chunk
            for frame in _split_jpegs(buf):  # JPEGs straight out of the multipart body
                write(frame)


async def _stream(session, args: argparse.Namespace, write) -> None:
    base = session.app_url.rstrip("/")
    uid = str(uuid.uuid4())  # the server validates the path as a real UUID
    auto = not args.prompt.strip()

    # ssl=False: the orchestrator proxy serves a self-signed cert on localhost.
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as http:
        await _set_prompt(http, base, args.prompt.strip() or prompts.random_prompt())
        log.info(
            "auto-cycling every %ss" % args.prompt_interval if auto else "prompt pinned"
        )
        # Connect the WS up front so a failure surfaces before anything else starts.
        async with http.ws_connect(  # Livepeer: 2
            f"{base}/api/ws/{uid}", max_msg_size=0
        ) as ws:
            state: dict = {"latest": None}
            reader = await _stdin_reader()
            tasks = [
                asyncio.create_task(_read_input(reader, state)),  # ends at stdin EOF
                asyncio.create_task(_read_output(http, base, uid, write)),
                asyncio.create_task(_publish(ws, state)),
            ]
            if auto:
                tasks.append(
                    asyncio.create_task(
                        _cycle_prompts(http, base, args.prompt_interval)
                    )
                )
            # End of input ends the run, and so does either half of the stream
            # failing — otherwise a dead socket would leave us waiting on stdin.
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()  # re-raise whatever ended the run, if it was an error


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args()
    to_stdout = args.output.strip() in {"-", "stdout"}
    out = sys.stdout.buffer if to_stdout else open(args.output, "wb")

    def write(jpeg: bytes) -> None:
        out.write(jpeg)
        if to_stdout:
            out.flush()

    session = None
    try:
        session = await reserve_session(  # Livepeer: 1
            discovery_url=args.discovery,  # omit if the signer does discovery itself
            app=APP_ID,
            signer_url=args.signer.strip() or None,
        )
        log.info("session_id=%s app_url=%s", session.session_id, session.app_url)
        # The session funds itself while it is held, and the meter runs for as long
        # as this block does — here, the lifetime of the stream.
        async with session:
            await _stream(session, args, write)
    except LivepeerGatewayError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    finally:
        if not to_stdout:
            out.close()
        if session is not None:
            with suppress(Exception):
                await stop_runner_session(session)  # Livepeer: 3


if __name__ == "__main__":
    asyncio.run(main())
