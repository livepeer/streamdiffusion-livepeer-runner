# streamdiffusion-livepeer-runner

Realtime prompt-driven img2img on the **Livepeer network**: point a video feed at it and get back an ever-shifting AI restyle, with the prompt auto-cycling through a curated art-style bank. The app is daydream's [StreamDiffusion](https://github.com/daydreamlive/StreamDiffusion) `realtime-img2img` server, **run unmodified** — it contains no Livepeer code at all. An orchestrator hosts it, health-polls it, and acts as a **transparent reverse proxy**, so a client reaches the server's own WebSocket and MJPEG endpoints while the orchestrator handles discovery, sessions, and payment.

That is the point worth taking away: a third-party container you did not write, and cannot change, runs on the network as-is.

```sh
docker compose up -d --build
ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 640x480 -i /dev/video0 \
  -vf scale=512:512 -f image2pipe -c:v mjpeg -q:v 5 - \
  | uv run client.py \
  | ffplay -f mjpeg -fflags nobuffer -flags low_delay -i -
```

|              |                                            |
| ------------ | ------------------------------------------ |
| App id       | `livepeer/streamdiffusion`                 |
| Runner mode  | persistent (held-open session)             |
| Registration | static (`runners.json`)                    |
| Transport    | WebSocket + MJPEG (the app's own protocol) |
| Pricing      | hour (metered per second while held)       |
| Port         | 7860 (the StreamDiffusion server)          |

**Requires an NVIDIA GPU.** You also need **Docker** (with the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)), [**uv**](https://docs.astral.sh/uv/), and **ffmpeg** for capture and playback.

## How it's wired

[compose.yml](compose.yml) builds the server from [Dockerfile](Dockerfile) and starts an orchestrator with `-liveRunnerConfig`, pointed at [runners.json](runners.json). That file is the whole registration: it names the app, where to reach it (`http://app:7860`), and what to poll for liveness (`/api/queue`). The orchestrator proxies every other path straight through, so nothing has to be added to the container.

All the Livepeer integration therefore lives in [client.py](client.py). Grep `# Livepeer:` for the three calls:

1. `reserve_session` — reserve a session and get back the proxied `app_url`. Metered pricing starts the meter here; funding runs for as long as the session is held.
2. `ws_connect` — from here on it is the app's own protocol, over that url.
3. `stop_runner_session` — release the session, which settles payment on-chain.

### The protocol split

Input, output and prompt control are **three separate channels**, which is easy to forget: frames go up a WebSocket, frames come back down a plain HTTP response, and the prompt is a third call that touches neither.

| Channel  | Endpoint                 | Carries                                               |
| -------- | ------------------------ | ----------------------------------------------------- |
| Input    | `WS /api/ws/{uuid}`      | control messages plus input JPEG frames               |
| Output   | `GET /api/stream/{uuid}` | MJPEG out (`multipart/x-mixed-replace`)               |
| Prompt   | `POST /api/blending`     | prompt updates, at any time                           |
| Liveness | `GET /api/queue`         | the runner's `health_url`, polled by the orchestrator |

Two consequences worth knowing. **Opening the output stream is what builds the pipeline** and drives the per-frame pump, so nothing happens until you `GET` it, and the first open compiles TensorRT engines. And the same `{uuid}` ties the two halves together, so input and output are one session in two directions, not a request and a response.

```mermaid
sequenceDiagram
    participant C as client.py
    participant O as orchestrator
    participant A as StreamDiffusion (port 7860)

    Note over O,A: static registration: runners.json names the app,<br/>so the container needs no SDK
    loop every few seconds
        O->>A: GET /api/queue
    end

    C->>O: reserve_session("livepeer/streamdiffusion")
    O-->>C: proxied app_url, meter starts

    Note over C,A: from here it is the app's own protocol;<br/>the orchestrator only forwards
    C->>O: POST /api/blending
    O->>A: POST /api/blending
    C->>O: WS /api/ws/{uuid} + JPEG frames
    O->>A: WS /api/ws/{uuid}
    C->>O: GET /api/stream/{uuid}
    O->>A: GET /api/stream/{uuid}
    A-->>O: MJPEG
    O-->>C: MJPEG

    C->>O: stop_runner_session, settles on-chain
```

The client reads MJPEG on stdin and writes MJPEG on stdout, so ffmpeg does capture and playback and the client stays a pipe stage. Input frames are kept drop-to-latest: a slow diffuser falls behind rather than building a backlog.

## Prompts

By default the prompt rotates every `--prompt-interval` seconds (60) from [prompts.py](prompts.py), a **style × modifier** combinator: watercolor, ukiyo-e, cyberpunk neon, claymation, and so on. It is deliberately **not** an LLM: for anything you point at an audience you want deterministic, safe output, so every token is hand-vetted. Edit the lists to taste, or pin one style with `--prompt "van Gogh oil painting, vivid colors"`.

## Run offchain (free)

No wallet, no funds.

```sh
docker compose up -d --build
curl -sk https://localhost:8935/discovery | jq '.[].runners[].app'   # confirm it registered
```

The **first** stream open compiles TensorRT engines for your GPU. That takes minutes and is cached under `./models` afterwards, so expect a slow first frame exactly once.

Then restyle a webcam, cycling prompts hands-free:

```sh
ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 640x480 -i /dev/video0 \
  -vf scale=512:512 -f image2pipe -c:v mjpeg -q:v 5 - \
  | uv run client.py \
  | ffplay -f mjpeg -fflags nobuffer -flags low_delay -i -
```

Device numbers vary, so confirm your camera node first (`v4l2-ctl --list-devices`, `ffplay -f v4l2 -i /dev/videoN`). macOS: `-f avfoundation -i 0`. Windows: `-f dshow -i video="<name>"`.

Any MJPEG source works, so a file restyles too — but pace it with `-re`, or ffmpeg decodes the whole file at once and the run ends the moment the pipe closes:

```sh
ffmpeg -re -i clip.mp4 -vf scale=512:512 -f image2pipe -c:v mjpeg -q:v 5 - \
  | uv run client.py --output out.mjpeg
```

`--output` writes MJPEG to a file instead of stdout; `ffplay -f mjpeg -i out.mjpeg` plays it back.

```sh
docker compose down
```

## Run on-chain (paid)

Layer the overlay to add a remote signer and put the orchestrator on-chain, so the held session is metered and paid for per second. Beyond the offchain prerequisites you need:

- An **Ethereum RPC** (Arbitrum One by default).
- A **signer wallet** (the payer) with an on-chain deposit and reserve.
- An **orchestrator wallet** with ETH for gas to redeem tickets.
- Both as **keystore directories outside this repo**, mounted read-only.

```sh
cp .env.example .env   # fill in RPC, network, keystore paths, accounts, price cap
docker compose -f compose.yml -f compose.onchain.yml up -d --build
ffmpeg ... | uv run client.py --signer http://localhost:7936 | ffplay -f mjpeg -i -
docker compose -f compose.yml -f compose.onchain.yml down
```

The price is unchanged by the overlay: static runners advertise it from `runners.json` (`price_info.price`, USD per hour by default), which both compose files mount. Keep demo sessions short — the meter runs for as long as the socket is open.

> [!WARNING]
> The signer runs with `-remoteSignerAllowNoAuth`, which signs for anyone who can reach it and spends your deposit. That is fine on a laptop and wrong anywhere else: authorize callers with `-remoteSignerWebhookUrl` before exposing it.

## Ship it to an orchestrator

CI publishes the image to `ghcr.io/livepeer/streamdiffusion-livepeer-runner` on `main` and `v*` tags. Tags: `latest` (current `main`), `stable` (latest `v*` release), `1.2` / `1.2.3`, `sha-<short>`. The package is public, so pulling needs no account and no login. An operator then runs it with a `runners.json` like this repo's, pointed at wherever they run the container.

`docker compose up` always builds from source. To run the published image instead, which is the sane path unless you are changing the Dockerfile:

```sh
docker compose up -d --pull always
```

The image is **~15 GB** (torch, TensorRT, ONNX Runtime), which is still close enough to what a GitHub-hosted runner has free that [build.yml](.github/workflows/build.yml) reclaims disk before building and skips the build on pull requests. Building locally is `docker compose build`.

## Development

```sh
uvx pre-commit install      # format on commit
uvx pre-commit run --all-files
```

CI runs the same hooks, checks the compose file parses, and builds the image.

## License and attribution

This repo is an **example** of how to run StreamDiffusion on the [live runner](https://github.com/livepeer/go-livepeer/blob/master/doc/live-runner.md), not a production-ready pipeline. The wrapper here (Dockerfile, [client.py](client.py), the compose files, [runners.json](runners.json)) is MIT, and CI publishing to `ghcr.io/livepeer/` is packaging convenience so an operator can pull it, not a product commitment.

What runs inside the image is daydream's [StreamDiffusion](https://github.com/daydreamlive/StreamDiffusion), itself a fork of [cumulo-autumn/StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion). Both are **Apache-2.0**, and this repo builds the fork pinned at `94b9b96` and **unmodified**, so redistribution is permitted and there are no changes to state under section 4(b). The fork ships no `NOTICE` file; its `LICENSE` travels in the image at `/src/LICENSE`. Model weights are downloaded from Hugging Face on first run under their own licenses and are not redistributed here.

## Building your own

Start from [**template-livepeer-runner**](https://github.com/livepeer/template-livepeer-runner), then list yours in [**runner-app-examples**](https://github.com/livepeer/runner-app-examples#external-examples). That repo also has a minimal example of each transport, mode, registration, and pricing option; the [live runner docs](https://github.com/livepeer/go-livepeer/blob/master/doc/live-runner.md) are the reference.
