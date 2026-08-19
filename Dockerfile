# Packaging-only image for daydream's StreamDiffusion realtime-img2img server.
#
# There is NO Livepeer code in here: the image runs the upstream FastAPI
# WebSocket server unmodified, and the orchestrator reverse-proxies it. That is
# the whole point of the static-registration path — an app needs no SDK, and no
# awareness of Livepeer at all, to run on the network.
#
# NOTE: the fork's demo/realtime-img2img/Dockerfile clones the UPSTREAM
# cumulo-autumn repo, not the fork, so it would not run this rewritten server.
# We clone the fork at a pinned commit instead.
#
# Base is python-slim, not nvidia/cuda:*-devel: nothing here compiles against
# CUDA (torch's cu128 wheels carry the runtime, install-tensorrt is pure pip),
# and the driver arrives through the container runtime.
FROM python:3.11-slim

LABEL org.opencontainers.image.title="streamdiffusion-livepeer-runner"
LABEL org.opencontainers.image.description="daydream's StreamDiffusion realtime-img2img server, packaged unmodified to run as a Livepeer live runner"
LABEL org.opencontainers.image.source="https://github.com/livepeer/streamdiffusion-livepeer-runner"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# git: pip resolves streamdiffusion from the fork, and demo/ is cloned below.
# libgl1/libglib2.0-0 are opencv's, not CUDA's, so slim needs them either way.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
        torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128
# insightface (ipadapter extra) is sdist-only, so it compiles here. Plain g++,
# nothing CUDA: install for this step only, purge in the same layer.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && python -m pip install --no-cache-dir \
        "streamdiffusion[tensorrt,controlnet,ipadapter] @ git+https://github.com/daydreamlive/StreamDiffusion.git@94b9b96cb8a17d401ffbce516393d6482326ce62" \
    && apt-get purge -y --auto-remove build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/tensorrt_libs

# Pure pip: the tensorrt + cuDNN wheels, polygraphy, onnx-graphsurgeon. polygraphy
# globs libcudart.so* over LD_LIBRARY_PATH, which on slim is the only place it
# will find one; the check fails the build rather than the first stream.
RUN python -m streamdiffusion.tools.install-tensorrt \
 && python -c "import torch,tensorrt; from polygraphy.cuda.cuda import Cuda; Cuda(); print('ok',torch.__version__,tensorrt.__version__)"

# The pip package doesn't ship the demo/ dir, so clone the fork (pinned) for the server.
RUN git clone https://github.com/daydreamlive/StreamDiffusion.git /src \
    && cd /src && git checkout 94b9b96cb8a17d401ffbce516393d6482326ce62

# Server deps for demo/realtime-img2img (the inference stack came with streamdiffusion).
RUN python -m pip install --no-cache-dir \
        fastapi==0.115.0 "uvicorn[standard]==0.32.0" markdown2 python-multipart PyYAML compel hf_transfer
# compel drags in a newer transformers/hub that drops MT5Tokenizer and breaks the
# diffusers import; pin back to the versions the streamdiffusion stack expects.
RUN python -m pip install --no-cache-dir \
        transformers==4.56.0 huggingface_hub==0.35.0

# Engines + HF cache under /models (mount a host dir so they persist / can be prebuilt).
# The demo writes engines to a RELATIVE ./engines dir (it ignores --engine-dir), so
# symlink that onto the mounted volume to persist compiled engines across rebuilds.
ENV HF_HOME=/models/hf HUGGINGFACE_HUB_CACHE=/models/hf
WORKDIR /src/demo/realtime-img2img
RUN rm -rf engines && ln -s /models/engines engines
EXPOSE 7860

# --api-only: skip the built Node frontend (a Livepeer client drives the API directly).
# /models/engines is made at start because the volume masks a build-time mkdir,
# and mkdir(exist_ok=True) re-raises on the symlink while it dangles.
CMD ["sh", "-c", "mkdir -p /models/engines && exec python main.py --host=0.0.0.0 --port=7860 --acceleration=tensorrt --api-only"]
