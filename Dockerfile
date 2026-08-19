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
# Base is plain python-slim, NOT nvidia/cuda:*-devel, because nothing here
# compiles against CUDA: torch carries its own runtime in the cu128 wheels, and
# streamdiffusion.tools.install-tensorrt is pure pip — it installs the tensorrt
# and nvidia-cudnn-cu12 wheels and reads torch.version.cuda, never nvcc.
# Upstream's devel base is a single commit from Dec 2023, back when xformers and
# stable-fast still built native extensions; this image installs neither. The
# driver arrives through the NVIDIA container runtime (compose `gpus: all`),
# which is how the other Livepeer runner examples get a GPU too.
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
# insightface (via the ipadapter extra) is sdist-only on PyPI, so it compiles a
# Cython/C++ extension here. That toolchain is the one thing the devel base was
# really providing; it is plain g++, nothing CUDA, so install it just for this
# step and purge it in the same layer rather than shipping it.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && python -m pip install --no-cache-dir \
        "streamdiffusion[tensorrt,controlnet,ipadapter] @ git+https://github.com/daydreamlive/StreamDiffusion.git@94b9b96cb8a17d401ffbce516393d6482326ce62" \
    && apt-get purge -y --auto-remove build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Pulls the tensorrt + cuDNN wheels, polygraphy and onnx-graphsurgeon. Pure pip:
# no compiler and no CUDA headers are involved.
RUN python -m streamdiffusion.tools.install-tensorrt

# With no system CUDA tree, the dynamic loader has to find cuDNN, cuBLAS and
# TensorRT inside the wheels. Register whatever lib dirs the wheels actually
# installed rather than hardcoding paths that move on every version bump.
RUN python - <<'PY'
import glob, site
dirs = sorted(d for r in site.getsitepackages()
              for pat in ("nvidia/*/lib", "tensorrt_libs", "tensorrt/lib")
              for d in glob.glob(r + "/" + pat))
open("/etc/ld.so.conf.d/nvidia-wheels.conf", "w").write(chr(10).join(dirs) + chr(10))
print(*dirs, sep=chr(10))
PY
RUN ldconfig

# Fail the build rather than the first stream if the CUDA stack cannot load.
# install-tensorrt pins nvidia-cudnn-cu12, which can disagree with the cuDNN
# torch wants, and on this base there is no system copy to fall back on.
RUN python - <<'PY'
import ctypes, glob, site, sys
import torch, tensorrt
print("torch", torch.__version__, "/ cuda", torch.version.cuda, "/ tensorrt", tensorrt.__version__)
libs = [p for r in site.getsitepackages() for p in glob.glob(r + "/nvidia/cudnn/lib/libcudnn.so.*")]
if not libs:
    sys.exit("cuDNN wheel not found")
ctypes.CDLL(sorted(libs)[0])
print("cudnn loadable:", sorted(libs)[0])
PY

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
#
# /models/engines is created here, not at build time: the volume mounts over
# /models at start and would mask a build-time mkdir. The symlink above would
# then be dangling, and Path.mkdir(exist_ok=True) re-raises on a dangling
# symlink, so the first stream dies with FileExistsError: 'engines'.
CMD ["sh", "-c", "mkdir -p /models/engines && exec python main.py --host=0.0.0.0 --port=7860 --acceleration=tensorrt --api-only"]
