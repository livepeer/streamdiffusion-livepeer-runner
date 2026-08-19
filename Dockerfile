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

# Pure pip: the tensorrt + cuDNN wheels, polygraphy, onnx-graphsurgeon.
RUN python -m streamdiffusion.tools.install-tensorrt

# No system CUDA tree, so point the loader at the wheels' lib dirs. Globbed
# rather than hardcoded: the paths move on every version bump.
RUN python - <<'PY'
import glob, os, site
dirs = sorted(d for r in site.getsitepackages()
              for pat in ("nvidia/*/lib", "tensorrt_libs", "tensorrt/lib")
              for d in glob.glob(r + "/" + pat))
open("/etc/ld.so.conf.d/nvidia-wheels.conf", "w").write(chr(10).join(dirs) + chr(10))
# Wheels ship only versioned SONAMEs, but parts of the TensorRT path dlopen the
# plain "libcudart.so", so recreate the symlinks the -dev packages gave us.
for d in dirs:
    for so in sorted(glob.glob(d + "/*.so.*"), key=len):
        base = so.split(".so.")[0] + ".so"
        if not os.path.exists(base):
            os.symlink(os.path.basename(so), base)
            print("linked", base)
print(*dirs, sep=chr(10))
PY
RUN ldconfig

# Fail the build, not the first stream: install-tensorrt pins a cuDNN that can
# disagree with torch's, and there is no system copy to fall back on.
RUN python - <<'PY'
import ctypes, glob, site, sys
import torch, tensorrt
print("torch", torch.__version__, "/ cuda", torch.version.cuda, "/ tensorrt", tensorrt.__version__)
libs = [p for r in site.getsitepackages() for p in glob.glob(r + "/nvidia/cudnn/lib/libcudnn.so.*")]
if not libs:
    sys.exit("cuDNN wheel not found")
ctypes.CDLL(sorted(libs)[0])
print("cudnn loadable:", sorted(libs)[0])
for soname in ("libcudart.so", "libnvinfer.so"):
    ctypes.CDLL(soname)
    print("dlopen ok:", soname)
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
# /models/engines is made at start because the volume masks a build-time mkdir,
# and mkdir(exist_ok=True) re-raises on the symlink while it dangles.
CMD ["sh", "-c", "mkdir -p /models/engines && exec python main.py --host=0.0.0.0 --port=7860 --acceleration=tensorrt --api-only"]
