FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- system deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# --- install python deps first (better caching) ---
COPY requirements.txt /workspace/requirements.txt

RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel \
    && python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
       torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    && python3 -m pip install --no-cache-dir -r requirements.txt

# --- copy source code ---
COPY . /workspace

# default entry
CMD ["bash"]