# =============================================================================
# data-core: CUDA + Python base image
#
# Base image provides CUDA, cuDNN, and Ubuntu 22.04 dev tools.
# Install lerobot yourself into /opt/venv after building the container.
#
# Apptainer usage:
#   apptainer build data_core.sif docker-daemon://data-core:latest
#   apptainer run --nv \
#     --bind /path/to/dataset:/workspace/data \
#     --bind /path/to/outputs:/workspace/outputs \
#     data_core.sif \
#     python3 examples/train_act_lazy.py --dataset_path data/my_dataset
# =============================================================================

# NVIDIA marks this CUDA 12.1 tag as deprecated. Any Ubuntu 22.04 based image
# works; override with --build-arg BASE_IMAGE=... if the tag is removed.
ARG BASE_IMAGE=nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
FROM ${BASE_IMAGE}

USER root

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.10 (native in Ubuntu 22.04)
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    # C / C++ toolchain: Rust system linker and C runtime
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    pkg-config \
    # FFmpeg dev headers (FFmpeg 4.4 on Ubuntu 22.04)
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libavfilter-dev \
    libavdevice-dev \
    libswscale-dev \
    libswresample-dev \
    # libclang for bindgen (ffmpeg-sys-next generates its bindings at build time)
    clang \
    libclang-dev \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip

# Rust stable toolchain installed into /opt so it is world-readable
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --no-modify-path \
    && chmod -R a+rX /opt/rustup /opt/cargo

RUN pip install "maturin[patchelf]>=1.0,<2.0"

WORKDIR /workspace
COPY . .

# VIRTUAL_ENV points maturin at /opt/venv, so no activation is needed.
RUN maturin develop --release

CMD ["bash"]
