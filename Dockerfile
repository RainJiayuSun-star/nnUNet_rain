FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Keep logs visible in real time and avoid bytecode clutter.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install minimal OS dependencies commonly needed by medical imaging stacks.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/nnunet

# Copy your forked nnU-Net source and install it in editable mode.
COPY . /opt/nnunet
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -e .

# Default nnU-Net data roots (override with -e or compose env as needed).
ENV nnUNet_raw=/workspace/nnunet_data/nnUNet_raw \
    nnUNet_preprocessed=/workspace/nnunet_data/nnUNet_preprocessed \
    nnUNet_results=/workspace/nnunet_data/nnUNet_results

# Optional workspace for mounted datasets and outputs.
WORKDIR /workspace

# Run commands explicitly, e.g.:
# docker run --gpus all -it --rm \
#   -v /path/to/nnunet_data:/workspace/nnunet_data \
#   nnunet-rain nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["python -c \"import torch; print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())\""]
