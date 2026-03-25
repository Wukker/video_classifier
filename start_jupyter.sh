#!/usr/bin/env bash
# Запуск JupyterLab с GPU-поддержкой для TensorFlow и PyTorch
# Использование: ./start_jupyter.sh

VENV_NVIDIA="$(dirname "$0")/.venv/lib/python3.12/site-packages/nvidia"

export LD_LIBRARY_PATH=\
$VENV_NVIDIA/cublas/lib:\
$VENV_NVIDIA/cudnn/lib:\
$VENV_NVIDIA/cuda_runtime/lib:\
$VENV_NVIDIA/cufft/lib:\
$VENV_NVIDIA/curand/lib:\
$VENV_NVIDIA/cusolver/lib:\
$VENV_NVIDIA/cusparse/lib:\
$VENV_NVIDIA/nccl/lib:\
$VENV_NVIDIA/nvjitlink/lib:\
${LD_LIBRARY_PATH}

echo "LD_LIBRARY_PATH установлен"
exec uv run jupyter lab "$@"
