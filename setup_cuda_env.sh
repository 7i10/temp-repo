#!/bin/bash
# CUDA 11.8対応のPyTorchをインストールするスクリプト

set -e

echo "=== Setting up CUDA 11.8 GPU environment ==="

# 環境の作成
echo "Creating conda environment..."
mamba clean -a --yes
mamba env remove -n 7i10 --yes
mamba env create -f env-cuda118.yml --yes

# PyTorchのインストール（CUDA 11.8版）
echo "Installing PyTorch 2.2.2 with CUDA 11.8 support..."
conda activate 7i10
pip cache purge
pip install --force-reinstall --no-cache-dir \
  torch==2.2.2 \
  torchvision==0.17.2 \
  torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu118

# GPU認識確認
echo ""
echo "=== Verifying GPU setup ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"

echo ""
echo "=== Setup complete! ==="
echo "To activate the environment, run: conda activate 7i10"
