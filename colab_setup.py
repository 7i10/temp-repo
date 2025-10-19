#!/usr/bin/env python
"""
Colab セットアップスクリプト
Google Colab で訓練環境を構築します
"""

import subprocess
import sys
import os


def run_command(cmd, description=""):
    """コマンド実行とエラーハンドリング"""
    if description:
        print(f"\n{'='*60}")
        print(f"📦 {description}")
        print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True)
        print(f"✅ {description} 完了")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失敗: {e}")
        return False


def main():
    
    # 1. GPU確認
    print("\n🔍 GPU確認...")
    run_command("nvidia-smi", "NVIDIA GPU情報取得")
    
    # 2. pip アップグレード
    run_command(
        "pip install --upgrade pip setuptools wheel",
        "pip / setuptools / wheel アップグレード"
    )
    
    # 3. PyTorch CUDA 12.4版インストール
    run_command(
        "pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124",
        "PyTorch 2.4.0 (CUDA 12.4) インストール"
    )
    
    # 4. コア依存パッケージ
    core_packages = [
        "accelerate",
        "transformers",
        "peft",
        "safetensors",
    ]
    run_command(
        f"pip install {' '.join(core_packages)}",
        "コア依存パッケージインストール"
    )
    
    # 5. Diffusers（最新版）
    run_command(
        "pip install git+https://github.com/huggingface/diffusers.git",
        "Diffusers（開発版）インストール"
    )
    
    # 6. 攻撃・評価ライブラリ
    attack_packages = [
        "torchattacks",
        "advex_uar==0.0.6.dev0",
        "lpips==0.1.4",
    ]
    run_command(
        f"pip install {' '.join(attack_packages)}",
        "攻撃・評価ライブラリインストール"
    )
    
    # 7. Auto-attack
    run_command(
        "pip install git+https://github.com/fra31/auto-attack.git",
        "Auto-Attack（開発版）インストール"
    )
    
    # 8. その他ユーティリティ
    utility_packages = [
        "datasets==2.19.1",
        "deeplake",
        "xformers==0.0.27.post2",
        "torchmetrics==0.6.0",
        "torcheval==0.0.7",
        "python-dotenv",
        "colorama==0.4.6",
    ]
    run_command(
        f"pip install {' '.join(utility_packages)}",
        "ユーティリティパッケージインストール"
    )
    
    # 9. 環境変数設定
    print("\n" + "="*60)
    print("📝 環境変数設定")
    print("="*60)
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("✅ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    
    # 10. 動作確認
    print("\n" + "="*60)
    print("✔️ 動作確認")
    print("="*60)
    
    confirmation_code = """
import torch
import transformers
import diffusers
import accelerate
import torchattacks
import xformers

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Transformers: {transformers.__version__}")
print(f"Diffusers: {diffusers.__version__}")
print(f"Accelerate: {accelerate.__version__}")
print(f"TorchAttacks: {torchattacks.__version__}")
print(f"\\n✅ セットアップ完了！訓練準備完了です")
"""
    
    run_command(
        f"python -c \"{confirmation_code}\"",
        "パッケージバージョン確認"
    )
    
    print("\n" + "="*60)
    print("🎉 セットアップ完了！")
    print("="*60)

if __name__ == "__main__":
    main()
