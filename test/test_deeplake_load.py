#!/usr/bin/env python3
"""
DeepLakeストリーミング版ImageNetの使用例

従来の方法:
- data/image_net/ ディレクトリに150GB+のImageNetデータが必要
- ストレージを大量消費

DeepLake版:
- ストレージ消費なし（ストリーミング）
- サブセットサイズを簡単に調整可能
- 事前準備不要

使用方法:
python test/test_deeplake_load.py
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from dataset import get_dataset


def main():
    print("=== DeepLake ImageNet ストリーミング例 ===")

    # 従来の方法（ローカルファイルが必要）
    print("\n1. 従来の方法（ローカルファイル版）:")
    try:
        train_dataset_local = get_dataset("imagenet", "train", use_deeplake=False)
        print(f"✓ ローカル訓練データセット: {len(train_dataset_local)} samples")
    except Exception as e:
        print(f"✗ ローカルデータが見つかりません: {e}")

    # DeepLake版（ストリーミング）
    print("\n2. DeepLake版（ストリーミング）:")
    try:
        # 40,000枚のサブセットを使用
        train_dataloader_deeplake = get_dataset(
            "imagenet", "train", use_deeplake=True, deeplake_subset=40000
        )
        print(f"✓ DeepLake訓練データローダー作成完了")

        # 数バッチのサンプルデータをテスト
        print("\n3. サンプルデータのテスト:")
        batch_count = 0
        for batch_data in train_dataloader_deeplake:
            images, labels = batch_data
            print(
                f"  バッチ {batch_count + 1}: 画像サイズ={images.shape}, ラベル={labels}"
            )
            batch_count += 1
            if batch_count >= 3:  # 3バッチだけテスト
                break

        print(f"\n✓ DeepLakeストリーミング動作確認完了！")
        print("メリット:")
        print("  - ローカルストレージ消費: 0GB")
        print("  - データ準備時間: 0分")
        print("  - サブセットサイズ調整: 簡単")

    except ImportError:
        print("✗ DeepLakeがインストールされていません")
        print("  pip install deeplake でインストールしてください")
    except Exception as e:
        print(f"✗ DeepLakeエラー: {e}")


if __name__ == "__main__":
    main()
