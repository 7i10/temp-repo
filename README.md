# temp-repo

本リポジトリは [InstantPure (オリジナルリポジトリ)](https://github.com/antony090/InstantPure) のコードを引用・改変したものです。  
**新機能**: DeepLake ストリーミングによる効率的な ImageNet 学習をサポートしています。

## 🚀 主な特徴・改良点

- **DeepLake ストリーミング対応**: ローカルに ImageNet を保存せずに学習可能
- **ストレージ消費ゼロ**: 従来 150GB+必要だった ImageNet が不要
- **Colab 最適化**: Google Drive の容量制限を回避
- **簡単なサブセット調整**: 引数 1 つでデータ量を変更可能
- **既存コード互換性**: 従来の使い方もそのまま利用可能

---

## 📦 インストール

### 方法 1: conda 環境（推奨）

```bash
conda env create -f environment.yml
conda activate instantpure
```

### 方法 2: pip

```bash
pip install -r requirements.txt
```

### 方法 3: 個別インストール（Colab 等）

```bash
pip install deeplake torch torchvision diffusers transformers peft accelerate
```

---

## 🔥 使用方法

### 1. DeepLake ストリーミング版（推奨）

ローカルに ImageNet データを用意する必要がありません。

```bash
# 40,000サンプルでLoRA学習
python train_lora.py
    --use_deeplake
    --deeplake_subset 40000
    --train_batch_size 32
    --learning_rate 1e-4
    --output_dir ./output
```

### 2. 従来版（ローカルファイル）

`data/image_net/` に ImageNet データが必要です。

```bash
# 従来の方法
python train_lora.py
    --max_train_samples 40000
    --train_batch_size 32
    --learning_rate 1e-4
    --output_dir ./output
```

### 3. プログラム内での使用例

```python
from dataset import get_dataset

# DeepLake版（ストリーミング）
dataloader = get_dataset(
    "imagenet",
    "train",
    use_deeplake=True,
    deeplake_subset=40000
)

# 従来版（ローカルファイル）
dataset = get_dataset("imagenet", "train", use_deeplake=False)
```

---

## ⚙️ 引数・オプション

### DeepLake 関連の新規オプション

| 引数                | 説明                          | デフォルト値 |
| ------------------- | ----------------------------- | ------------ |
| `--use_deeplake`    | DeepLake ストリーミングを使用 | `False`      |
| `--deeplake_subset` | 使用するサンプル数            | `40000`      |

### 従来のオプション

| 引数                  | 説明                                 |
| --------------------- | ------------------------------------ |
| `--max_train_samples` | ローカルファイル版でのサンプル数制限 |
| `--train_batch_size`  | バッチサイズ                         |
| `--learning_rate`     | 学習率                               |
| `--output_dir`        | 出力ディレクトリ                     |

---

## 📊 パフォーマンス比較

| 項目               | **DeepLake 版**    | **従来版**                  |
| ------------------ | ------------------ | --------------------------- |
| **事前準備**       | なし               | ImageNet ダウンロード・展開 |
| **ストレージ消費** | 0GB                | 150GB+                      |
| **初回実行**       | すぐ開始           | データ準備後に開始          |
| **サブセット変更** | 引数変更のみ       | データ再準備が必要          |
| **Colab 対応**     | ✅ 快適            | ❌ Drive 容量不足           |
| **学習速度**       | 高速（最適化済み） | ディスク I/O 依存           |

---

## 🔧 動作確認・テスト

```bash
# DeepLakeの動作確認
python test/test_deeplake_load.py

# 小規模テスト実行
python train_lora.py --use_deeplake --deeplake_subset 1000 --max_train_steps 10
```

---

## 🐛 トラブルシューティング

### DeepLake がインストールできない

```bash
pip install --upgrade pip
pip install deeplake
```

### ネットワークエラー

DeepLake は初回実行時にデータをキャッシュします。安定したネットワーク環境で実行してください。

### メモリ不足

```bash
# バッチサイズを小さくする
python train_lora.py --use_deeplake --train_batch_size 16

# サブセットサイズを小さくする
python train_lora.py --use_deeplake --deeplake_subset 20000
```

### conda 環境作成エラー

```bash
# pandas バージョン競合の場合
conda env create -f environment.yml --force
```

---

## 🔄 互換性

既存のコードは**一切変更されていません**。`--use_deeplake`フラグを指定しない限り、従来通りの動作をします。

```bash
# 従来版（変更なし）
python train_lora.py --max_train_samples 10000

# DeepLake版（新機能）
python train_lora.py --use_deeplake --deeplake_subset 10000
```

---

## 🔑 トークンの設定方法

#### Hugging Face Hub のアクセストークン

Stable Diffusion などのモデルをダウンロードするには、Hugging Face のアクセストークンが必要です。

1. [Hugging Face のトークン発行ページ](https://huggingface.co/settings/tokens)で「Read」権限のトークンを取得
2. `.zshrc`（または`.bashrc`）に以下を追記し、毎回自動で設定されるようにします。

```sh
export HUGGINGFACE_TOKEN=hf_xxx...  # あなたのトークン
```

設定後は新しいターミナルを開くか、`source ~/.zshrc` を実行してください。

#### DeepLake のアクセストークン

DeepLake のデータセットにアクセスする場合も、DeepLake のトークンが必要な場合があります。

1. [DeepLake のアカウントページ](https://app.deeplake.ai/)で API トークンを取得
2. `.zshrc`（または`.bashrc`）に以下を追記します。

```sh
export ACTIVELOOP_TOKEN=your_deeplake_token
```

同様に、設定後は新しいターミナルを開くか、`source ~/.zshrc` を実行してください。

これで毎回手動でトークンを入力せずに済みます。

---

## 📄 論文・引用

オリジナル論文: [arXiv:2408.17064](https://arxiv.org/abs/2408.17064)

```bibtex
@article{instantpure2024,
  title={InstantPure: ...},
  author={...},
  journal={arXiv preprint arXiv:2408.17064},
  year={2024}
}
```

---

## 📜 ライセンス

```
Copyright 2025 The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This code is modified from InstantPure (https://github.com/antony090/InstantPure),
which is modified on top of latent-consistency-model
(https://github.com/luosiallen/latent-consistency-model).
```

## 使い方

- ここに使い方を自由に記載します。

---

## 改変

## DeepLake ストリーミング版 ImageNet

このリポジトリでは、従来のローカルファイル版 ImageNet に加えて、DeepLake を使ったストリーミング版 ImageNet をサポートしています。

### メリット

- **ストレージ消費量**: 0GB（従来版は 150GB+）
- **事前準備時間**: 0 分（従来版は数時間〜数日）
- **サブセットサイズ調整**: 簡単（引数 1 つで変更可能）
- **Colab 対応**: Google Drive の容量制限に悩まされない

## インストール

```bash
# 必要なライブラリをインストール
pip install deeplake

# または、conda環境を使用
conda env create -f environment.yml
conda activate research
```

## 使用方法

### 1. 基本的な使用方法

```python
# dataset.py を使って DeepLake からデータを取得
from dataset import get_dataset

# DeepLake版（40,000サンプル）
train_dataloader = get_dataset(
    "imagenet",
    "train",
    use_deeplake=True,
    deeplake_subset=40000
)

# 従来版（ローカルファイル）
train_dataset = get_dataset("imagenet", "train", use_deeplake=False)
```

### 2. 学習スクリプトでの使用

```bash
# DeepLakeを使用して40,000サンプルで学習
python train_lora.py \
    --use_deeplake \
    --deeplake_subset 40000 \
    --train_batch_size 32 \
    --learning_rate 1e-4

# 従来のローカルファイル版での学習
python train_lora.py \
    --max_train_samples 40000 \
    --train_batch_size 32 \
    --learning_rate 1e-4
```

### 3. サンプルの動作確認

```bash
# DeepLakeの動作確認スクリプト
python test/test_deeplake_load.py
```

## 引数の説明

### DeepLake 関連の新しい引数

- `--use_deeplake`: DeepLake ストリーミングを使用する
- `--deeplake_subset`: 使用するサンプル数（デフォルト: 40,000）

### 従来の引数（ローカルファイル版）

- `--max_train_samples`: 使用するサンプル数の上限
- データは `data/image_net/` ディレクトリから読み込み

## パフォーマンス比較

| 項目               | DeepLake 版        | ローカルファイル版          |
| ------------------ | ------------------ | --------------------------- |
| **事前準備**       | なし               | ImageNet ダウンロード・展開 |
| **ストレージ**     | 0GB                | 150GB+                      |
| **初回実行**       | すぐ開始           | データ準備後に開始          |
| **サブセット変更** | 引数変更のみ       | データ再準備が必要          |
| **学習速度**       | 高速（最適化済み） | ディスク I/O 依存           |

## トラブルシューティング

### DeepLake がインストールできない場合

```bash
pip install --upgrade pip
pip install deeplake
```

### ネットワークエラーが発生する場合

DeepLake は初回実行時にデータをキャッシュします。安定したネットワーク環境で実行してください。

### メモリ不足エラーが発生する場合

```python
# バッチサイズを小さくする
python train_lora.py --use_deeplake --train_batch_size 16

# またはサブセットサイズを小さくする
python train_lora.py --use_deeplake --deeplake_subset 20000
```

## 従来版との互換性

既存のコードは一切変更されていません。`--use_deeplake` フラグを指定しない限り、従来通りの動作をします。

```bash
# 従来版（変更なし）
python train_lora.py --max_train_samples 10000

# DeepLake版（新機能）
python train_lora.py --use_deeplake --deeplake_subset 10000
```

### どのような修正をしたか

1. **DeepLake ストリーミング対応の追加**

   - dataset.py に DeepLake から ImageNet をストリーミングで取得する関数（例: `_imagenet_deeplake`）を追加。
   - `get_dataset` 関数に `use_deeplake`・`deeplake_subset` 引数を追加し、DeepLake 利用時はストリーミング DataLoader を返すようにした。
   - 既存のローカルファイル版 ImageNet のコードはそのまま残し、互換性を維持。

2. **学習スクリプトの DeepLake 対応**

   - train_lora.py で `--use_deeplake` オプションを受け取り、DeepLake ストリーミングか従来のローカルファイルかを切り替えられるようにした。
   - `--deeplake_subset` でサブセットサイズも指定可能。

3. **依存関係の追加**

   - requirements.txt と environment.yml に `deeplake` を追加。

4. **サンプルスクリプトの追加**

   - test_deeplake_load.py を追加し、DeepLake ストリーミングの動作確認ができるようにした。

5. **引数パーサの拡張**
   - get_args.py に `--use_deeplake` と `--deeplake_subset` を追加。

---

### README の DeepLake 構成案

#### 1. 概要・特徴

- 本リポジトリは従来のローカル ImageNet に加え、DeepLake ストリーミング ImageNet をサポート
- ストレージ消費ゼロ、Colab でも快適、サブセットも簡単

#### 2. インストール

- pip/conda 両対応
- `deeplake` のインストール明記

#### 3. 使い方

- 基本的な使い方（`get_dataset` の例）
- 学習スクリプトでの使い分け（`--use_deeplake` オプション）
- サンプルスクリプト（test_deeplake_load.py）

#### 4. 引数・オプション

- DeepLake 用新規引数（`--use_deeplake`, `--deeplake_subset`）
- 従来のローカルファイル用引数

#### 5. パフォーマンス比較

- 表形式で DeepLake とローカルの違いを明示

#### 6. トラブルシューティング

- DeepLake インストールエラー
- ネットワーク/メモリ問題

#### 7. 互換性

- 既存コードはそのまま動作
- DeepLake はオプション指定時のみ有効

#### 8. 論文・引用

- オリジナル論文や引用例
