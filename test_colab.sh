#!/bin/bash
# Google Colab用テスト実行スクリプト
# 訓練済みLoRAモデルの敵対的ロバスト性を評価

# ============================================================================
# 使用方法:
#   bash test_colab.sh
# または Colab セルで:
#   !bash test_colab.sh
# ============================================================================

# エラーが発生した場合は途中で終了
set -e

# 出力ディレクトリの設定
OUTPUT_DIR="./test_results"
LOG_DIR="$OUTPUT_DIR/logs"

# ディレクトリ作成
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "======================================================================"
echo "LoRA Model Adversarial Robustness Testing for Colab"
echo "======================================================================"
echo "Output Directory: $OUTPUT_DIR"
echo "Log Directory: $LOG_DIR"
echo ""

# ============================================================================
# テスト設定
# ============================================================================

# チェックポイントパス（ローカル訓練済みモデル）
CHECKPOINT_DIR="./logs/OSCP/checkpoint-1250/unet_lora"

# または Hugging Face Hub からのモデルを使用
# CHECKPOINT_DIR="your_username/your_model_name"

# ============================================================================
# テストケース1: 低ノイズ強度（高精度復元）
# ============================================================================
echo ""
echo "========== Test Case 1: Low Noise Strength =========="
echo "Parameters:"
echo "  - Strength: 0.1 (低ノイズ)"
echo "  - Inference Steps: 5"
echo "  - Validation Samples: 500"
echo ""

python test.py \
    --model="LCM" \
    --output_dir="$OUTPUT_DIR" \
    --num_validation_set=500 \
    --lora_input_dir="$CHECKPOINT_DIR" \
    --strength=0.1 \
    --num_inference_step=5 \
    --device="cuda:0" \
    --attack_method="Linf_pgd" \
    --guidance_scale=1.0 \
    --control_scale=0.8 \
    --classifier="resnet50" \
    --use_deeplake \
    --deeplake_subset=5000 \
    2>&1 | tee "$LOG_DIR/test_case_1_strength_0.1.log"

# ============================================================================
# テストケース2: 中程度のノイズ強度（バランス型）
# ============================================================================
echo ""
echo "========== Test Case 2: Medium Noise Strength =========="
echo "Parameters:"
echo "  - Strength: 0.2 (中程度ノイズ)"
echo "  - Inference Steps: 4"
echo "  - Validation Samples: 500"
echo ""

python test.py \
    --model="LCM" \
    --output_dir="$OUTPUT_DIR" \
    --num_validation_set=500 \
    --lora_input_dir="$CHECKPOINT_DIR" \
    --strength=0.2 \
    --num_inference_step=4 \
    --device="cuda:0" \
    --attack_method="Linf_pgd" \
    --guidance_scale=1.0 \
    --control_scale=0.8 \
    --classifier="resnet50" \
    --use_deeplake \
    --deeplake_subset=5000 \
    2>&1 | tee "$LOG_DIR/test_case_2_strength_0.2.log"

# ============================================================================
# テストケース3: 高ノイズ強度（高堅牢性評価）
# ============================================================================
echo ""
echo "========== Test Case 3: High Noise Strength =========="
echo "Parameters:"
echo "  - Strength: 0.3 (高ノイズ)"
echo "  - Inference Steps: 3"
echo "  - Validation Samples: 500"
echo ""

python test.py \
    --model="LCM" \
    --output_dir="$OUTPUT_DIR" \
    --num_validation_set=500 \
    --lora_input_dir="$CHECKPOINT_DIR" \
    --strength=0.3 \
    --num_inference_step=3 \
    --device="cuda:0" \
    --attack_method="Linf_pgd" \
    --guidance_scale=1.0 \
    --control_scale=0.8 \
    --classifier="resnet50" \
    --use_deeplake \
    --deeplake_subset=5000 \
    2>&1 | tee "$LOG_DIR/test_case_3_strength_0.3.log"

# ============================================================================
# テストケース4: 異なる攻撃方法（AutoAttack）
# ============================================================================
echo ""
echo "========== Test Case 4: AutoAttack Method =========="
echo "Parameters:"
echo "  - Strength: 0.2"
echo "  - Inference Steps: 4"
echo "  - Attack Method: AutoAttack (より強力な攻撃)"
echo "  - Validation Samples: 200 (計算負荷が高いため減少)"
echo ""

python test.py \
    --model="LCM" \
    --output_dir="$OUTPUT_DIR" \
    --num_validation_set=200 \
    --lora_input_dir="$CHECKPOINT_DIR" \
    --strength=0.2 \
    --num_inference_step=4 \
    --device="cuda:0" \
    --attack_method="AutoAttack" \
    --guidance_scale=1.0 \
    --control_scale=0.8 \
    --classifier="resnet50" \
    --use_deeplake \
    --deeplake_subset=5000 \
    2>&1 | tee "$LOG_DIR/test_case_4_autoattack.log"

# ============================================================================
# 結果のサマリー
# ============================================================================
echo ""
echo "======================================================================"
echo "✅ All tests completed!"
echo "======================================================================"
echo ""
echo "📊 Results saved to:"
echo "   - Output: $OUTPUT_DIR"
echo "   - Logs: $LOG_DIR"
echo ""
echo "📁 Generated files:"
find "$OUTPUT_DIR" -type f -name "stat.csv" | head -5
echo ""
echo "📝 Log files:"
ls -lh "$LOG_DIR"/*.log
echo ""
echo "======================================================================"
echo "🔍 Next Steps:"
echo "   1. Check the stat.csv files for quantitative results"
echo "   2. Review generated images in output directories"
echo "   3. Compare performance across different strength values"
echo "======================================================================"
