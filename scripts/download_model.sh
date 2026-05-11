#!/usr/bin/env bash
# scripts/download_model.sh
#
# Downloads a pre-trained MobileNetV2 plant-disease TFLite model.
#
# Option A: Use Kaggle PlantVillage-trained model (recommended).
# Option B: Download a generic MobileNetV2 and fine-tune yourself.
#
# This script downloads a quantised MobileNetV2 from TF Hub as a
# placeholder.  Replace with your fine-tuned model for production.

set -euo pipefail

MODEL_DIR="$(dirname "$0")/../ai"
MODEL_PATH="${MODEL_DIR}/disease_model.tflite"

echo "==> Downloading MobileNetV2 TFLite model …"

# ── Option A: Your own model hosted somewhere ────────────────────────────────
# Uncomment and fill in your URL:
# curl -L "https://your-storage.example.com/disease_model.tflite" \
#      -o "${MODEL_PATH}"

# ── Option B: Generic MobileNetV2 (ImageNet) as placeholder ─────────────────
# This will classify ImageNet classes, not diseases!
# Replace after fine-tuning on PlantVillage.
curl -L \
  "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v2_1.0_224_quant.tflite" \
  -o "${MODEL_PATH}"

echo "==> Model saved to ${MODEL_PATH}"
echo ""
echo "NOTE: The downloaded model is a generic ImageNet classifier."
echo "      Fine-tune it on the PlantVillage dataset for real disease detection."
echo "      See: https://www.tensorflow.org/lite/guide/model_maker"
