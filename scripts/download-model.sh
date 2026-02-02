#!/bin/bash
set -e

MODEL_DIR="${MODEL_PATH:-/models/parakeet}"
MODEL_URL="https://huggingface.co/smcleod/parakeet-tdt-0.6b-v2-int8/resolve/main"

echo "=== Parakeet Model Downloader ==="
echo "Model directory: $MODEL_DIR"

mkdir -p "$MODEL_DIR"

download_file() {
    local url="$1"
    local output="$2"

    if [ -f "$output" ]; then
        echo "File already exists: $output"
        return 0
    fi

    echo "Downloading: $output"
    curl -L --progress-bar -o "$output" "$url"
}

echo ""
echo "Downloading Parakeet TDT 0.6B v2 (int8 quantized)..."
echo ""

# Download encoder model
download_file "${MODEL_URL}/encoder-model.int8.onnx" "${MODEL_DIR}/encoder.onnx"

# Download encoder external data (if exists)
download_file "${MODEL_URL}/encoder-model.int8.onnx.data" "${MODEL_DIR}/encoder.onnx.data" 2>/dev/null || true

# Download decoder/joint model
download_file "${MODEL_URL}/decoder_joint-model.int8.onnx" "${MODEL_DIR}/decoder.onnx"

# Download mel-spectrogram preprocessor
download_file "${MODEL_URL}/nemo128.onnx" "${MODEL_DIR}/preprocessor.onnx"

# Download tokenizer/vocabulary
download_file "${MODEL_URL}/tokenizer.json" "${MODEL_DIR}/tokenizer.json" 2>/dev/null || \
download_file "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/resolve/main/tokenizer.model" "${MODEL_DIR}/tokenizer.model" 2>/dev/null || true

# Try to get tokens.txt from sherpa-onnx version
if [ ! -f "${MODEL_DIR}/tokens.txt" ]; then
    echo "Downloading tokens.txt..."
    curl -L -o "${MODEL_DIR}/tokens.txt" \
        "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main/tokens.txt" 2>/dev/null || true
fi

echo ""
echo "=== Download Complete ==="
echo "Model files in $MODEL_DIR:"
ls -lh "$MODEL_DIR"
echo ""
