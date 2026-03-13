#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODEL_DIR="${MODEL_DIR:-./model/tokenizer}"
ONNX_DIR="${ONNX_DIR:-./model}"
PORT="${PORT:-8081}"
DEVICE="${DEVICE:-cpu}"

echo "Starting Qwen3-ASR Streaming Server..."
echo "  Model:     $MODEL_DIR"
echo "  ONNX:      $ONNX_DIR"
echo "  Port:      $PORT"
echo "  Device:    $DEVICE"

python streaming_qwen3_asr/realtime_ws_server.py \
    --model "$MODEL_DIR" \
    --conv_frontend "$ONNX_DIR/conv_frontend.onnx" \
    --encoder "$ONNX_DIR/encoder.int8.onnx" \
    --decoder "$ONNX_DIR/decoder.int8.onnx" \
    --vad_model "$ONNX_DIR/silero_vad.onnx" \
    --device "$DEVICE" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --chunk_size_sec 0.5
