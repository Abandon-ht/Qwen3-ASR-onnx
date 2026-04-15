#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

pulsar2 build --config config_Qwen3-ASR-0.6B-conv_frontend_u16.json --input conv_frontend.onnx --output_dir . --output_name conv_frontend.axmodel
