#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

pulsar2 build --config config_Qwen3-ASR-0.6B-decoder_u16.json --input decoder.onnx --output_dir . --output_name decoder.axmodel
