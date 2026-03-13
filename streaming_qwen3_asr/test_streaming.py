#!/usr/bin/env python3

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming_qwen3_asr import StreamingQwen3ASR, StreamingState, register_qwen3_asr


def test_streaming():
    model_dir = "./model/tokenizer"
    conv_frontend = "./model/conv_frontend.onnx"
    encoder = "./model/encoder.onnx"
    decoder = "./model/decoder.onnx"
    wav_path = "./test_wavs/far_3.wav"

    print("Loading model...")
    asr = StreamingQwen3ASR(
        model_dir=model_dir,
        conv_frontend_path=conv_frontend,
        encoder_path=encoder,
        decoder_path=decoder,
        device="cpu",
    )

    print("Loading audio...")
    import scipy.io.wavfile
    rate, data = scipy.io.wavfile.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        wav = data.astype(np.float32) / 32768.0
    else:
        wav = data.astype(np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    print(f"Audio: {wav.shape[0]} samples, {wav.shape[0]/16000:.2f}s")

    print("\n=== Testing streaming (chunk by chunk) ===")
    chunk_size_sec = 2.0
    state = asr.init_streaming_state(
        chunk_size_sec=chunk_size_sec,
    )

    chunk_size = int(chunk_size_sec * 16000)
    num_chunks = (wav.shape[0] + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, wav.shape[0])
        chunk = wav[start:end]
        state = asr.streaming_transcribe(chunk, state)
        print(f"Chunk {i+1}/{num_chunks}: {state.text}")

    print("\n=== Testing finish ===")
    state = asr.finish_streaming_transcribe(state)
    print(f"Final: {state.text}")

    print("\n=== Expected output ===")
    print("周末要不要去露营？最近天气超舒服。露营？我怕虫子咬，而且晚上睡帐篷，会不会很冷啊？放心，我借了专业装备，还有暖宝宝，再带点火锅食材，边吃边看星星超惬意。")


if __name__ == "__main__":
    test_streaming()
