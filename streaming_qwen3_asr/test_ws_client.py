#!/usr/bin/env python3

import asyncio
import json
import sys
import numpy as np
import scipy.io.wavfile
import websockets

SAMPLE_RATE = 16000
CHUNK_SIZE = int(0.1 * SAMPLE_RATE)  # 100ms chunks


async def test_ws():
    wav_path = "./test_wavs/far_3.wav"

    # Load audio
    rate, data = scipy.io.wavfile.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        wav = data.astype(np.float32) / 32768.0
    else:
        wav = data.astype(np.float32)
    wav = np.clip(wav, -1.0, 1.0)

    print(f"Audio: {wav.shape[0]} samples, {wav.shape[0]/16000:.2f}s")

    uri = "ws://localhost:8081/api/realtime/ws"
    try:
        async with websockets.connect(uri) as ws:
            # Receive init response
            msg = await ws.recv()
            j = json.loads(msg)
            print(f"Init: {j}")
            session_id = j.get("session_id")

            # Send audio in chunks
            pos = 0
            while pos < wav.shape[0]:
                chunk = wav[pos:pos + CHUNK_SIZE]
                # Convert to int16
                chunk_i16 = np.clip(chunk * 32768.0, -32768, 32767).astype(np.int16)
                await ws.send(chunk_i16.tobytes())
                pos += CHUNK_SIZE

                # Try to receive responses
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    j = json.loads(msg)
                    if j.get("type") == "TranscriptionResponse":
                        print(f"  [ Interim ] {j.get('text')}")
                except asyncio.TimeoutError:
                    pass

            # Send EOS
            await ws.send(json.dumps({"type": "EOS"}))

            # Receive final response
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    j = json.loads(msg)
                    if j.get("type") == "TranscriptionResponse":
                        print(f"  [ Final ] is_final={j.get('is_final')}, text={j.get('text')}")
                    elif j.get("type") == "VADEvent":
                        print(f"  [ VAD ] is_active={j.get('is_active')}")
            except asyncio.TimeoutError:
                pass

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(test_ws())
