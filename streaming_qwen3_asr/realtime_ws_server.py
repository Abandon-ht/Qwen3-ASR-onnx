#!/usr/bin/env python3

import argparse
import json
import os
import sys
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streaming_qwen3_asr import StreamingQwen3ASR, StreamingState

SAMPLE_RATE = 16000


def pick_providers(device: str) -> List[str]:
    providers = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" in providers:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class VADConfig:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        threshold: float = 0.3,
        window_size: int = 512,
        hop_size: int = 160,
        prepad_ms: int = 200,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.5,
    ):
        self.model_path = model_path
        self.device = device
        self.threshold = threshold
        self.window_size = window_size
        self.hop_size = hop_size
        self.prepad_ms = prepad_ms
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.sample_rate = SAMPLE_RATE


class SileroVADCore:
    def __init__(self, cfg: VADConfig):
        self.cfg = cfg

        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1

        self.sess = ort.InferenceSession(
            cfg.model_path,
            sess_options=so,
            providers=pick_providers(cfg.device),
        )

        ins = self.sess.get_inputs()
        outs = self.sess.get_outputs()

        self.in_names = [i.name for i in ins]
        self.out_names = [o.name for o in outs]

        self.in_x = self.in_names[0]
        self.in_sr = None
        self.in_h = None
        self.in_c = None

        for n in self.in_names:
            nl = n.lower()
            if "sr" in nl:
                self.in_sr = n
            elif nl == "h":
                self.in_h = n
            elif nl == "c":
                self.in_c = n

        self.out_p = self.out_names[0]
        self.out_hn = None
        self.out_cn = None

        for n in self.out_names:
            nl = n.lower()
            if nl in ("hn", "h_n", "h_out", "h"):
                self.out_hn = n
            elif nl in ("cn", "c_n", "c_out", "c"):
                self.out_cn = n

    def init_state(self):
        if self.in_h is None or self.in_c is None:
            return None, None
        h = np.zeros((2, 1, 64), dtype=np.float32)
        c = np.zeros((2, 1, 64), dtype=np.float32)
        return h, c

    def infer(self, frame: np.ndarray, h, c):
        x = frame.astype(np.float32).reshape(1, -1)
        feed = {self.in_x: x}

        if self.in_sr is not None:
            feed[self.in_sr] = np.array([self.cfg.sample_rate], dtype=np.int64)

        if (
            self.in_h is not None
            and self.in_c is not None
            and h is not None
            and c is not None
        ):
            feed[self.in_h] = h
            feed[self.in_c] = c

        outs = self.sess.run(None, feed)

        p = float(np.array(outs[0]).reshape(-1)[0])

        hn = None
        cn = None
        if self.out_hn is not None and self.out_cn is not None:
            idx_hn = self.out_names.index(self.out_hn)
            idx_cn = self.out_names.index(self.out_cn)
            hn = outs[idx_hn]
            cn = outs[idx_cn]
        elif len(outs) >= 3 and self.in_h is not None and self.in_c is not None:
            hn = outs[1]
            cn = outs[2]

        return p, hn, cn


class VADStream:
    def __init__(self, core: SileroVADCore, cfg: VADConfig):
        if core is None:
            raise ValueError("VADStream requires a valid SileroVADCore")
        self.core = core
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.sample_index = 0
        self.in_speech = False
        self._buf = np.zeros((0,), dtype=np.float32)
        self._prepad = np.zeros((0,), dtype=np.float32)
        self._speech_run = 0
        self._silence_run = 0
        self._h, self._c = self.core.init_state()
        self._speech_start_sample = 0

    def feed(
        self, chunk_f32: np.ndarray
    ) -> List[Tuple[dict, np.ndarray, bool, int, int]]:
        cfg = self.cfg
        chunk_f32 = np.asarray(chunk_f32, dtype=np.float32).reshape(-1)
        self._buf = np.concatenate([self._buf, chunk_f32], axis=0)

        prepad_keep = int(cfg.sample_rate * cfg.prepad_ms / 1000)
        min_speech_frames = max(
            1, int(cfg.min_speech_duration * cfg.sample_rate / cfg.hop_size)
        )
        min_silence_frames = max(
            1, int(cfg.min_silence_duration * cfg.sample_rate / cfg.hop_size)
        )

        events: List[Tuple[dict, np.ndarray, bool, int, int]] = []
        produced = np.zeros((0,), dtype=np.float32)

        while self._buf.shape[0] >= cfg.window_size:
            frame = self._buf[: cfg.window_size]
            p, self._h, self._c = self.core.infer(frame, self._h, self._c)
            is_speech = p >= cfg.threshold

            cur_hop = self._buf[: cfg.hop_size]

            if not self.in_speech:
                self._prepad = np.concatenate([self._prepad, cur_hop], axis=0)
                if self._prepad.shape[0] > prepad_keep:
                    self._prepad = self._prepad[-prepad_keep:]
            else:
                produced = np.concatenate([produced, cur_hop], axis=0)

            if is_speech:
                self._speech_run += 1
                self._silence_run = 0
            else:
                self._silence_run += 1
                if not self.in_speech:
                    self._speech_run = 0

            if (
                (not self.in_speech)
                and is_speech
                and self._speech_run >= min_speech_frames
            ):
                self.in_speech = True
                self._speech_start_sample = self.sample_index

                start_audio = np.concatenate([self._prepad, cur_hop], axis=0)
                self._prepad = np.zeros((0,), dtype=np.float32)

                s0 = max(0, self._speech_start_sample - len(start_audio))
                s1 = self.sample_index + cfg.hop_size
                events.append(({"start": s0}, start_audio, False, s0, s1))

            if (
                self.in_speech
                and (not is_speech)
                and self._silence_run >= min_silence_frames
            ):
                self.in_speech = False

                end_audio = np.concatenate([produced, cur_hop], axis=0)
                produced = np.zeros((0,), dtype=np.float32)

                s0 = self._speech_start_sample
                s1 = self.sample_index + cfg.hop_size
                events.append(({"end": s1}, end_audio, True, s0, s1))

                self._speech_run = 0
                self._silence_run = 0

            self._buf = self._buf[cfg.hop_size :]
            self.sample_index += cfg.hop_size

        if self.in_speech and produced.shape[0] > 0:
            s0 = self._speech_start_sample
            s1 = self.sample_index
            events.append(({}, produced, False, s0, s1))

        return events

    def force_end(self):
        if not self.in_speech:
            return None

        tail = (
            self._buf.copy()
            if self._buf.size > 0
            else np.zeros((0,), dtype=np.float32)
        )
        s1 = self.sample_index + tail.shape[0]

        self.in_speech = False
        self._buf = np.zeros((0,), dtype=np.float32)
        self._speech_run = 0
        self._silence_run = 0
        self._h, self._c = self.core.init_state()

        return {"end": s1}, tail, True, self._speech_start_sample, s1


class TranscriptionResponse(BaseModel):
    type: str = "TranscriptionResponse"
    text: str = ""
    is_final: bool = False
    segment_id: int = 0


class VADEvent(BaseModel):
    type: str = "VADEvent"
    is_active: bool


app = FastAPI()

sessions: Dict[str, StreamingState] = {}
vad_sessions: Dict[str, VADStream] = {}

asr_model: Optional[StreamingQwen3ASR] = None
vad_model: Optional[SileroVADCore] = None
vad_model_path: str = ""
args = None
chunk_size_sec = 2.0


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", type=str, required=True, help="Qwen3-ASR checkpoint path"
    )
    p.add_argument(
        "--conv_frontend",
        type=str,
        required=True,
        help="conv_frontend.onnx path",
    )
    p.add_argument(
        "--encoder", type=str, required=True, help="encoder.onnx path"
    )
    p.add_argument(
        "--decoder", type=str, required=True, help="decoder.onnx path"
    )
    p.add_argument(
        "--vad_model", type=str, default="", help="silero_vad.onnx path"
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--chunk_size_sec", type=float, default=0.5)
    return p.parse_args()


@app.get("/")
async def index():
    html_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "index.html"
    )
    return FileResponse(html_path, media_type="text/html")


@app.websocket("/api/realtime/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = uuid.uuid4().hex

    state = asr_model.init_streaming_state(chunk_size_sec=chunk_size_sec)
    sessions[session_id] = state

    use_vad = vad_model is not None and bool(vad_model_path)
    vad_stream = None

    if use_vad:
        vad_cfg = VADConfig(
            model_path=vad_model_path,
            device=args.device,
            threshold=0.3,
            window_size=512,
            hop_size=160,
            prepad_ms=200,
            min_speech_duration=0.25,
            min_silence_duration=0.5,
        )
        vad_stream = VADStream(vad_model, vad_cfg)
        vad_sessions[session_id] = vad_stream

    sentence_count = 0
    audio_buf = np.zeros((0,), dtype=np.float32)
    chunk_size = int(0.1 * SAMPLE_RATE)
    in_speech = False
    segment_open = False
    last_sent_text = ""

    async def handle_vad_events(events):
        nonlocal state, sentence_count, in_speech, last_sent_text

        for speech_dict, speech_f32, is_last, s0, s1 in events:
            if "start" in speech_dict:
                sentence_count += 1
                state = asr_model.init_streaming_state(
                    chunk_size_sec=chunk_size_sec
                )
                sessions[session_id] = state
                in_speech = True
                last_sent_text = ""

                await websocket.send_json(
                    {
                        "type": "NewSegment",
                        "segment_id": sentence_count,
                    }
                )
                await websocket.send_json(VADEvent(is_active=True).model_dump())

            if in_speech or speech_f32.size > 0:
                if speech_f32.size > 0:
                    speech_i16 = np.clip(
                        speech_f32 * 32768.0, -32768, 32767
                    ).astype(np.int16)
                    state = asr_model.streaming_transcribe(speech_i16, state)

                    if state.text and state.text != last_sent_text:
                        await websocket.send_json(
                            TranscriptionResponse(
                                text=state.text,
                                is_final=False,
                                segment_id=sentence_count,
                            ).model_dump()
                        )
                        last_sent_text = state.text

                if is_last:
                    state = asr_model.finish_streaming_transcribe(state)

                    if state.text:
                        await websocket.send_json(
                            TranscriptionResponse(
                                text=state.text,
                                is_final=True,
                                segment_id=sentence_count,
                            ).model_dump()
                        )

                    state = asr_model.init_streaming_state(
                        chunk_size_sec=chunk_size_sec
                    )
                    sessions[session_id] = state
                    in_speech = False
                    last_sent_text = ""
                    await websocket.send_json(
                        VADEvent(is_active=False).model_dump()
                    )

    try:
        await websocket.send_json(
            {
                "type": "InitResponse",
                "session_id": session_id,
            }
        )

        while True:
            msg = await websocket.receive()

            if msg["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()

            if msg["type"] == "websocket.receive" and "text" in msg:
                try:
                    j = json.loads(msg["text"])
                except Exception:
                    continue

                if isinstance(j, dict) and j.get("type") == "EOS":
                    if use_vad:
                        if audio_buf.shape[0] > 0:
                            events = vad_stream.feed(audio_buf)
                            audio_buf = np.zeros((0,), dtype=np.float32)
                            await handle_vad_events(events)

                        forced = vad_stream.force_end()
                        if forced is not None:
                            await handle_vad_events([forced])
                    else:
                        state = asr_model.finish_streaming_transcribe(state)
                        if state.text:
                            await websocket.send_json(
                                TranscriptionResponse(
                                    text=state.text,
                                    is_final=True,
                                    segment_id=sentence_count
                                    if sentence_count > 0
                                    else 1,
                                ).model_dump()
                            )
                    break

                continue

            if msg["type"] != "websocket.receive" or "bytes" not in msg:
                continue

            data = msg["bytes"]
            if not data:
                continue

            samples_i16 = np.frombuffer(data, dtype=np.int16)
            if samples_i16.size == 0:
                continue

            if not use_vad:
                if not segment_open:
                    sentence_count = 1
                    segment_open = True
                    await websocket.send_json(
                        {
                            "type": "NewSegment",
                            "segment_id": sentence_count,
                        }
                    )

                state = asr_model.streaming_transcribe(samples_i16, state)
                if state.text and state.text != last_sent_text:
                    await websocket.send_json(
                        TranscriptionResponse(
                            text=state.text,
                            is_final=False,
                            segment_id=sentence_count,
                        ).model_dump()
                    )
                    last_sent_text = state.text
                continue

            samples_f32 = samples_i16.astype(np.float32) / 32768.0
            audio_buf = np.concatenate([audio_buf, samples_f32], axis=0)

            while audio_buf.shape[0] >= chunk_size:
                chunk = audio_buf[:chunk_size]
                audio_buf = audio_buf[chunk_size:]

                events = vad_stream.feed(chunk)
                await handle_vad_events(events)

    except WebSocketDisconnect:
        pass
    finally:
        sessions.pop(session_id, None)
        vad_sessions.pop(session_id, None)


def main():
    global asr_model, vad_model, vad_model_path, args, chunk_size_sec

    args = get_args()

    asr_model = StreamingQwen3ASR(
        model_dir=args.model,
        conv_frontend_path=args.conv_frontend,
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        device=args.device,
    )

    vad_model_path = args.vad_model
    if vad_model_path and os.path.exists(vad_model_path):
        vad_cfg = VADConfig(model_path=vad_model_path, device=args.device)
        vad_model = SileroVADCore(vad_cfg)
    else:
        if vad_model_path:
            print(
                f"[warn] VAD model not found, fallback to no-VAD mode: {vad_model_path}"
            )
        vad_model_path = ""
        vad_model = None

    chunk_size_sec = args.chunk_size_sec

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
