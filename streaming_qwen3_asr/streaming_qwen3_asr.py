#!/usr/bin/env python3

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import scipy.io.wavfile
import scipy.signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoProcessor, AutoTokenizer

SAMPLE_RATE = 16000


def _make_sess(path: str, device: str = "cpu") -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = False

    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    return ort.InferenceSession(path, sess_options=so, providers=providers)


def _choose_existing_onnx_path(path: str) -> str:
    if os.path.exists(path):
        return path

    if path.endswith(".int8.onnx"):
        alt = path[:-10] + ".onnx"
        if os.path.exists(alt):
            return alt

    raise FileNotFoundError(f"ONNX file not found: {path}")


def _load_audio_any(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        wav = np.load(path)
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        return wav

    rate, data = scipy.io.wavfile.read(path)

    if data.ndim > 1:
        data = data.mean(axis=1)

    if data.dtype == np.int16:
        wav = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        wav = data.astype(np.float32) / 2147483648.0
    else:
        wav = data.astype(np.float32)

    if rate != SAMPLE_RATE:
        wav = scipy.signal.resample_poly(wav, SAMPLE_RATE, rate).astype(np.float32)

    wav = np.clip(wav, -1.0, 1.0)
    return wav


def _feat_to_audio_tokens_len_np(feat_len: np.ndarray, chunk_size: int = 100) -> np.ndarray:
    def _conv_out_len_3x_stride2(n: int) -> int:
        x = (int(n) + 1) // 2
        x = (x + 1) // 2
        return (x + 1) // 2

    def _aftercnn(x: np.ndarray) -> np.ndarray:
        x = (x - 1) // 2 + 1
        x = (x - 1) // 2 + 1
        return (x - 1) // 2 + 1

    cs = int(chunk_size)
    n = np.asarray(feat_len, dtype=np.int64)
    full = n // cs
    rem = n % cs
    tn = _conv_out_len_3x_stride2(cs)
    out = full * tn + _aftercnn(rem)
    return np.maximum(out, 0).astype(np.int64)


def _infer_cache_meta(dec_sess: ort.InferenceSession) -> Tuple[int, Optional[int], int, int]:
    inps = {i.name: i for i in dec_sess.get_inputs()}
    keys = sorted(
        [n for n in inps if n.startswith("cache_key_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if not keys:
        raise RuntimeError("decoder inputs missing cache_key_*")

    L = len(keys)
    s = inps[keys[0]].shape

    max_total_len = int(s[1]) if isinstance(s[1], int) else None
    kv = int(s[2]) if isinstance(s[2], int) else None
    hd = int(s[3]) if isinstance(s[3], int) else None

    if kv is None or hd is None:
        raise RuntimeError(f"Unable to infer decoder cache shape from input {keys[0]}: {s}")

    return L, max_total_len, kv, hd


def register_qwen3_asr() -> bool:
    try:
        from qwen3_asr import Qwen3ASRConfig, Qwen3ASRProcessor
        from transformers import AutoConfig, AutoProcessor
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
        return True
    except Exception as e:
        print(f"[warn] register_qwen3_asr failed: {e}")
        return False


def _load_tokenizer(model_dir: str):
    for kwargs in (
        {"trust_remote_code": True, "use_slow_tokenizer": True, "fix_mistral_regex": True},
        {"trust_remote_code": True, "fix_mistral_regex": True},
        {"trust_remote_code": True, "use_slow_tokenizer": True},
        {"trust_remote_code": True},
    ):
        try:
            return AutoTokenizer.from_pretrained(model_dir, **kwargs)
        except TypeError:
            continue
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def _load_processor(model_dir: str):
    for kwargs in ({"trust_remote_code": True, "fix_mistral_regex": True}, {"trust_remote_code": True}):
        try:
            return AutoProcessor.from_pretrained(model_dir, **kwargs)
        except TypeError:
            continue
    return AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)


def _trim_audio_features(audio_features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if audio_features.ndim != 3:
        return audio_features

    B, A, _ = audio_features.shape
    if B != 1:
        return audio_features

    energy = np.max(np.abs(audio_features[0]), axis=-1)
    idx = np.where(energy > eps)[0]
    if idx.size == 0:
        return audio_features

    A_valid = int(idx[-1] + 1)
    return audio_features[:, :A_valid, :]


def _parse_asr_output(raw_text: str, force_language: Optional[str] = None) -> Tuple[str, str]:
    if not raw_text:
        return force_language or "", ""

    lang = force_language or ""
    text = raw_text

    if "<asr_text>" in raw_text:
        idx = raw_text.find("<asr_text>")
        text = raw_text[idx + len("<asr_text>") :]
        text = (
            text.replace("<|im_end|>", "")
            .replace("<|endoftext|>", "")
            .replace("<|im_start|>", "")
            .strip()
        )

        before = raw_text[:idx]
        if "language" in before:
            lang_start = before.find("language")
            maybe_lang = before[lang_start + len("language") :].strip()
            if "<" in maybe_lang:
                maybe_lang = maybe_lang[: maybe_lang.find("<")].strip()
            if maybe_lang:
                lang = maybe_lang

    return lang, text


@dataclass
class StreamingState:
    chunk_size_sec: float
    chunk_size_samples: int
    buffer: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    language: str = ""
    text: str = ""


class StreamingQwen3ASR:
    def __init__(
        self,
        model_dir: str,
        conv_frontend_path: str,
        encoder_path: str,
        decoder_path: str,
        device: str = "cpu",
        max_total_len: int = 2048,
        max_new_tokens: int = 128,
    ):
        register_qwen3_asr()

        self.tokenizer = _load_tokenizer(model_dir)
        self.processor = _load_processor(model_dir)

        conv_path = _choose_existing_onnx_path(conv_frontend_path)
        enc_path = _choose_existing_onnx_path(encoder_path)
        dec_path = _choose_existing_onnx_path(decoder_path)

        self.conv_sess = _make_sess(conv_path, device=device)
        self.enc_sess = _make_sess(enc_path, device=device)
        self.dec_sess = _make_sess(dec_path, device=device)

        self.device = device
        self.max_new_tokens = int(max_new_tokens)

        self.L, inferred_max_total_len, self.kv_cache_k, self.kv_cache_v = _infer_cache_meta(
            self.dec_sess
        )
        self.max_total_len = inferred_max_total_len if inferred_max_total_len is not None else int(max_total_len)

        self.dec_out_names = [o.name for o in self.dec_sess.get_outputs()]
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

    def _build_prompt(self, audio_len: int, force_language: Optional[str] = None) -> str:
        system_text = "<|im_start|>system\n<|im_end|>\n"
        user_text = "<|im_start|>user\n"

        if force_language:
            user_text += f"language {force_language}<asr_text>\n"

        user_text += f"<|audio_start|>{'<|audio_pad|>' * audio_len}<|audio_end|><|im_end|>\n"
        assistant_text = "<|im_start|>assistant\n"

        return system_text + user_text + assistant_text

    def _run_conv_encoder(self, wav: np.ndarray) -> np.ndarray:
        audio_inputs = self.processor.feature_extractor(
            [wav],
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="np",
        )

        input_features = np.asarray(audio_inputs["input_features"], dtype=np.float32)
        feat_mask = np.asarray(audio_inputs["attention_mask"], dtype=np.int32)

        mel_input = input_features.transpose(0, 2, 1)
        conv_outputs = self.conv_sess.run(["conv_output"], {"input_features": mel_input})
        conv_output_np = np.asarray(conv_outputs[0], dtype=np.float32)

        valid = feat_mask != 0
        feat_len_np = valid.sum(axis=1).astype(np.int64)

        a_len = _feat_to_audio_tokens_len_np(feat_len_np, chunk_size=100)
        A = int(conv_output_np.shape[1])
        pos = np.arange(A, dtype=np.int64).reshape(1, A)
        tok_mask = pos < a_len.reshape(-1, 1)

        enc_inputs = {
            "input_features": conv_output_np,
            "feature_attention_mask": tok_mask.astype(np.bool_),
        }
        (audio_features,) = self.enc_sess.run(["audio_features"], enc_inputs)

        audio_features = np.asarray(audio_features, dtype=np.float32)
        audio_features = _trim_audio_features(audio_features)
        return audio_features

    def _alloc_caches(self, prompt_len: int) -> List[np.ndarray]:
        need_len = int(prompt_len) + int(self.max_new_tokens) + 8

        if self.max_total_len is not None and need_len > self.max_total_len:
            raise RuntimeError(
                f"Decoder cache too small: need={need_len}, max_total_len={self.max_total_len}"
            )

        total_len = self.max_total_len if self.max_total_len is not None else need_len

        caches: List[np.ndarray] = []
        for _ in range(self.L):
            caches.append(
                np.zeros((1, total_len, self.kv_cache_k, self.kv_cache_v), dtype=np.float32)
            )
            caches.append(
                np.zeros((1, total_len, self.kv_cache_k, self.kv_cache_v), dtype=np.float32)
            )
        return caches

    def _run_decoder_step(
        self,
        input_ids: np.ndarray,
        audio_features: np.ndarray,
        caches: List[np.ndarray],
        cur_len: int,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        B = 1
        S = int(input_ids.shape[1])

        cache_capacity = int(caches[0].shape[1])
        if cur_len + S > cache_capacity:
            raise RuntimeError(
                f"KV cache overflow: cur_len={cur_len}, step={S}, capacity={cache_capacity}"
            )

        attn_mask = np.ones((B, S), dtype=np.int64)
        cache_pos = np.arange(cur_len, cur_len + S, dtype=np.int64)

        feed: Dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "audio_features": audio_features,
            "attention_mask": attn_mask,
            "cache_position": cache_pos,
        }

        for i in range(self.L):
            feed[f"cache_key_{i}"] = caches[2 * i]
            feed[f"cache_value_{i}"] = caches[2 * i + 1]

        outs = self.dec_sess.run(self.dec_out_names, feed)
        out_map = {name: val for name, val in zip(self.dec_out_names, outs)}

        logits = np.asarray(out_map["logits"], dtype=np.float32)

        new_caches: List[np.ndarray] = []
        for i in range(self.L):
            kd = np.asarray(out_map[f"key_delta_{i}"], dtype=np.float32)
            vd = np.asarray(out_map[f"value_delta_{i}"], dtype=np.float32)

            caches[2 * i][:, cur_len : cur_len + S] = kd
            caches[2 * i + 1][:, cur_len : cur_len + S] = vd

            new_caches.append(caches[2 * i])
            new_caches.append(caches[2 * i + 1])

        return logits, new_caches

    def _sample_token(self, logits: np.ndarray, temperature: float = 0.0) -> int:
        last_logits = logits[0, -1, :]

        if temperature <= 0.0:
            return int(np.argmax(last_logits, axis=-1))

        x = last_logits / max(temperature, 1e-5)
        x = x - np.max(x)
        probs = np.exp(x)
        probs = probs / np.sum(probs)
        return int(np.random.choice(len(probs), p=probs))

    def _decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=True)

    def _generate_from_wav(
        self,
        wav: np.ndarray,
        language: Optional[str] = None,
        token_callback=None,
    ) -> Tuple[str, str]:
        audio_features = self._run_conv_encoder(wav)
        _, A, _ = audio_features.shape

        prompt = self._build_prompt(A, language)
        input_ids = np.asarray(
            [self.tokenizer.encode(prompt, add_special_tokens=False)],
            dtype=np.int64,
        )
        S0 = int(input_ids.shape[1])

        caches = self._alloc_caches(S0)

        cur_len = 0
        logits, caches = self._run_decoder_step(input_ids, audio_features, caches, cur_len)
        cur_len = S0

        gen_text = ""
        for _ in range(self.max_new_tokens - 1):
            token_id = self._sample_token(logits, temperature=0.0)

            if self.eos_token_id is not None and token_id == self.eos_token_id:
                break

            token_str = self._decode_token(token_id)
            gen_text += token_str

            if token_callback is not None:
                _, txt = _parse_asr_output(gen_text, language)
                token_callback(txt, False)

            input_ids = np.asarray([[token_id]], dtype=np.int64)
            logits, caches = self._run_decoder_step(input_ids, audio_features, caches, cur_len)
            cur_len += 1

        lang, txt = _parse_asr_output(gen_text, language)
        return lang, txt

    def init_streaming_state(self, chunk_size_sec: float = 2.0) -> StreamingState:
        chunk_size_samples = int(round(chunk_size_sec * SAMPLE_RATE))
        chunk_size_samples = max(1, chunk_size_samples)

        return StreamingState(
            chunk_size_sec=chunk_size_sec,
            chunk_size_samples=chunk_size_samples,
        )

    def streaming_transcribe(
        self,
        pcm16k: np.ndarray,
        state: StreamingState,
        language: Optional[str] = None,
        token_callback=None,
    ) -> StreamingState:
        x = np.asarray(pcm16k)
        if x.ndim != 1:
            x = x.reshape(-1)

        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32, copy=False)

        if x.shape[0] > 0:
            state.buffer = np.concatenate([state.buffer, x], axis=0)

        min_samples = int(0.5 * SAMPLE_RATE)
        if state.buffer.shape[0] < min_samples:
            return state

        lang, txt = self._generate_from_wav(
            state.buffer,
            language=language,
            token_callback=token_callback,
        )
        state.language = lang
        state.text = txt

        return state

    def finish_streaming_transcribe(
        self,
        state: StreamingState,
        language: Optional[str] = None,
    ) -> StreamingState:
        if state.buffer is None or state.buffer.shape[0] == 0:
            return state

        wav = state.buffer
        lang, txt = self._generate_from_wav(wav, language=language)

        state.language = lang
        state.text = txt
        state.buffer = np.zeros((0,), dtype=np.float32)

        return state