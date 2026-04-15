#!/usr/bin/env python3
#
# Qwen3-ASR ONNX inference. Supports FP32/INT8 with ConvFrontend in Python.

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import scipy.io.wavfile
import scipy.signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoProcessor, AutoTokenizer


def _feat_to_audio_tokens_len_np(
    feat_len: np.ndarray, chunk_size: int = 100
) -> np.ndarray:
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


def _register_qwen3_asr() -> bool:
    try:
        from qwen3_asr import Qwen3ASRConfig, Qwen3ASRProcessor
        from transformers import AutoConfig, AutoProcessor

        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
        return True
    except Exception as e:
        print(f"[warn] _register_qwen3_asr: {e}")
        return False


def _make_sess(path: str, device: str = "cpu") -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = False
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=so, providers=providers)


def _make_sess_with_fallback(
    path: str, device: str, fp32_fallback: Optional[str] = None
) -> ort.InferenceSession:
    try:
        return _make_sess(path, device=device)
    except Exception as e:
        msg = str(e)
        if fp32_fallback and ("ConvInteger" in msg or "NOT_IMPLEMENTED" in msg):
            print(
                f"[warn] load failed for {path}: {msg}\n       fallback -> {fp32_fallback}"
            )
            return _make_sess(fp32_fallback, device=device)
        raise


def _infer_cache_meta(
    dec_sess: ort.InferenceSession,
) -> Tuple[int, Optional[int], Optional[int], Optional[int]]:
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
    return L, max_total_len, kv, hd


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

    if rate != 16000:
        wav = scipy.signal.resample_poly(wav, 16000, rate).astype(np.float32)

    wav = np.clip(wav, -1.0, 1.0)
    return wav


def _check_model_dir(model_dir: str) -> None:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise ValueError(
            f"Model dir has no config.json: {os.path.abspath(model_dir)}\n"
            "Set --model to the Qwen3-ASR checkpoint path."
        )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"Failed to read {config_path}: {e}") from e
    if config.get("model_type") != "qwen3_asr":
        raise ValueError(
            f"config.json model_type is {config.get('model_type')!r}, expected 'qwen3_asr'.\n"
            f"Set --model to the Qwen3-ASR checkpoint path: {os.path.abspath(model_dir)}"
        )


def _load_tokenizer(model_dir: str):
    for kwargs in (
        dict(
            trust_remote_code=True,
            use_slow_tokenizer=True,
            fix_mistral_regex=True,
        ),
        dict(trust_remote_code=True, fix_mistral_regex=True),
        dict(trust_remote_code=True, use_slow_tokenizer=True),
        dict(trust_remote_code=True),
    ):
        try:
            return AutoTokenizer.from_pretrained(model_dir, **kwargs)
        except TypeError:
            continue
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def _load_processor(model_dir: str):
    for kwargs in (
        dict(trust_remote_code=True, fix_mistral_regex=True),
        dict(trust_remote_code=True),
    ):
        try:
            return AutoProcessor.from_pretrained(model_dir, **kwargs)
        except TypeError:
            continue
    return AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)


def _normalize_contexts(context_arg: Optional[List[str]], n: int) -> List[str]:
    if context_arg is None or len(context_arg) == 0:
        return [""] * n
    if len(context_arg) == 1 and n > 1:
        return [context_arg[0] or ""] * n
    if len(context_arg) != n:
        raise ValueError(
            f"expected 1 or {n} --context values, got {len(context_arg)}"
        )
    return [c or "" for c in context_arg]


def _build_messages(context: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]


def _build_text_prompt(
    proc, context: str, force_language: Optional[str]
) -> str:
    msgs = _build_messages(context)
    base = proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False
    )
    if force_language:
        base = base + f"language {force_language}<asr_text>"
    return base


def _resolve_audio_token_and_id(tok, proc) -> Tuple[str, int]:
    candidates: List[str] = []
    for holder in (getattr(proc, "tokenizer", None), tok):
        if holder is None:
            continue
        v = getattr(holder, "audio_token", None)
        if isinstance(v, str) and v:
            candidates.append(v)
    candidates.append("<|audio_pad|>")

    seen = set()
    unk_id = getattr(tok, "unk_token_id", None)

    for token in candidates:
        if token in seen:
            continue
        seen.add(token)

        tid = tok.convert_tokens_to_ids(token)
        if isinstance(tid, (int, np.integer)) and int(tid) >= 0:
            if unk_id is None or int(tid) != int(unk_id) or token == tok.unk_token:
                return token, int(tid)

        ids = tok.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            return token, int(ids[0])

    raise RuntimeError("Cannot resolve audio token id")


def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
        "--encoder",
        type=str,
        required=True,
        help="encoder.onnx or encoder.int8.onnx",
    )
    p.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="decoder.onnx or decoder.int8.onnx",
    )
    p.add_argument("--wav-dir", type=str, required=True, help="directory of .wav/.npy files")
    p.add_argument("--output-dir", type=str, default="npy_outputs", help="root directory to save per-node input .npy files")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--max-total-len", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--context",
        type=str,
        default="",
        help="context string, same semantics as official transcribe(context=...)",
    )
    return p.parse_args()


STATIC_BATCH = 1
STATIC_CONV_T = 3000
STATIC_CONV_F = 128
STATIC_AUDIO_TOKENS = 390
STATIC_DECODER_SEQ = 390
STATIC_HIDDEN = 1024


def _pad_or_truncate_last_dim(x: np.ndarray, target_len: int) -> np.ndarray:
    cur = int(x.shape[-1])
    if cur == target_len:
        return x
    if cur > target_len:
        return x[..., :target_len]
    pad_shape = list(x.shape)
    pad_shape[-1] = target_len - cur
    pad = np.zeros(pad_shape, dtype=x.dtype)
    return np.concatenate([x, pad], axis=-1)


def _fit_prompt_to_static_shape(
    input_ids: np.ndarray,
    prompt_attn: np.ndarray,
    seq_len: int,
    pad_id: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    S_raw = int(input_ids.shape[1])
    ids = np.full((1, seq_len), int(pad_id), dtype=np.int64)
    attn = np.zeros((1, seq_len), dtype=np.int64)

    keep = min(S_raw, seq_len)
    ids[:, :keep] = input_ids[:, :keep]
    attn[:, :keep] = prompt_attn[:, :keep]

    # Decoder wrapper does not consume attention_mask for masking, so we use
    # the true prompt length only for cache updates and picking logits index.
    effective_prompt_len = int(np.clip(attn.sum(), 1, seq_len))
    return ids, attn, effective_prompt_len


def _save_npy(
    output_dir: str, model_name: str, tensor_name: str, idx: int, arr: np.ndarray
) -> None:
    folder = os.path.join(output_dir, model_name, tensor_name)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, f"{idx:03d}.npy"), arr)


def main():
    args = get_args()

    _check_model_dir(args.model)
    _register_qwen3_asr()

    tok = _load_tokenizer(args.model)
    proc = _load_processor(args.model)
    _, audio_token_id = _resolve_audio_token_and_id(tok, proc)

    enc_fp32_guess = (
        args.encoder.replace(".int8.onnx", ".onnx")
        if args.encoder.endswith(".int8.onnx")
        else None
    )
    enc = _make_sess_with_fallback(
        args.encoder, device=args.device, fp32_fallback=enc_fp32_guess
    )

    dec_fp32_guess = (
        args.decoder.replace(".int8.onnx", ".onnx")
        if args.decoder.endswith(".int8.onnx")
        else None
    )
    dec = _make_sess_with_fallback(
        args.decoder, device=args.device, fp32_fallback=dec_fp32_guess
    )

    conv_sess = _make_sess(args.conv_frontend, device=args.device)

    exts = {".wav", ".npy"}
    wav_files = sorted(
        f for f in os.listdir(args.wav_dir)
        if os.path.splitext(f)[1].lower() in exts
    )
    if not wav_files:
        raise RuntimeError(f"no .wav/.npy files found in {args.wav_dir}")

    for idx, fname in enumerate(wav_files):
        wav_path = os.path.join(args.wav_dir, fname)
        _infer_one(
            args,
            tok,
            proc,
            enc,
            dec,
            conv_sess,
            audio_token_id,
            wav_path,
            _load_audio_any(wav_path),
            args.context or "",
            output_dir=args.output_dir,
            idx=idx,
        )


def _infer_one(
    args,
    tok,
    proc,
    enc,
    dec,
    conv_sess,
    audio_token_id,
    wav_path: str,
    wav: np.ndarray,
    context: str,
    output_dir: str = "",
    idx: int = 0,
) -> None:
    if wav.ndim != 1:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if int(wav.shape[0]) <= 0:
        raise RuntimeError("empty audio after load")

    total_audio_sec = max(len(wav) / 16000.0, 1e-6)

    texts = [_build_text_prompt(proc, context, None)]
    combined = proc(
        text=texts,
        audio=[wav],
        sampling_rate=16000,
        padding=True,
        truncation=True,
        return_tensors="np",
    )
    input_features = np.asarray(combined["input_features"], dtype=np.float32)
    feat_mask = np.asarray(combined["feature_attention_mask"], dtype=np.int32)
    input_ids = np.asarray(combined["input_ids"], dtype=np.int64)
    prompt_attn = np.asarray(combined["attention_mask"], dtype=np.int64)

    if input_features.shape[0] != STATIC_BATCH:
        raise RuntimeError(
            f"static model only supports batch=1, got {input_features.shape[0]}"
        )

    if input_features.shape[1] != STATIC_CONV_F:
        raise RuntimeError(
            f"unexpected mel bins: {input_features.shape[1]} != {STATIC_CONV_F}"
        )

    input_features = _pad_or_truncate_last_dim(input_features, STATIC_CONV_T)
    feat_mask = _pad_or_truncate_last_dim(feat_mask, STATIC_CONV_T)

    pad_id = (
        int(tok.pad_token_id)
        if tok.pad_token_id is not None
        else (int(tok.eos_token_id) if tok.eos_token_id is not None else 0)
    )
    input_ids, prompt_attn, S0 = _fit_prompt_to_static_shape(
        input_ids, prompt_attn, STATIC_DECODER_SEQ, pad_id
    )

    if args.debug:
        print(
            "batch:",
            1,
            "wav range:",
            float(wav.min()),
            float(wav.max()),
            "len:",
            wav.shape[0],
        )
        print("input_features:", input_features.shape, input_features.dtype)
        print(
            "feature_attention_mask:",
            feat_mask.shape,
            feat_mask.dtype,
            "sum:",
            int(feat_mask.sum()),
        )

    mel_input = input_features.transpose(0, 2, 1)
    if mel_input.shape != (STATIC_BATCH, STATIC_CONV_T, STATIC_CONV_F):
        raise RuntimeError(
            f"conv input shape mismatch: {mel_input.shape}, expected "
            f"({STATIC_BATCH}, {STATIC_CONV_T}, {STATIC_CONV_F})"
        )

    conv_inputs = {"input_features": mel_input}
    if output_dir:
        _save_npy(output_dir, "conv_frontend", "input_features", idx, mel_input)

    (conv_output_np,) = conv_sess.run(["conv_output"], conv_inputs)

    valid = feat_mask != 0
    feat_len_np = valid.sum(axis=1).astype(np.int64)
    a_len = _feat_to_audio_tokens_len_np(
        feat_len_np, chunk_size=args.chunk_size
    )

    if conv_output_np.shape[0] != STATIC_BATCH:
        raise RuntimeError(
            f"conv output batch mismatch: {conv_output_np.shape[0]} != {STATIC_BATCH}"
        )

    A_conv = int(conv_output_np.shape[1])
    if A_conv != STATIC_AUDIO_TOKENS:
        raise RuntimeError(
            f"conv output token mismatch: {A_conv} != {STATIC_AUDIO_TOKENS}"
        )

    pos = np.arange(A_conv, dtype=np.int64).reshape(1, A_conv)
    tok_mask = pos < np.minimum(a_len, STATIC_AUDIO_TOKENS).reshape(-1, 1)

    if args.debug:
        print("conv_output:", conv_output_np.shape, conv_output_np.dtype)
        print(
            "tok_mask:",
            tok_mask.shape,
            tok_mask.dtype,
            "sum:",
            int(tok_mask.sum()),
        )

    enc_inputs = {
        "input_features": conv_output_np,
        "feature_attention_mask": tok_mask.astype(np.bool_),
    }
    if output_dir:
        for _name, _arr in enc_inputs.items():
            _save_npy(output_dir, "encoder", _name, idx, _arr)
    (audio_features,) = enc.run(["audio_features"], enc_inputs)
    audio_features = np.asarray(audio_features, dtype=np.float32)

    if audio_features.shape != (STATIC_BATCH, STATIC_AUDIO_TOKENS, STATIC_HIDDEN):
        raise RuntimeError(
            "unexpected static encoder output shape: "
            f"{audio_features.shape}, expected "
            f"({STATIC_BATCH}, {STATIC_AUDIO_TOKENS}, {STATIC_HIDDEN})"
        )

    B = STATIC_BATCH
    A = STATIC_AUDIO_TOKENS

    if args.debug:
        print("audio_features:", audio_features.shape, audio_features.dtype)

    slots = (input_ids == audio_token_id).sum(axis=1)
    if int(slots[0]) <= 0:
        raise RuntimeError(
            "decoder input_ids has no audio slots; please check model/tokenizer compatibility"
        )
    if args.debug:
        print(
            "audio slots in prompt:",
            int(slots[0]),
            "audio_features_len:",
            STATIC_AUDIO_TOKENS,
        )

    L, model_max_total_len, kv, hd = _infer_cache_meta(dec)
    if kv is None or hd is None:
        raise RuntimeError("decoder cache shape has dynamic kv/hd")

    runtime_max_total_len = (
        int(model_max_total_len)
        if model_max_total_len is not None
        else int(args.max_total_len)
    )

    required_total_len = S0 + int(args.max_new_tokens)
    if required_total_len > runtime_max_total_len:
        raise RuntimeError(
            f"max_total_len not enough: S0({S0}) + max_new_tokens({args.max_new_tokens}) = "
            f"{required_total_len}, but max_total_len={runtime_max_total_len}"
        )

    if args.debug:
        print("S0:", S0)
        print("A:", A)
        print("required_total_len:", required_total_len)
        print("runtime_max_total_len:", runtime_max_total_len)
        print("num_layers:", L, "kv:", kv, "hd:", hd)
        print("decoder static seq:", STATIC_DECODER_SEQ)

    caches: List[np.ndarray] = []
    for _ in range(L):
        caches.append(
            np.zeros((B, runtime_max_total_len, kv, hd), dtype=np.float32)
        )
        caches.append(
            np.zeros((B, runtime_max_total_len, kv, hd), dtype=np.float32)
        )

    dec_out_names = [o.name for o in dec.get_outputs()]
    if "logits" not in dec_out_names:
        raise RuntimeError(f"decoder outputs missing logits")

    def _run_decoder(
        step_input_ids: np.ndarray,
        cur_len: int,
        attn_override: np.ndarray,
        num_new_tokens: int,
        save_inputs: bool = False,
    ) -> np.ndarray:
        Sb = int(step_input_ids.shape[0])
        S = int(step_input_ids.shape[1])
        if Sb != B:
            raise RuntimeError(f"step_input_ids batch {Sb} != B {B}")
        if S != STATIC_DECODER_SEQ:
            raise RuntimeError(
                f"static decoder requires S={STATIC_DECODER_SEQ}, got {S}"
            )
        if attn_override.shape != (B, STATIC_DECODER_SEQ):
            raise RuntimeError(
                "attention_mask shape mismatch: "
                f"{attn_override.shape} != ({B}, {STATIC_DECODER_SEQ})"
            )

        if cur_len + int(num_new_tokens) > runtime_max_total_len:
            raise RuntimeError(
                "cur_len overflow: "
                f"{cur_len}+{num_new_tokens} > {runtime_max_total_len}"
            )

        cache_pos = np.arange(
            cur_len, cur_len + STATIC_DECODER_SEQ, dtype=np.int64
        )
        feed: Dict[str, np.ndarray] = {
            "input_ids": step_input_ids,
            "audio_features": audio_features,
            "attention_mask": attn_override.astype(np.int64, copy=False),
            "cache_position": cache_pos,
        }
        for i in range(L):
            feed[f"cache_key_{i}"] = caches[2 * i]
            feed[f"cache_value_{i}"] = caches[2 * i + 1]

        if save_inputs and output_dir:
            for _name, _arr in feed.items():
                _save_npy(output_dir, "decoder", _name, idx, _arr)

        outs = dec.run(dec_out_names, feed)
        out_map = {name: val for name, val in zip(dec_out_names, outs)}
        logits = np.asarray(out_map["logits"], dtype=np.float32)

        u = int(num_new_tokens)
        if u > 0:
            for i in range(L):
                kd = np.asarray(out_map[f"key_delta_{i}"], dtype=np.float32)
                vd = np.asarray(out_map[f"value_delta_{i}"], dtype=np.float32)
                caches[2 * i][:, cur_len : cur_len + u] = kd[:, :u]
                caches[2 * i + 1][:, cur_len : cur_len + u] = vd[:, :u]
        return logits

    cur_len = 0
    logits = _run_decoder(input_ids, cur_len, prompt_attn, num_new_tokens=S0, save_inputs=bool(output_dir))
    cur_len += S0

    eos_id = tok.eos_token_id
    out_ids: List[int] = []

    infer_start_time = time.time()

    next_id = int(np.argmax(logits[0, S0 - 1, :], axis=-1))
    out_ids.append(next_id)
    active = not (eos_id is not None and next_id == int(eos_id))

    step_attn = np.ones((1, STATIC_DECODER_SEQ), dtype=np.int64)

    for _ in range(int(args.max_new_tokens) - 1):
        if not active:
            break
        step_ids = np.full((1, STATIC_DECODER_SEQ), pad_id, dtype=np.int64)
        step_ids[0, 0] = out_ids[-1]
        logits = _run_decoder(step_ids, cur_len, step_attn, num_new_tokens=1)
        cur_len += 1
        next_id = int(np.argmax(logits[0, 0, :], axis=-1))
        out_ids.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            active = False

    text = tok.decode(out_ids, skip_special_tokens=True)
    text = text.replace("\ufffd", "")
    print(f"[{wav_path}] {text}")

    processing_time = time.time() - infer_start_time
    rtf = processing_time / total_audio_sec
    print(f"RTF: {rtf:.4f}")


if __name__ == "__main__":
    main()
