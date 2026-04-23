#!/usr/bin/env python3
"""
dump_asr_debug_npy.py
---------------------
运行 Qwen3-ASR 的音频前端和编码器，生成用于 C++ 调试的中间数据文件。

输出文件（存放在 --output_dir）：
  input_ids.npy            int64 [1, S]   完整 token ID 序列（含音频占位符 151676）
  audio_features.npy       float32 [1, A, 1024]  encoder 输出
  attention_mask.npy       int64 [1, S]   prefill 阶段注意力掩码
  combined_embed_f32.npy   float32 [S, 1024]     拼接后的 embedding（验证用）
  combined_embed_bf16.bin  raw bytes      bfloat16 packed，C++ 直接 fread
  meta.json                JSON           形状和配置元信息

使用示例：
  python dump_asr_debug_npy.py \\
    --model  Qwen3-ASR-0.6B \\
    --conv_frontend Qwen3-ASR-0.6B_Static_ONNX/conv_frontend.onnx \\
    --encoder       Qwen3-ASR-0.6B_Static_ONNX/encoder.onnx \\
    --wav    test_wavs/asr_example_zh.wav \\
    --output_dir debug_npy
"""

import argparse
import json
import os
import struct
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import scipy.io.wavfile
import scipy.signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import axengine as axe
except ImportError:
    axe = None

from transformers import AutoProcessor, AutoTokenizer


# ──────────────────────────────────────────────
# 复用 infer_qwen3_asr_static.py 的工具函数
# ──────────────────────────────────────────────

STATIC_BATCH = 1
STATIC_CONV_T = 3000
STATIC_CONV_F = 128
STATIC_AUDIO_TOKENS = 390
STATIC_DECODER_SEQ = 390
STATIC_HIDDEN = 1024
AUDIO_TOKEN_ID = 151676


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


def _register_qwen3_asr() -> bool:
    try:
        from qwen3_asr import Qwen3ASRConfig, Qwen3ASRProcessor
        from transformers import AutoConfig
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
        return True
    except Exception as e:
        print(f"[warn] _register_qwen3_asr: {e}")
        return False


def _load_tokenizer(model_dir: str):
    for kwargs in (
        dict(trust_remote_code=True, use_slow_tokenizer=True, fix_mistral_regex=True),
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


def _load_audio_any(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32).reshape(-1)
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
    return np.clip(wav, -1.0, 1.0)


def _pad_or_truncate_last_dim(x: np.ndarray, target_len: int) -> np.ndarray:
    cur = int(x.shape[-1])
    if cur == target_len:
        return x
    if cur > target_len:
        return x[..., :target_len]
    pad_shape = list(x.shape)
    pad_shape[-1] = target_len - cur
    return np.concatenate([x, np.zeros(pad_shape, dtype=x.dtype)], axis=-1)


def _fit_prompt_to_static_shape(
    input_ids: np.ndarray, prompt_attn: np.ndarray, seq_len: int, pad_id: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    S_raw = int(input_ids.shape[1])
    ids = np.full((1, seq_len), int(pad_id), dtype=np.int64)
    attn = np.zeros((1, seq_len), dtype=np.int64)
    keep = min(S_raw, seq_len)
    ids[:, :keep] = input_ids[:, :keep]
    attn[:, :keep] = prompt_attn[:, :keep]
    effective_prompt_len = int(np.clip(attn.sum(), 1, seq_len))
    return ids, attn, effective_prompt_len


def _make_sess(path: str, device: str = "cpu") -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = False
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=so, providers=providers)


def _make_axe_sess(path: str):
    if axe is None:
        raise RuntimeError("axengine not installed; cannot load .axmodel file")
    return axe.InferenceSession(path)


# ──────────────────────────────────────────────
# float32 → bfloat16 转换
# ──────────────────────────────────────────────

def float32_to_bfloat16_u16(x: np.ndarray) -> np.ndarray:
    """将 float32 ndarray 转换为 bfloat16 的 uint16 表示（截断最低 16 位）。"""
    x = np.asarray(x, dtype=np.float32)
    u32 = x.view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="生成 Qwen3-ASR 调试用 npy/bin 文件",
    )
    p.add_argument("--model", required=True, help="Qwen3-ASR-0.6B checkpoint 目录")
    p.add_argument("--conv_frontend", required=True, help="conv_frontend.onnx 或 .axmodel 路径")
    p.add_argument("--encoder", required=True, help="encoder.onnx 或 .axmodel 路径")
    p.add_argument("--wav", required=True, help=".wav 或 .npy 音频路径")
    p.add_argument("--output_dir", default="debug_npy", help="输出目录")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--chunk_size", type=int, default=100)
    p.add_argument("--context", type=str, default="", help="ASR 上下文提示")
    return p.parse_args()


def _build_text_prompt(proc, context: str) -> str:
    msgs = [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    _register_qwen3_asr()
    tok = _load_tokenizer(args.model)
    proc = _load_processor(args.model)
    _, audio_token_id = _resolve_audio_token_and_id(tok, proc)
    print(f"audio_token_id = {audio_token_id}")

    # 加载会话
    if args.conv_frontend.endswith(".axmodel"):
        conv_sess = _make_axe_sess(args.conv_frontend)
    else:
        conv_sess = _make_sess(args.conv_frontend, device=args.device)

    if args.encoder.endswith(".axmodel"):
        enc_sess = _make_axe_sess(args.encoder)
    else:
        enc_sess = _make_sess(args.encoder, device=args.device)

    # 加载音频
    wav = _load_audio_any(args.wav)
    print(f"wav shape: {wav.shape}, dur: {wav.shape[0]/16000:.2f}s")

    # ── 1. 特征提取 ──
    text_prompt = _build_text_prompt(proc, args.context)
    combined = proc(
        text=[text_prompt],
        audio=[wav],
        sampling_rate=16000,
        padding=True,
        truncation=True,
        return_tensors="np",
    )
    input_features = np.asarray(combined["input_features"], dtype=np.float32)
    feat_mask = np.asarray(combined["feature_attention_mask"], dtype=np.int32)
    input_ids_raw = np.asarray(combined["input_ids"], dtype=np.int64)
    prompt_attn_raw = np.asarray(combined["attention_mask"], dtype=np.int64)

    # 调整到静态形状
    input_features = _pad_or_truncate_last_dim(input_features, STATIC_CONV_T)
    feat_mask = _pad_or_truncate_last_dim(feat_mask, STATIC_CONV_T)

    pad_id = int(tok.pad_token_id) if tok.pad_token_id is not None else (
        int(tok.eos_token_id) if tok.eos_token_id is not None else 0
    )
    input_ids, attention_mask, S0 = _fit_prompt_to_static_shape(
        input_ids_raw, prompt_attn_raw, STATIC_DECODER_SEQ, pad_id
    )
    print(f"input_ids shape: {input_ids.shape}, effective_prompt_len (S0): {S0}")
    audio_slots = int((input_ids == audio_token_id).sum())
    print(f"audio slot count in prompt: {audio_slots}")

    # ── 2. conv_frontend 推理 ──
    mel_input = input_features.transpose(0, 2, 1)  # [1, 3000, 128]
    conv_out_names = [o.name for o in conv_sess.get_outputs()]
    conv_vals = conv_sess.run(conv_out_names, {"input_features": mel_input})
    conv_out_map = {n: np.asarray(v) for n, v in zip(conv_out_names, conv_vals)}
    conv_output_np = conv_out_map.get("conv_output", np.asarray(conv_vals[0]))
    print(f"conv_output shape: {conv_output_np.shape}")  # [1, 390, 896]

    # ── 3. encoder 推理 ──
    valid = feat_mask != 0
    feat_len_np = valid.sum(axis=1).astype(np.int64)
    a_len = _feat_to_audio_tokens_len_np(feat_len_np, chunk_size=args.chunk_size)
    A_conv = int(conv_output_np.shape[1])
    pos = np.arange(A_conv, dtype=np.int64).reshape(1, A_conv)
    tok_mask = pos < np.minimum(a_len, STATIC_AUDIO_TOKENS).reshape(-1, 1)

    _mask_dtype = np.uint8 if args.encoder.endswith(".axmodel") else np.bool_
    enc_inputs = {
        "input_features": conv_output_np,
        "feature_attention_mask": tok_mask.astype(_mask_dtype),
    }
    (audio_features,) = enc_sess.run(["audio_features"], enc_inputs)
    audio_features = np.asarray(audio_features, dtype=np.float32)  # [1, 390, 1024]
    print(f"audio_features shape: {audio_features.shape}")

    # ── 4. 构建 combined embedding ──
    #   对 input_ids 中每个位置，根据是否为 audio_token_id 选取 embedding 来源
    #   文字 token 的 embedding 从 embed_tokens 权重文件读取
    embed_weight_path = os.path.join(args.model, "thinker.model.embed_tokens.weight")
    if not os.path.isfile(embed_weight_path):
        # 尝试从 ONNX 输出目录找
        onnx_dir = os.path.dirname(args.encoder)
        embed_weight_path = os.path.join(onnx_dir, "thinker.model.embed_tokens.weight")

    S = int(input_ids.shape[1])
    H = STATIC_HIDDEN  # 1024
    A = int(audio_features.shape[1])  # 390

    combined_embed_f32 = np.zeros((S, H), dtype=np.float32)

    if os.path.isfile(embed_weight_path):
        # 读取 bfloat16 权重并转 float32
        raw = np.fromfile(embed_weight_path, dtype=np.uint16)
        vocab_size = len(raw) // H
        print(f"embed_tokens: vocab_size={vocab_size}, hidden_size={H}")
        # bfloat16 → float32
        embed_u32 = raw.astype(np.uint32) << 16
        embed_f32_table = embed_u32.view(np.float32).reshape(vocab_size, H)
    else:
        # 从 safetensors 或 pytorch 模型加载（慢，仅备用）
        print(f"[warn] embed_weight not found at {embed_weight_path}, loading from model (slow)...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype="auto"
        )
        embed_f32_table = model.thinker.model.embed_tokens.weight.detach().float().numpy()
        vocab_size = embed_f32_table.shape[0]

    audio_rank = 0
    ids_flat = input_ids[0]  # [S]
    for i in range(S):
        tid = int(ids_flat[i])
        if tid == audio_token_id:
            if audio_rank < A:
                combined_embed_f32[i] = audio_features[0, audio_rank, :]
                audio_rank += 1
            else:
                combined_embed_f32[i] = 0.0
        else:
            if 0 <= tid < vocab_size:
                combined_embed_f32[i] = embed_f32_table[tid]
            # else leave zeros for out-of-range (pad)

    print(f"combined_embed_f32 shape: {combined_embed_f32.shape}, audio slots filled: {audio_rank}")

    # bfloat16 版本（C++ 直接使用）
    combined_embed_bf16 = float32_to_bfloat16_u16(combined_embed_f32)

    # ── 5. 保存所有文件 ──
    np.save(os.path.join(args.output_dir, "input_ids.npy"), input_ids)
    np.save(os.path.join(args.output_dir, "audio_features.npy"), audio_features)
    np.save(os.path.join(args.output_dir, "attention_mask.npy"), attention_mask)
    np.save(os.path.join(args.output_dir, "combined_embed_f32.npy"), combined_embed_f32)

    combined_embed_bf16.tofile(os.path.join(args.output_dir, "combined_embed_bf16.bin"))

    meta = {
        "S": S,
        "S0": S0,
        "A": A,
        "hidden_size": H,
        "audio_token_id": audio_token_id,
        "pad_id": pad_id,
        "audio_slots_filled": audio_rank,
        "wav_path": args.wav,
        "context": args.context,
        "vocab_size": int(vocab_size),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[done] 输出文件保存到: {os.path.abspath(args.output_dir)}")
    for fname in ["input_ids.npy", "audio_features.npy", "attention_mask.npy",
                  "combined_embed_f32.npy", "combined_embed_bf16.bin", "meta.json"]:
        fpath = os.path.join(args.output_dir, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname:35s}  {size:>12,} bytes")

    print(f"\nmeta: {json.dumps(meta, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
