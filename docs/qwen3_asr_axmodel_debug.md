# Qwen3-ASR AX650 Decoder 调试技术文档

## 1. 整体推理流程

```
WAV 音频
  │
  ▼
AutoProcessor → mel features [1, 128, 3000]
  │
  ▼
conv_frontend（.axmodel 或 .onnx）
  Input:  input_features [1, 3000, 128]  (转置后的 mel)
  Output: conv_output    [1, 390,  896]  (降采样特征)
  │
  ▼
encoder（.axmodel 或 .onnx）
  Input:  input_features       [1, 390, 896]  (conv 输出)
          feature_attention_mask [1, 390]      (有效帧掩码)
  Output: audio_features       [1, 390, 1024] (与 LLM hidden_size 对齐)
  │
  ▼
Decoder 前处理（Python 或 C++）
  input_ids       [1, S]        文字提示的 token ID 序列，
                                其中音频占位符 token_id = 151676 (audio_token_id)
  audio_features  [1, 390, 1024] 上步得到的音频编码结果
  attention_mask  [1, S]         有效 token 位置标记（1=有效，0=padding）
  │
  ▼
Combined Embedding 构建（特征拼接核心）
  对 input_ids 中每个位置 i：
    if input_ids[i] == 151676:
        combined_embed[i] = audio_features[audio_rank]  # audio_rank 从 0 累加
    else:
        combined_embed[i] = embed_tokens.weight[input_ids[i]]  # 文字 token embedding
  Shape: [S, 1024], dtype: bfloat16
  │
  ▼
LLM 层（qwen3_asr_p64_l{n}_together.axmodel，共 28 层）
  Input:  input [1, seq, 1024] bfloat16  ← combined_embed
          K_cache, V_cache, mask, indices
  Output: output [1, seq, 1024] bfloat16
          K_cache_out, V_cache_out
  │
  ▼
Post（qwen3_asr_post.axmodel）
  Input:  input [1, 1, 1024] bfloat16
  Output: output [1, 1, 151936] bfloat16  ← logits over vocabulary
  │
  ▼
argmax → 下一个 token ID → 解码文字
```

---

## 2. 各输入的详细计算方式

### 2.1 `input_ids` — token ID 序列

- **类型**：`int64 [1, S]`，静态模型中 `S = STATIC_DECODER_SEQ = 390`
- **构造过程**：
  1. 使用 `AutoProcessor.apply_chat_template()` 构造文字提示（含系统提示 + 音频占位符）
  2. 提示中包含若干 `<|audio_pad|>` token（id=**151676**），数量等于 encoder 输出的音频 token 数（390）
  3. `AutoProcessor(text=..., audio=wav, ...)` 返回 `input_ids`，shape `[1, raw_S]`
  4. Padding 到 `STATIC_DECODER_SEQ=390`，多余部分用 `pad_id` 填充

- **示例**（简化）：
  ```
  [151644, 8948, 198, ..., 151669, 151676, 151676, ...(×390)..., 151670, ...]
   ^system_token                   ^audio_start  ^audio_pad tokens  ^audio_end
  ```

### 2.2 `audio_features` — 音频编码特征

- **类型**：`float32 [1, 390, 1024]`
- **构造过程**：
  1. 将音频 wav → mel 频谱 `[1, 128, T]`，`T` 最大为 3000 帧
  2. `conv_frontend` 做 3 次步长为 2 的卷积下采样：`3000 → 390`（90% 长度），每帧维度 128 → 896
  3. `encoder`（18 层 Transformer）将 `[1, 390, 896]` 映射为 `[1, 390, 1024]`
  4. 输出维度 1024 恰好等于 LLM 的 `hidden_size`

### 2.3 `attention_mask` — 注意力掩码

- **Prefill 阶段**：`int64 [1, S]`，有效 token 位置为 1，padding 位置为 0
- **Decode 阶段**：全 1 向量 `ones([1, S])`，每步生成的 token 都有效
- 在 decoder.onnx 内部用于计算 causal mask（SDPA 中）

### 2.4 `cache_position` — 位置 ID

- Prefill：`arange(0, S)`（当前处理的 token 在 KV cache 中的位置）
- Decode step n：`[cur_len]`（当前 decode token 的位置）

---

## 3. Decoder ONNX 内部的音频特征注入

`_inject_audio_features`（见 `decoder.py`）：

```python
def _inject_audio_features(self, tok, input_ids, audio_features):
    # tok: [B, S, H] — token embeddings（embed_tokens(input_ids)）
    # 找出 input_ids == audio_token_id 的位置
    mask = (input_ids == self.audio_token_id)  # [B, S]
    # 计算每个音频位置对应 audio_features 的下标（从 0 累加）
    rank = cumsum(mask) - 1                     # [B, S]
    rank = clamp(rank, 0, A-1)
    # 按 rank 从 audio_features 中 gather
    gathered = audio_features[:, rank, :]       # [B, S, H]
    # 仅在音频位置替换 embedding
    return where(mask, gathered, tok)           # [B, S, H]
```

**结论**：decoder.onnx 内部先用 `embed_tokens` 对所有 token 做 embedding，再将音频位置的 embedding 替换为 `audio_features` 向量。**特征拼接在 ONNX 模型内部完成**。

---

## 4. AX-LLM C++ 中的对应实现

由于 axmodel 层只接受 `input [bfloat16, 1, seq, 1024]`（embedding 向量），**特征拼接必须在 C++ 侧完成**，再送入 `LLM::Run(embed)`。

### 4.1 关键数据结构

| 数据 | 来源 | C++ 类型 |
|------|------|----------|
| 文字 token embedding | `LLaMaEmbedSelector::getByIndex(token_id, ...)` | `vector<unsigned short>` (bfloat16) |
| 音频特征 | encoder axmodel 输出 / npy 文件 | `float32 → bfloat16` 转换 |
| Combined embedding | 两者按 token 位置拼接 | `vector<unsigned short>` |

### 4.2 构建 Combined Embedding（C++ 伪代码）

```cpp
int audio_rank = 0;
for (int i = 0; i < num_tokens; i++) {
    if (input_ids[i] == AUDIO_TOKEN_ID) {
        // 将 float32 audio_features[0][audio_rank][*] 转 bfloat16 写入 embed
        for (int d = 0; d < hidden_size; d++) {
            bfloat16 bf(audio_features[audio_rank * hidden_size + d]);
            combined_embed[i * hidden_size + d] = bf.data;
        }
        audio_rank++;
    } else {
        embed_selector.getByIndex(input_ids[i], combined_embed.data() + i * hidden_size);
    }
}
// 调用 LLM 推理（prefill + greedy decode）
std::string result = llm.Run(combined_embed);
```

### 4.3 `LLM::Run(embed)` 内部流程

1. 按 `prefill_token_num=64` 分块做 prefill（每块 64 个 token）
2. 每次 prefill 调用每个 axmodel 层，更新 KV cache
3. Prefill 结束后调用 `post.axmodel` 得到第一个 token logits
4. 循环 decode：每步输入 `embed[1, 1, 1024]`，更新 KV，得到下一个 token
5. 遇到 EOS (`token_id = 151645` 等) 停止
6. 调用 tokenizer 将 token ID 序列解码为文字

---

## 5. 模型文件说明

| 文件 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `conv_frontend.axmodel` | 音频卷积前端 | `[1,3000,128]` mel | `[1,390,896]` |
| `encoder.axmodel` | 音频 Transformer 编码器 | `[1,390,896]` + mask | `[1,390,1024]` |
| `qwen3_asr_p64_l{n}_together.axmodel` | LLM 层 n（共28层）| embedding + KV cache | 新 embedding + KV out |
| `qwen3_asr_post.axmodel` | 语言模型头（lm_head）| `[1,1,1024]` | `[1,1,151936]` logits |
| `model.embed_tokens.weight.bfloat16.bin` | 词嵌入表 | token_id → lookup | `[1024]` bfloat16 |

---

## 6. 调试 npy 文件说明

由 `dump_asr_debug_npy.py` 生成，存放在 `<output_dir>/` 下：

| 文件 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `input_ids.npy` | `[1, S]` | int64 | 完整 token ID 序列（含音频占位符） |
| `audio_features.npy` | `[1, A, 1024]` | float32 | encoder 输出音频特征 |
| `attention_mask.npy` | `[1, S]` | int64 | prefill 注意力掩码 |
| `combined_embed_f32.npy` | `[S, 1024]` | float32 | 拼接后的 embedding（验证用）|
| `combined_embed_bf16.bin` | `S × 1024 × 2 bytes` | bfloat16 raw | C++ 直接加载的 bfloat16 嵌入 |
| `meta.json` | — | JSON | 元信息（S, A, hidden_size, audio_token_id 等）|
