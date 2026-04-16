#!/usr/bin/env python3
#
# Static encoder export wrapper for Qwen3-ASR ONNX (avoids dynamic Reshape/Pad).

import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_frontend import _pick


class StaticEncoderExport(nn.Module):
    """Encoder export wrapper with pre-computed static dimensions."""

    def __init__(
        self,
        backend_layers: nn.ModuleList,
        positional_embedding: nn.Module,
        ln_post: nn.Module,
        proj1: nn.Module,
        proj2: nn.Module,
        audio_proj,
        batch: int,
        time: int,
        tokens_per_chunk: int,
        window_size: int,
    ):
        super().__init__()
        self.backend_layers = backend_layers
        self.positional_embedding = positional_embedding
        self.ln_post = ln_post
        self.proj1 = proj1
        self.proj2 = proj2
        self.audio_proj = audio_proj
        
        self.batch = int(batch)
        self.time = int(time)
        self.tokens_per_chunk = int(tokens_per_chunk)
        self.window_size = int(window_size)
        
        pad_len = (self.window_size - (self.time % self.window_size)) % self.window_size
        self.pad_len = int(pad_len)
        self.tpad = self.time + self.pad_len
        self.nblk = self.tpad // self.window_size

    def forward(
        self, input_features: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        device = input_features.device
        pos_idx = torch.arange(self.time, device=device) % self.tokens_per_chunk
        pos_emb = F.embedding(
            pos_idx, self.positional_embedding.positional_embedding
        )
        x = input_features + pos_emb.unsqueeze(0).to(dtype=input_features.dtype)

        if token_mask is not None:
            x = x * token_mask.unsqueeze(-1).to(dtype=x.dtype)

        x = F.pad(x, (0, 0, 0, self.pad_len))
        x = x.view(self.batch, self.nblk, self.window_size, -1).contiguous()
        x = x.view(self.batch * self.nblk, self.window_size, -1)

        km = None
        if token_mask is not None:
            km = F.pad(token_mask, (0, self.pad_len), value=False)
            km = km.view(self.batch, self.nblk, self.window_size).contiguous()
            km = km.view(self.batch * self.nblk, self.window_size)

        for layer in self.backend_layers:
            residual = x
            attn_mod = _pick(layer, ["self_attn", "attention"])
            ln_mod = _pick(
                layer,
                ["self_attn_layer_norm", "input_layer_norm", "input_layernorm"],
            )
            x_norm = ln_mod(x)
            
            q = attn_mod.q_proj(x_norm)
            k = attn_mod.k_proj(x_norm)
            v = attn_mod.v_proj(x_norm)
            
            num_heads = int(
                getattr(attn_mod, "num_heads", getattr(attn_mod, "n_heads", 0)) or 0
            )
            head_dim = int(getattr(attn_mod, "head_dim", 0) or 0)
            if num_heads <= 0 or head_dim <= 0:
                q_out = int(attn_mod.q_proj.weight.shape[0])
                num_heads = max(1, num_heads)
                head_dim = int(q_out // num_heads)
            
            scale = head_dim**-0.5
            q = q.view(
                self.batch * self.nblk, self.window_size, num_heads, head_dim
            ).transpose(1, 2)
            k = k.view(
                self.batch * self.nblk, self.window_size, num_heads, head_dim
            ).transpose(1, 2)
            v = v.view(
                self.batch * self.nblk, self.window_size, num_heads, head_dim
            ).transpose(1, 2)
            
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale
            if km is not None:
                km_mask = km.to(dtype=scores.dtype).unsqueeze(1).unsqueeze(2)
                scores = scores + (1.0 - km_mask) * (-1e4)
            
            attn = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, v)
            attn_out = attn_out.transpose(1, 2).contiguous()
            attn_out = attn_out.view(
                self.batch * self.nblk, self.window_size, num_heads * head_dim
            )
            
            proj = getattr(attn_mod, "out_proj", None) or getattr(
                attn_mod, "o_proj", None
            )
            attn_out = proj(attn_out)
            if km is not None:
                attn_out = attn_out * km.unsqueeze(-1).to(dtype=attn_out.dtype)
            x = residual + attn_out

            residual = x
            ln_mod2 = _pick(
                layer,
                [
                    "final_layer_norm",
                    "post_attention_layer_norm",
                    "post_attention_layernorm",
                ],
            )
            x_norm2 = ln_mod2(x)
            
            fc1 = _pick(layer, ["fc1"])
            fc2 = _pick(layer, ["fc2"])
            if fc1 is None or fc2 is None:
                mlp = _pick(layer, ["mlp"])
                gate_out = mlp.gate_proj(x_norm2)
                up_out = mlp.up_proj(x_norm2)
                mlp_out = mlp.down_proj(F.silu(gate_out) * up_out)
            else:
                h = fc1(x_norm2)
                mlp_out = fc2(F.gelu(h))
            
            x = residual + mlp_out

            if km is not None:
                x = x * km.unsqueeze(-1).to(dtype=x.dtype)

        x = x.view(self.batch, self.nblk * self.window_size, -1)[:, : self.time]
        x = self.ln_post(x)
        x = self.proj1(x)
        x = F.gelu(x)
        x = self.proj2(x)

        if self.audio_proj is not None:
            x = self.audio_proj(x)

        x = x * token_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x.to(torch.float32)
