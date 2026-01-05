# csm_backbone_step_wrapper.py
import math
import torch
import torch.nn as nn

from transformers.models.csm.modeling_csm import apply_rotary_pos_emb


class ExportableAttnOneStepStaticKV(nn.Module):
    """
    One-step attention with STATIC KV cache.

    Returns only (k_new, v_new) = [B,kv,1,hd].
    """
    def __init__(self, attn: nn.Module):
        super().__init__()
        self.config = attn.config

        self.head_dim = attn.head_dim
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.scaling = float(getattr(attn, "scaling", 1.0 / math.sqrt(self.head_dim)))

        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.o_proj = attn.o_proj

    def _repeat_kv_noalloc(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, kv, T, hd] -> [B, h, T, hd] view
        if self.num_kv_heads == self.num_heads:
            return x
        B, kv, T, hd = x.shape
        x = x[:, :, None, :, :]                        # [B,kv,1,T,hd]
        x = x.expand(B, kv, self.num_kv_groups, T, hd)  # view
        return x.reshape(B, kv * self.num_kv_groups, T, hd)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_k, past_v):
        B, q_len, _ = hidden_states.shape
        assert q_len == 1

        q = self.q_proj(hidden_states).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)      # [B,h,1,hd]
        k = self.k_proj(hidden_states).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [B,kv,1,hd]
        v = self.v_proj(hidden_states).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [B,kv,1,hd]

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k_past = self._repeat_kv_noalloc(past_k)  # [B,h,T,hd]
        v_past = self._repeat_kv_noalloc(past_v)  # [B,h,T,hd]
        k_new_h = self._repeat_kv_noalloc(k)      # [B,h,1,hd]
        v_new_h = self._repeat_kv_noalloc(v)      # [B,h,1,hd]

        scores_past = torch.matmul(q, k_past.transpose(-1, -2))  # [B,h,1,T]
        scores_new = torch.matmul(q, k_new_h.transpose(-1, -2))  # [B,h,1,1]
        scores = torch.cat([scores_past, scores_new], dim=-1)    # [B,h,1,T+1]

        scores = scores * self.scaling
        scores = scores + attention_mask  # [B,1,1,T+1] broadcast

        # IMPORTANT: keep in fp16/bf16. Do NOT force fp32 here.
        attn = torch.softmax(scores, dim=-1)  # [B,h,1,T+1]

        w_past = attn[..., :-1]
        w_new = attn[..., -1:]

        out_past = torch.matmul(w_past, v_past)  # [B,h,1,hd]
        out_new = w_new * v_new_h                # [B,h,1,hd]
        attn_out = out_past + out_new

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, self.num_heads * self.head_dim)
        attn_out = self.o_proj(attn_out)

        # Return only the new KV in kv-head space
        return attn_out, k, v


class ExportableDecoderLayerOneStepStaticKV(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp
        self.self_attn = ExportableAttnOneStepStaticKV(layer.self_attn)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_k, past_v):
        residual = hidden_states
        x = self.input_layernorm(hidden_states)

        attn_out, k_new, v_new = self.self_attn(
            x,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_k=past_k,
            past_v=past_v,
        )
        x = residual + attn_out

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x, k_new, v_new


class BackboneOneStepExport(nn.Module):
    """
    Outputs:
      logits: [B,1,V]
      last_hidden_state: [B,H]
      new_0..new_(2L-1): [B,kv,1,hd]
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        if hasattr(model, "backbone_model"):
            backbone = model.backbone_model
            self.lm_head = model.lm_head
        else:
            backbone = model
            self.lm_head = None

        self.backbone = backbone
        self.config = backbone.config
        self.layers = nn.ModuleList([ExportableDecoderLayerOneStepStaticKV(l) for l in backbone.layers])
        self.norm = backbone.norm
        self.rotary_emb = backbone.rotary_emb

    def forward(self, inputs_embeds, attention_mask, cache_position, *past_kv_flat):
        B, S, _ = inputs_embeds.shape
        assert S == 1
        L = len(self.layers)
        assert len(past_kv_flat) == 2 * L

        position_ids = cache_position.unsqueeze(0)  # [1,1]
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)

        hidden_states = inputs_embeds
        new_kv = []
        for i in range(L):
            past_k = past_kv_flat[2 * i + 0]
            past_v = past_kv_flat[2 * i + 1]
            hidden_states, k_new, v_new = self.layers[i](
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_k=past_k,
                past_v=past_v,
            )
            new_kv.append(k_new)
            new_kv.append(v_new)

        hidden_states = self.norm(hidden_states)
        last_hidden_state = hidden_states[:, -1, :]

        if self.lm_head is None:
            return (last_hidden_state, *new_kv)

        logits = self.lm_head(hidden_states)
        return (logits, last_hidden_state, *new_kv)
