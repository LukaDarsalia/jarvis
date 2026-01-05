# csm_depth_decoder_wrapper.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.csm.modeling_csm import apply_rotary_pos_emb


class ExportableCsmCodebooksHead(nn.Module):
    """
    Match HF behavior:
      codebook_idxs = cache_position - 1   (NO CLAMP)
      weight[codebook_idxs] -> [S,H,V]
      logits = linear(hidden_states[:,i,:], weight[i].T)

    For one-step (S=1) this reduces to [B,1,V].
    """
    def __init__(self, hf_codebooks_head: nn.Module):
        super().__init__()
        self.weight = hf_codebooks_head.weight  # [num_codebooks-1, H, V]

    def forward(self, hidden_states: torch.Tensor, cache_position: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, S=1, H]
        # cache_position: [S=1]
        idx = cache_position - 1                      # [1]  (can be -1 if cache_position==0)
        w = self.weight[idx]                          # [1, H, V]
        logits = torch.einsum("bsh,shv->bsv", hidden_states, w)
        return logits


class ExportableAttnOneStep(nn.Module):
    """
    Correct cache semantics:
      past_k/past_v in kv-head space: [B, kv_heads, Tpast, hd]
      q in heads space, k/v in kv space
      RoPE on q and k
      concat cache in kv space
      repeat kv to heads only for SDPA compute
      return present in kv space
    """
    def __init__(self, attn: nn.Module):
        super().__init__()
        self.config = attn.config
        self.layer_idx = attn.layer_idx

        self.head_dim = attn.head_dim
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # IMPORTANT: do NOT multiply q by scaling when using torch SDPA.
        # If torch supports `scale=`, we pass it. Otherwise rely on default 1/sqrt(d).
        self.scaling = getattr(attn, "scaling", None)

        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.o_proj = attn.o_proj

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv_heads == self.num_heads:
            return x
        return x.repeat_interleave(self.num_kv_groups, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,     # [B,1,D]
        position_embeddings,             # (cos, sin)
        attention_mask: torch.Tensor,    # [B,1,1,T] additive
        past_k: torch.Tensor,            # [B,kv,Tpast,hd]
        past_v: torch.Tensor,            # [B,kv,Tpast,hd]
    ):
        B, q_len, _ = hidden_states.shape
        assert q_len == 1, "one-step only"

        q = self.q_proj(hidden_states).view(B, q_len, self.num_heads, self.head_dim).transpose(1, 2)      # [B,h,1,hd]
        k = self.k_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [B,kv,1,hd]
        v = self.v_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [B,kv,1,hd]

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k_full = torch.cat([past_k, k], dim=2)  # [B,kv,Tpast+1,hd]
        v_full = torch.cat([past_v, v], dim=2)  # [B,kv,Tpast+1,hd]

        k_rep = self._repeat_kv(k_full)         # [B,h,T,hd]
        v_rep = self._repeat_kv(v_full)         # [B,h,T,hd]

        # SDPA: do not pre-scale q. Use scale= if supported.
        try:
            attn_out = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=(float(self.scaling) if self.scaling is not None else None),
            )
        except TypeError:
            # older torch: no `scale=` arg. Default internal scaling is 1/sqrt(d).
            attn_out = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, self.num_heads * self.head_dim)
        attn_out = self.o_proj(attn_out)

        return attn_out, k_full, v_full


class ExportableDecoderLayerOneStep(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp
        self.self_attn = ExportableAttnOneStep(layer.self_attn)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_k, past_v):
        residual = hidden_states
        x = self.input_layernorm(hidden_states)

        attn_out, k_full, v_full = self.self_attn(
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
        return x, k_full, v_full


class DepthDecoderOneStepExport(nn.Module):
    """
    One-step depth decoder export wrapper.

    Inputs:
      input_ids: [B,1]
      backbone_last_hidden_state: [B,Hbb]
      attention_mask: [B,1,1,Tpast+1] additive
      cache_position: [1]
      past_0..past_(2L-1): [B,kv,Tpast,hd]
    Outputs:
      logits: [B,1,V]
      present_0..present_(2L-1): [B,kv,Tpast+1,hd]
    """
    def __init__(self, dd_module: nn.Module):
        super().__init__()
        self.dd = dd_module
        self.model = dd_module.model
        self.config = dd_module.config
        self.vocab_size = dd_module.config.vocab_size

        self.layers = nn.ModuleList([ExportableDecoderLayerOneStep(l) for l in self.model.layers])
        self.norm = self.model.norm
        self.rotary_emb = self.model.rotary_emb
        self.codebooks_head = ExportableCsmCodebooksHead(dd_module.codebooks_head)

    def forward(self, input_ids, backbone_last_hidden_state, attention_mask, cache_position, *past_kv_flat):
        B, S = input_ids.shape
        assert S == 1, "one-step only: input_ids must be [B,1]"
        L = len(self.layers)
        assert len(past_kv_flat) == 2 * L

        # Embedding offset matches HF:
        codebook_idxs = torch.clamp(cache_position - 1, min=0)     # [1]
        offset = codebook_idxs * self.vocab_size                   # [1]
        inputs_embeds = self.model.embed_tokens(input_ids + offset)

        # Match Generation semantics when backbone_last_hidden_state is always passed:
        # Use it only at cache_position==0
        use_bb = (cache_position == 0).to(dtype=inputs_embeds.dtype).view(1, 1, 1)  # [1,1,1]
        bb = backbone_last_hidden_state.to(inputs_embeds.dtype).view(B, 1, -1)
        inputs_embeds = inputs_embeds * (1.0 - use_bb) + bb * use_bb

        hidden_states = self.model.inputs_embeds_projector(inputs_embeds)

        position_ids = cache_position.unsqueeze(0)  # [1,1]
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        new_kv = []
        for i in range(L):
            past_k = past_kv_flat[2 * i + 0]
            past_v = past_kv_flat[2 * i + 1]
            hidden_states, k_full, v_full = self.layers[i](
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_k=past_k,
                past_v=past_v,
            )
            new_kv.append(k_full)
            new_kv.append(v_full)

        hidden_states = self.norm(hidden_states)
        logits = self.codebooks_head(hidden_states, cache_position=cache_position)
        return (logits, *new_kv)
