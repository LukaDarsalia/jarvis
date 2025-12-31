# csm_backbone_step_wrapper.py
import torch
import torch.nn as nn

from csm_depth_decoder_wrapper import ExportableDecoderLayerOneStep


class BackboneOneStepExport(nn.Module):
    """
    One-step backbone export wrapper.

    Inputs:
      inputs_embeds: [B,1,D]
      attention_mask: [B,1,1,Tpast+1] additive
      cache_position: [1]
      past_0..past_(2L-1): [B,kv,Tpast,hd]
    Outputs:
      logits: [B,1,V]
      last_hidden_state: [B,H]
      present_0..present_(2L-1): [B,kv,Tpast+1,hd]
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
        self.layers = nn.ModuleList([ExportableDecoderLayerOneStep(l) for l in backbone.layers])
        self.norm = backbone.norm
        self.rotary_emb = backbone.rotary_emb

    def forward(self, inputs_embeds, attention_mask, cache_position, *past_kv_flat):
        B, S, _ = inputs_embeds.shape
        assert S == 1, "one-step only: inputs_embeds must be [B,1,D]"
        L = len(self.layers)
        assert len(past_kv_flat) == 2 * L

        position_ids = cache_position.unsqueeze(0)  # [1,1]
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)

        hidden_states = inputs_embeds
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
        last_hidden_state = hidden_states[:, -1, :]

        if self.lm_head is None:
            return (last_hidden_state, *new_kv)

        logits = self.lm_head(hidden_states)
        return (logits, last_hidden_state, *new_kv)
