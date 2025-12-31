#!/usr/bin/env python3
# export_backbone_step_onnx.py

import os
import torch
from transformers import CsmForConditionalGeneration

from csm_backbone_step_wrapper import BackboneOneStepExport
from export_depth_decoder_onnx import (
    BACKBONE_ONNX_PATH,
    BACKBONE_PAST_LEN,
    EXPORT_DEVICE,
    EXPORT_DTYPE,
    MODEL_PATH,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_grad_enabled(False)


def main():
    device = EXPORT_DEVICE
    dtype = EXPORT_DTYPE

    model = CsmForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map=device,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).eval()

    wrapper = BackboneOneStepExport(model).eval().to(device)

    L = len(model.backbone_model.layers)
    kv_heads = model.config.num_key_value_heads
    hd = model.backbone_model.layers[0].self_attn.head_dim
    H = model.config.hidden_size
    V = model.config.vocab_size

    B = 1
    S = 1
    past_len = BACKBONE_PAST_LEN
    k_len = past_len + 1

    print(
        "Export backbone static-KV:",
        f"L={L}",
        f"kv_heads={kv_heads}",
        f"hd={hd}",
        f"H={H}",
        f"V={V}",
        f"past_len={past_len}",
        f"k_len={k_len}",
    )

    inputs_embeds = torch.zeros((B, S, H), dtype=dtype, device=device)
    attention_mask = torch.zeros((B, 1, 1, k_len), dtype=dtype, device=device)
    cache_position = torch.zeros((S,), dtype=torch.long, device=device)

    # STATIC cache buffers (fixed past_len)
    past_kv = []
    for _ in range(L):
        past_kv.append(torch.zeros((B, kv_heads, past_len, hd), dtype=dtype, device=device))
        past_kv.append(torch.zeros((B, kv_heads, past_len, hd), dtype=dtype, device=device))

    input_names = ["inputs_embeds", "attention_mask", "cache_position"] + [f"past_{i}" for i in range(2 * L)]
    output_names = ["logits", "last_hidden_state"] + [f"new_{i}" for i in range(2 * L)]

    torch.onnx.export(
        wrapper,
        (inputs_embeds, attention_mask, cache_position, *past_kv),
        BACKBONE_ONNX_PATH,
        opset_version=18,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        dynamo=True,
    )

    print("Wrote:", BACKBONE_ONNX_PATH)


if __name__ == "__main__":
    main()
