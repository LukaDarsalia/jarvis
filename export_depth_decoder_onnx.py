#!/usr/bin/env python3
# export_depth_decoder_onnx.py

import os
import torch
from transformers import CsmForConditionalGeneration
from csm_depth_decoder_wrapper import DepthDecoderOneStepExport

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_grad_enabled(False)

def main():
    device = "cuda"
    dtype = torch.float16

    model_path = "local_models/tts_model/georgian-csm-1b"
    out_path = "csm_depth_decoder_step_past31.onnx"

    model = CsmForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).eval()

    dd = model.depth_decoder.eval()
    wrapper = DepthDecoderOneStepExport(dd).eval().to(device)

    L = len(dd.model.layers)
    kv_heads = dd.config.num_key_value_heads
    hd = dd.model.layers[0].self_attn.head_dim
    Hbb = dd.config.backbone_hidden_size
    V = dd.config.vocab_size

    B = 1
    S = 1
    past_len = 31
    k_len = past_len + S  # 32

    print(f"Export: L={L} kv_heads={kv_heads} hd={hd} Hbb={Hbb} V={V} past_len={past_len} k_len={k_len}")

    input_ids = torch.zeros((B, S), dtype=torch.long, device=device)
    backbone_last_hidden_state = torch.zeros((B, Hbb), dtype=dtype, device=device)
    attention_mask = torch.zeros((B, 1, 1, k_len), dtype=dtype, device=device)
    cache_position = torch.zeros((S,), dtype=torch.long, device=device)  # [1]

    past_kv = []
    for _ in range(L):
        past_kv.append(torch.zeros((B, kv_heads, past_len, hd), dtype=dtype, device=device))
        past_kv.append(torch.zeros((B, kv_heads, past_len, hd), dtype=dtype, device=device))

    input_names = ["input_ids", "backbone_last_hidden_state", "attention_mask", "cache_position"] + [
        f"past_{i}" for i in range(2 * L)
    ]
    output_names = ["logits"] + [f"present_{i}" for i in range(2 * L)]

    torch.onnx.export(
        wrapper,
        (input_ids, backbone_last_hidden_state, attention_mask, cache_position, *past_kv),
        out_path,
        opset_version=18,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        dynamo=True,
    )

    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
