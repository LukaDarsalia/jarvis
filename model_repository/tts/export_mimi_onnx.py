#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from torch import nn
from transformers import CsmForConditionalGeneration


DEFAULT_MODEL_PATH = "local_models/tts_model/georgian-csm-1b"
DEFAULT_REF_AUDIO = "local_models/tts_model/georgian-csm-1b/context_audio_for_inference.wav"
DEFAULT_OUT_DIR = "local_models/tts_model/mimi_onnx_local"


class MimiEncoderWrapper(nn.Module):
    def __init__(self, codec_model: nn.Module):
        super().__init__()
        self.codec_model = codec_model

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.codec_model.encode(input_values).audio_codes


class MimiDecoderWrapper(nn.Module):
    def __init__(self, codec_model: nn.Module):
        super().__init__()
        self.codec_model = codec_model

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        audio = self.codec_model.decode(audio_codes).audio_values
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return audio


def _simple_causal_mask(
    config=None,
    input_embeds: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.Tensor] = None,
    past_key_values=None,
    position_ids: Optional[torch.Tensor] = None,
):
    if input_embeds is None:
        raise ValueError("input_embeds is required for simple causal mask.")
    batch_size, tgt_len, _ = input_embeds.shape
    src_len = tgt_len
    dtype = input_embeds.dtype
    device = input_embeds.device
    neg_inf = torch.finfo(dtype).min

    mask = torch.triu(torch.ones((tgt_len, src_len), device=device, dtype=dtype), diagonal=1)
    mask = mask * neg_inf
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, tgt_len, src_len)

    if attention_mask is not None:
        if attention_mask.dim() == 2:
            attn = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 4:
            attn = attention_mask
        else:
            attn = attention_mask.view(batch_size, 1, 1, -1)
        mask = mask + (1.0 - attn.to(dtype)) * neg_inf
    return mask


def _quantize_no_cdist(self, hidden_states: torch.Tensor) -> torch.Tensor:
    x = hidden_states.float()
    y = self.embed.float()
    x_norm = (x * x).sum(dim=-1, keepdim=True)
    y_norm = (y * y).sum(dim=-1).unsqueeze(0)
    dists = x_norm + y_norm - 2.0 * (x @ y.t())
    return dists.argmin(dim=-1)


def _get_audio_length(audio_path: str) -> int:
    audio, _ = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return int(audio.shape[0])


def _parse_dtype(value: str) -> torch.dtype:
    value = value.lower()
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Mimi codec (from CSM) to ONNX.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--reference-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--encoder-length", type=int, default=0)
    parser.add_argument("--decoder-codes", type=int, default=32)
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--encoder-name", default="encoder_model.onnx")
    parser.add_argument("--decoder-name", default="decoder_model.onnx")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dynamo", action="store_true")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_out = out_dir / args.encoder_name
    decoder_out = out_dir / args.decoder_name

    encoder_length = args.encoder_length
    if encoder_length <= 0:
        encoder_length = _get_audio_length(args.reference_audio)

    print(f"Loading CSM model from: {args.model_path}")
    model = CsmForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="cpu",
        attn_implementation="sdpa",
        torch_dtype=dtype,
    ).eval()
    try:
        import transformers.masking_utils as masking_utils
        import transformers.models.mimi.modeling_mimi as modeling_mimi
        masking_utils.create_causal_mask = _simple_causal_mask
        modeling_mimi.create_causal_mask = _simple_causal_mask
        modeling_mimi.MimiEuclideanCodebook.quantize = _quantize_no_cdist
        print("Using simplified causal mask for export.")
    except Exception as exc:
        print(f"Failed to patch causal mask: {exc}")
    codec_model = model.codec_model.to(device=device, dtype=dtype).eval()

    encoder = MimiEncoderWrapper(codec_model).eval().to(device)
    decoder = MimiDecoderWrapper(codec_model).eval().to(device)

    num_codebooks = int(codec_model.config.num_codebooks)
    codebook_size = int(codec_model.config.codebook_size)

    encoder_input = torch.randn(
        (1, 1, encoder_length),
        device=device,
        dtype=dtype,
    )
    decoder_input = torch.randint(
        0,
        codebook_size,
        (1, num_codebooks, args.decoder_codes),
        device=device,
        dtype=torch.long,
    )

    print(f"Exporting encoder -> {encoder_out} (length={encoder_length})")
    torch.onnx.export(
        encoder,
        encoder_input,
        str(encoder_out),
        opset_version=args.opset,
        input_names=["input_values"],
        output_names=["audio_codes"],
        do_constant_folding=True,
        dynamo=args.dynamo,
    )

    print(f"Exporting decoder -> {decoder_out} (codes={args.decoder_codes})")
    torch.onnx.export(
        decoder,
        decoder_input,
        str(decoder_out),
        opset_version=args.opset,
        input_names=["audio_codes"],
        output_names=["audio_values"],
        dynamic_axes={
            "audio_codes": {0: "batch_size", 2: "codes_length"},
            "audio_values": {0: "batch_size", 2: "sequence_length"},
        },
        do_constant_folding=True,
        dynamo=args.dynamo,
    )

    print("Done.")
    print(f"Encoder ONNX: {encoder_out}")
    print(f"Decoder ONNX: {decoder_out}")


if __name__ == "__main__":
    main()
