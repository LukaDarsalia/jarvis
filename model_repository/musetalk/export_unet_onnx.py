import argparse
import json
import os
from typing import Any, Dict, Tuple

import torch

from unet import UNet

DEFAULT_UNET_DIR = os.path.join("local_models", "musetalk_model", "musetalkV15")
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_UNET_DIR, "musetalk.json")
DEFAULT_WEIGHTS_PATH = os.path.join(DEFAULT_UNET_DIR, "unet.pth")
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_UNET_DIR, "unet.onnx")
DEFAULT_LATENTS_PATH = os.path.join(
    "local_models",
    "musetalk_model",
    "testing_avatar_creation",
    "v15",
    "avatars",
    "default",
    "latents.pt",
)


class UNetWrapper(torch.nn.Module):
    def __init__(self, unet: torch.nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_spatial(config: Dict[str, Any], height: int | None, width: int | None) -> Tuple[int, int]:
    if height is not None and width is not None:
        return height, width
    sample_size = config.get("sample_size")
    if isinstance(sample_size, (list, tuple)) and len(sample_size) == 2:
        return int(sample_size[0]), int(sample_size[1])
    if isinstance(sample_size, (int, float)):
        size = int(sample_size)
        return size, size
    raise ValueError("Unable to determine sample size; please set --height and --width.")

def infer_latent_shape() -> Tuple[int, int, int]:
    if not os.path.exists(DEFAULT_LATENTS_PATH):
        raise FileNotFoundError(f"Latents not found at {DEFAULT_LATENTS_PATH}")
    latents = torch.load(DEFAULT_LATENTS_PATH, map_location="cpu")
    if isinstance(latents, (list, tuple)):
        first = latents[0]
    else:
        first = latents
    if not isinstance(first, torch.Tensor) or first.ndim != 4:
        raise ValueError("Latents must be a 4D torch tensor")
    _, channels, height, width = first.shape
    return int(channels), int(height), int(width)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MuseTalk UNet to ONNX.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output ONNX path")
    parser.add_argument("--seq-len", type=int, default=50, help="Encoder hidden states sequence length")
    parser.add_argument("--fp16", action="store_true", help="Export in float16")
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    args = parser.parse_args()

    config = load_config(DEFAULT_CONFIG_PATH)
    latent_channels, height, width = infer_latent_shape()
    in_channels = int(config.get("in_channels", latent_channels))
    if in_channels != latent_channels:
        print(f"Warning: in_channels={in_channels} differs from latents={latent_channels}, using latents")
        in_channels = latent_channels

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.float16 if args.fp16 else torch.float32

    unet_loader = UNet(
        unet_config=DEFAULT_CONFIG_PATH,
        model_path=DEFAULT_WEIGHTS_PATH,
        device=device,
        use_float16=args.fp16,
    )
    unet_loader.model.to(device=device, dtype=dtype).eval()

    wrapper = UNetWrapper(unet_loader.model).to(device=device, dtype=dtype).eval()

    batch = 1
    sample = torch.randn(batch, in_channels, height, width, device=device, dtype=dtype)
    timestep = torch.zeros(1, device=device, dtype=torch.int64)
    encoder_hidden_states = torch.randn(batch, args.seq_len, 384, device=device, dtype=dtype)

    dynamic_axes = {
        "sample": {0: "batch"},
        "timestep": {0: "timestep"},
        "encoder_hidden_states": {0: "batch", 1: "seq_len"},
        "output": {0: "batch"},
    }

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (sample, timestep, encoder_hidden_states),
        args.output,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    print(f"Exported UNet ONNX to {args.output}")
    _run_smoke_test(args.output, sample, timestep, encoder_hidden_states, wrapper)


def _run_smoke_test(
    onnx_path: str,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    wrapper: torch.nn.Module,
) -> None:
    try:
        import onnxruntime as ort
    except Exception as exc:
        print(f"ONNX Runtime not available for test: {exc}")
        return

    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)

    inputs = {
        "sample": sample.detach().cpu().numpy(),
        "timestep": timestep.detach().cpu().numpy(),
        "encoder_hidden_states": encoder_hidden_states.detach().cpu().numpy(),
    }
    outputs = sess.run(None, inputs)
    onnx_out = outputs[0]
    torch_out = wrapper(
        sample,
        timestep,
        encoder_hidden_states,
    ).detach().cpu().numpy()

    onnx_out_f32 = onnx_out.astype("float32")
    torch_out_f32 = torch_out.astype("float32")
    diff = onnx_out_f32 - torch_out_f32
    max_abs = float(abs(diff).max())
    mean_abs = float(abs(diff).mean())
    print(f"ONNX test output shape: {onnx_out.shape}")
    print(f"ONNX vs Torch | max abs diff: {max_abs:.6f} | mean abs diff: {mean_abs:.6f}")


if __name__ == "__main__":
    main()
