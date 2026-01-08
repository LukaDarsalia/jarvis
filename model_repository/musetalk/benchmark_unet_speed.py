import argparse
import os
import time
from typing import Tuple

import numpy as np
import torch

from unet import UNet

DEFAULT_UNET_DIR = os.path.join("local_models", "musetalk_model", "musetalkV15")
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_UNET_DIR, "musetalk.json")
DEFAULT_WEIGHTS_PATH = os.path.join(DEFAULT_UNET_DIR, "unet.pth")
DEFAULT_ONNX_PATH = os.path.join(DEFAULT_UNET_DIR, "unet.onnx")
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


def load_latent_shape(latents_path: str) -> Tuple[int, int, int]:
    latents = torch.load(latents_path, map_location="cpu")
    if isinstance(latents, (list, tuple)):
        first = latents[0]
    else:
        first = latents
    if not isinstance(first, torch.Tensor) or first.ndim != 4:
        raise ValueError("Latents must be a 4D torch tensor")
    _, channels, height, width = first.shape
    return int(channels), int(height), int(width)


def get_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def torch_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def summarize_times(label: str, times_ms: list[float]) -> None:
    arr = np.array(times_ms, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    mean = float(arr.mean())
    print(f"{label} | mean {mean:.2f} ms | p50 {p50:.2f} ms | p95 {p95:.2f} ms | iters {len(times_ms)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MuseTalk UNet speed (PyTorch vs ONNX).")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--fp16", action="store_true", help="Use float16 for PyTorch inputs")
    parser.add_argument("--seq-len", type=int, default=50, help="Encoder hidden states sequence length")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--onnx-path", default=DEFAULT_ONNX_PATH, help="Path to ONNX model")
    parser.add_argument("--unet-config", default=DEFAULT_CONFIG_PATH, help="Path to unet.json")
    parser.add_argument("--unet-weights", default=DEFAULT_WEIGHTS_PATH, help="Path to unet.pth")
    parser.add_argument("--latents", default=DEFAULT_LATENTS_PATH, help="Path to latents.pt")
    args = parser.parse_args()

    channels, height, width = load_latent_shape(args.latents)
    device = get_device(args.device)
    dtype = torch.float16 if args.fp16 else torch.float32

    unet_loader = UNet(
        unet_config=args.unet_config,
        model_path=args.unet_weights,
        device=device,
        use_float16=args.fp16,
    )
    wrapper = UNetWrapper(unet_loader.model).to(device=device, dtype=dtype).eval()

    batch = 1
    sample = torch.randn(batch, channels, height, width, device=device, dtype=dtype)
    timestep = torch.zeros(1, device=device, dtype=torch.int64)
    encoder_hidden_states = torch.randn(batch, args.seq_len, 384, device=device, dtype=dtype)

    try:
        import onnxruntime as ort
    except Exception as exc:
        ort = None
        print(f"ONNX Runtime not available: {exc}")

    ort_session = None
    if ort is not None:
        providers = ["CPUExecutionProvider"]
        if device.type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ort_session = ort.InferenceSession(args.onnx_path, providers=providers)

    # Warmup PyTorch
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = wrapper(sample, timestep, encoder_hidden_states)
        torch_sync(device)

    torch_times = []
    with torch.no_grad():
        for _ in range(args.iters):
            start = time.perf_counter()
            _ = wrapper(sample, timestep, encoder_hidden_states)
            torch_sync(device)
            torch_times.append((time.perf_counter() - start) * 1000.0)
    summarize_times("PyTorch", torch_times)

    if ort_session is None:
        return

    ort_inputs = {
        "sample": sample.detach().cpu().numpy(),
        "timestep": timestep.detach().cpu().numpy(),
        "encoder_hidden_states": encoder_hidden_states.detach().cpu().numpy(),
    }

    # Warmup ONNX
    for _ in range(args.warmup):
        _ = ort_session.run(None, ort_inputs)

    ort_times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        _ = ort_session.run(None, ort_inputs)
        ort_times.append((time.perf_counter() - start) * 1000.0)
    summarize_times("ONNX Runtime", ort_times)


if __name__ == "__main__":
    main()
