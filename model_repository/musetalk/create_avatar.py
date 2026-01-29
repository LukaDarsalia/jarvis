"""Utility script to precompute avatar assets.

Run this script inside the legacy `avatar_venv` environment where mmpose is
available. It prepares all intermediate files (frames, masks, latents, etc.) so
that other scripts can load the avatar without requiring mmpose.
"""

import argparse
import os
import sys

import torch

# Ensure repository root is on the path when running as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from avatar import Avatar, AvatarConfig, AvatarRuntime  # noqa: E402
from face_parsing import FaceParsing  # noqa: E402
from utils import load_all_model  # noqa: E402


def build_runtime(
    device: torch.device,
    version: str,
    left_cheek_width: int,
    right_cheek_width: int,
    unet_model_path: str,
    unet_config: str,
    vae_model_path: str,
    face_parse_resnet_path: str,
    face_parse_model_path: str,
) -> AvatarRuntime:
    """Create a minimal AvatarRuntime for asset preparation."""
    vae, unet, pe = load_all_model(
        device=device,
        unet_model_path=unet_model_path,
        unet_config=unet_config,
        vae_model_path=vae_model_path,
    )

    face_parser = (
        FaceParsing(
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
            resnet_path=face_parse_resnet_path,
            model_pth=face_parse_model_path,
            device=device,
        )
        if version == "v15"
        else FaceParsing(
            resnet_path=face_parse_resnet_path,
            model_pth=face_parse_model_path,
            device=device,
        )
    )

    # Positional encoding and diffusion UNet expect tensors on the same device.
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    return AvatarRuntime(
        device=device,
        vae=vae,
        unet=unet,
        pe=pe,
        audio_processor=None,
        whisper=None,
        face_parser=face_parser,
        timesteps=torch.tensor([0], device=device),
        weight_dtype=unet.model.dtype,
    )


def prepare_avatar(args: argparse.Namespace) -> None:
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu_id}" if use_cuda else "cpu")
    model_root = args.model_root
    unet_model_path = args.unet_model_path or os.path.join(model_root, "musetalkV15", "unet.pth")
    unet_config = args.unet_config or os.path.join(model_root, "musetalkV15", "musetalk.json")
    vae_model_path = args.vae_model_path or os.path.join(model_root, "sd-vae")
    face_parse_resnet_path = args.face_parse_resnet_path or os.path.join(
        model_root,
        "face-parse-bisent",
        "resnet18-5c106cde.pth",
    )
    face_parse_model_path = args.face_parse_model_path or os.path.join(
        model_root,
        "face-parse-bisent",
        "79999_iter.pth",
    )
    runtime = build_runtime(
        device=device,
        version=args.version,
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width,
        unet_model_path=unet_model_path,
        unet_config=unet_config,
        vae_model_path=vae_model_path,
        face_parse_resnet_path=face_parse_resnet_path,
        face_parse_model_path=face_parse_model_path,
    )

    config = AvatarConfig(
        version=args.version,
        result_root=args.result_dir,
        extra_margin=args.extra_margin,
        parsing_mode=args.parsing_mode,
        skip_save_images=args.skip_save_images,
        fps=args.fps,
        batch_size=args.batch_size,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
        max_side=args.max_side,
    )

    avatar_id = args.avatar_id or os.path.splitext(os.path.basename(args.video_path))[0]

    print("Preparing avatar assets...")
    Avatar(
        avatar_id=avatar_id,
        video_path=args.video_path,
        bbox_shift=args.bbox_shift,
        preparation=True,
        config=config,
        runtime=runtime,
        interactive=False,
        force_recreate=args.force_recreate,
    )
    print(f"Avatar '{avatar_id}' assets stored under {config.result_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute avatar assets (requires mmpose)")
    parser.add_argument("--video_path", type=str, required=True, help="Source video for the avatar")
    parser.add_argument("--avatar_id", type=str, default=None, help="Identifier for the avatar folder")
    parser.add_argument("--result_dir", type=str, default="./results", help="Root directory for results")
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Model version")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding-box adjustment for v1")
    parser.add_argument("--extra_margin", type=int, default=10, help="Additional pixels for v15 crops")
    parser.add_argument("--parsing_mode", type=str, default="jaw", help="Face parsing mode")
    parser.add_argument("--fps", type=int, default=25, help="Frame rate for cached assets")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size stored in avatar config")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding used during inference")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding used during inference")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Face parsing parameter (v15 only)")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Face parsing parameter (v15 only)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU index to use if CUDA is available")
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("AVATAR_DEVICE", "cuda"),
        choices=["cuda", "cpu"],
        help="Force device for avatar creation (cuda or cpu)",
    )
    parser.add_argument(
        "--max_side",
        type=int,
        default=int(os.environ.get("AVATAR_MAX_SIDE", "0")),
        help="Optional max image side (downscale frames to reduce inference cost)",
    )
    parser.add_argument("--skip_save_images", action="store_true", help="Store avatar config with skip-save flag")
    parser.add_argument("--force_recreate", action="store_true", help="Overwrite existing avatar assets")
    parser.add_argument(
        "--model_root",
        type=str,
        default=os.environ.get("MUSETALK_MODEL_ROOT", "/local_models/musetalk_model"),
        help="Root directory for musetalk model weights",
    )
    parser.add_argument("--unet_model_path", type=str, default=None, help="Path to UNet weights")
    parser.add_argument("--unet_config", type=str, default=None, help="Path to UNet config")
    parser.add_argument("--vae_model_path", type=str, default=None, help="Path to VAE model directory")
    parser.add_argument("--face_parse_resnet_path", type=str, default=None, help="Path to face parse resnet weights")
    parser.add_argument("--face_parse_model_path", type=str, default=None, help="Path to face parse model weights")
    return parser.parse_args()


if __name__ == "__main__":
    prepare_avatar(parse_args())
