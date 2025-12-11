import copy
import glob
import json
import os
import pickle
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from blending import get_image_blending, get_image_prepare_material
from utils import datagen


def video2imgs(vid_path: str, save_path: str, ext: str = ".png", cut_frame: int = 10_000_000) -> None:
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}{ext}", frame)
            count += 1
        else:
            break
    cap.release()


def osmakedirs(path_list) -> None:
    """Create directories if they do not exist."""
    for path in path_list:
        os.makedirs(path, exist_ok=True)


@dataclass
class AvatarConfig:
    version: str = "v15"
    result_root: str = "./results"
    extra_margin: int = 10
    parsing_mode: str = "jaw"
    skip_save_images: bool = False
    fps: int = 25
    batch_size: int = 20
    audio_padding_length_left: int = 2
    audio_padding_length_right: int = 2


@dataclass
class AvatarRuntime:
    device: torch.device
    vae: Any
    unet: Any
    pe: torch.nn.Module
    audio_processor: Any
    whisper: Any
    face_parser: Any
    timesteps: torch.Tensor
    weight_dtype: torch.dtype


class Avatar:
    """Manage avatar assets (frames, masks, latents) and perform inference."""

    def __init__(
        self,
        avatar_id: str,
        video_path: Optional[str],
        bbox_shift: int,
        preparation: bool,
        config: AvatarConfig,
        runtime: AvatarRuntime,
        interactive: bool = True,
        force_recreate: bool = False,
        avatar_path_override: Optional[str] = None,
    ) -> None:
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.preparation = preparation
        self.config = config
        self.runtime = runtime
        self.interactive = interactive
        self.force_recreate = force_recreate

        # Paths
        if avatar_path_override is not None:
            base_path = avatar_path_override
        else:
            if self.config.version == "v15":
                base_path = os.path.join(self.config.result_root, self.config.version, "avatars", avatar_id)
            else:
                base_path = os.path.join(self.config.result_root, "avatars", avatar_id)

        self.avatar_path = base_path
        self.full_imgs_path = os.path.join(self.avatar_path, "full_imgs")
        self.coords_path = os.path.join(self.avatar_path, "coords.pkl")
        self.latents_out_path = os.path.join(self.avatar_path, "latents.pt")
        self.video_out_path = os.path.join(self.avatar_path, "vid_output")
        self.mask_out_path = os.path.join(self.avatar_path, "mask")
        self.mask_coords_path = os.path.join(self.avatar_path, "mask_coords.pkl")
        self.avatar_info_path = os.path.join(self.avatar_path, "avator_info.json")

        self.avatar_info: Dict[str, Any] = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": self.config.version,
        }

        self.batch_size = self.config.batch_size
        self.idx = 0

        # Prepared data containers
        self.input_latent_list_cycle: Optional[list] = None
        self.coord_list_cycle: Optional[list] = None
        self.frame_list_cycle: Optional[list] = None
        self.mask_list_cycle: Optional[list] = None
        self.mask_coords_list_cycle: Optional[list] = None

        self._init_assets()

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _read_image_sequence(paths: List[str]) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        if not paths:
            return frames
        print(f"loading {len(paths)} image(s) from disk...")
        for img_path in tqdm(paths, desc="load_imgs"):
            frame = cv2.imread(img_path)
            if frame is None:
                raise FileNotFoundError(f"unable to read image: {img_path}")
            frames.append(frame)
        return frames

    # ---------------------------------------------------------------------
    # Preparation utilities
    # ---------------------------------------------------------------------
    def _init_assets(self) -> None:
        if self.preparation:
            if os.path.exists(self.avatar_path):
                if self.force_recreate:
                    shutil.rmtree(self.avatar_path)
                    self._prepare_fresh_avatar()
                elif self.interactive:
                    response = input(f"{self.avatar_id} exists, re-create it? (y/n)")
                    if response.lower() == "y":
                        shutil.rmtree(self.avatar_path)
                        self._prepare_fresh_avatar()
                    else:
                        self._load_cached_material()
                else:
                    self._load_cached_material()
            else:
                self._prepare_fresh_avatar()
        else:
            if not os.path.exists(self.avatar_path):
                raise FileNotFoundError(
                    f"Avatar {self.avatar_id} does not exist. Set preparation=True to create it."
                )
            self._validate_cached_avatar()
            self._load_cached_material()

    def _prepare_fresh_avatar(self) -> None:
        print("*********************************")
        print(f"  creating avatar: {self.avatar_id}")
        print("*********************************")
        osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
        self._prepare_material()

    def _validate_cached_avatar(self) -> None:
        if not os.path.exists(self.avatar_info_path):
            raise FileNotFoundError(f"Missing avatar metadata at {self.avatar_info_path}")
        with open(self.avatar_info_path, "r") as f:
            avatar_info = json.load(f)
        expected_bbox_shift = self.avatar_info["bbox_shift"]
        if avatar_info.get("bbox_shift") != expected_bbox_shift:
            if self.force_recreate:
                shutil.rmtree(self.avatar_path)
                self._prepare_fresh_avatar()
            elif self.interactive:
                response = input("bbox_shift changed; re-create avatar? (y/n)")
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    self._prepare_fresh_avatar()
                else:
                    raise RuntimeError("Avatar configuration mismatch; aborting.")
            else:
                raise RuntimeError("Avatar configuration mismatch; set force_recreate=True to rebuild.")
        self.avatar_info = avatar_info
        if not self.video_path:
            self.video_path = avatar_info.get("video_path")

    def _load_cached_material(self) -> None:
        if not os.path.exists(self.latents_out_path):
            raise FileNotFoundError(f"Latents not found at {self.latents_out_path}")
        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        with open(self.coords_path, "rb") as f:
            self.coord_list_cycle = pickle.load(f)
        input_img_list = glob.glob(os.path.join(self.full_imgs_path, "*.[jpJP][pnPN]*[gG]"))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = self._read_image_sequence(input_img_list)
        with open(self.mask_coords_path, "rb") as f:
            self.mask_coords_list_cycle = pickle.load(f)
        input_mask_list = glob.glob(os.path.join(self.mask_out_path, "*.[jpJP][pnPN]*[gG]"))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = self._read_image_sequence(input_mask_list)

    def _prepare_material(self) -> None:
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if not self.video_path:
            raise ValueError("video_path must be provided when preparing avatar assets.")

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext=".png")
        else:
            raise ValueError(f"Video file not found: {self.video_path}")

        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, "*.[jpJP][pnPN]*[gG]")))

        print("extracting landmarks...")
        # Import here to avoid pulling heavy dependencies when only loading avatars.
        from musetalk.utils.preprocessing import get_landmark_and_bbox

        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)

        input_latent_list = []
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if self.config.version == "v15":
                y2 = min(y2 + self.config.extra_margin, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = self.runtime.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)
            x1, y1, x2, y2 = self.coord_list_cycle[i]
            if self.config.version == "v15":
                mode = self.config.parsing_mode
            else:
                mode = "raw"
            mask, crop_box = get_image_prepare_material(
                frame,
                [x1, y1, x2, y2],
                fp=self.runtime.face_parser,
                mode=mode,
            )
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle.append(crop_box)
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, "wb") as f:
            pickle.dump(self.mask_coords_list_cycle, f)
        with open(self.coords_path, "wb") as f:
            pickle.dump(self.coord_list_cycle, f)
        torch.save(self.input_latent_list_cycle, self.latents_out_path)

    # ---------------------------------------------------------------------
    # Inference helpers
    # ---------------------------------------------------------------------
    def process_frames(self, res_frame_queue: "queue.Queue", video_len: int, skip_save_images: bool) -> None:
        while True:
            if self.idx >= video_len - 1:
                break
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            bbox = self.coord_list_cycle[self.idx % len(self.coord_list_cycle)]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % len(self.frame_list_cycle)])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception:
                continue
            mask = self.mask_list_cycle[self.idx % len(self.mask_list_cycle)]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % len(self.mask_coords_list_cycle)]
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            if skip_save_images is False:
                cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png", combine_frame)
            self.idx += 1

    @torch.no_grad()
    def inference(
        self,
        audio_path: str,
        out_vid_name: Optional[str],
        fps: Optional[int] = None,
        skip_save_images: Optional[bool] = None,
    ) -> None:
        if fps is None:
            fps = self.config.fps
        if skip_save_images is None:
            skip_save_images = self.config.skip_save_images

        os.makedirs(os.path.join(self.avatar_path, "tmp"), exist_ok=True)
        print("start inference")

        runtime = self.runtime
        config = self.config

        start_time = time.time()
        whisper_input_features, librosa_length = runtime.audio_processor.get_audio_feature(
            audio_path,
            weight_dtype=runtime.weight_dtype,
        )
        whisper_chunks = runtime.audio_processor.get_whisper_chunk(
            whisper_input_features,
            runtime.device,
            runtime.weight_dtype,
            runtime.whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=config.audio_padding_length_left,
            audio_padding_length_right=config.audio_padding_length_right,
        )
        print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000:.2f}ms")

        video_num = len(whisper_chunks)
        res_frame_queue: "queue.Queue" = queue.Queue()
        self.idx = 0
        process_thread = threading.Thread(
            target=self.process_frames,
            args=(res_frame_queue, video_num, skip_save_images),
        )
        process_thread.start()

        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        start_time = time.time()

        for whisper_batch, latent_batch in tqdm(
            gen, total=int(np.ceil(float(video_num) / self.batch_size))
        ):
            audio_feature_batch = runtime.pe(whisper_batch.to(runtime.device))
            latent_batch = latent_batch.to(device=runtime.device, dtype=runtime.unet.model.dtype)
            pred_latents = runtime.unet.model(
                latent_batch,
                runtime.timesteps,
                encoder_hidden_states=audio_feature_batch,
            ).sample
            pred_latents = pred_latents.to(device=runtime.device, dtype=runtime.vae.vae.dtype)
            recon = runtime.vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)

        process_thread.join()

        if skip_save_images:
            print(
                "Total process time of {} frames without saving images = {}s".format(
                    video_num, time.time() - start_time
                )
            )
        else:
            print(
                "Total process time of {} frames including saving images = {}s".format(
                    video_num, time.time() - start_time
                )
            )

        if out_vid_name is not None and skip_save_images is False:
            tmp_video_path = os.path.join(self.avatar_path, "temp.mp4")
            cmd_img2video = (
                f"ffmpeg -y -v warning -r {fps} -f image2 -i {self.avatar_path}/tmp/%08d.png "
                f"-vcodec libx264 -vf format=yuv420p -crf 18 {tmp_video_path}"
            )
            print(cmd_img2video)
            os.system(cmd_img2video)

            output_vid = os.path.join(self.video_out_path, f"{out_vid_name}.mp4")
            os.makedirs(self.video_out_path, exist_ok=True)
            cmd_combine_audio = (
                f"ffmpeg -y -v warning -i {audio_path} -i {tmp_video_path} {output_vid}"
            )
            print(cmd_combine_audio)
            os.system(cmd_combine_audio)

            os.remove(tmp_video_path)
            shutil.rmtree(os.path.join(self.avatar_path, "tmp"))
            print(f"result is save to {output_vid}")
        print("\n")

    # ------------------------------------------------------------------
    # Accessors for streaming use-cases
    # ------------------------------------------------------------------
    def get_frame_data(self, frame_idx: int) -> Dict[str, Any]:
        if self.frame_list_cycle is None or self.coord_list_cycle is None:
            raise RuntimeError("Avatar data not prepared")
        idx = frame_idx % len(self.frame_list_cycle)
        return {
            "latent": self.input_latent_list_cycle[idx],
            "bbox": self.coord_list_cycle[idx],
            "ori_frame": copy.deepcopy(self.frame_list_cycle[idx]),
            "mask": self.mask_list_cycle[idx],
            "mask_coords": self.mask_coords_list_cycle[idx],
        }

    def prepared_length(self) -> int:
        if self.frame_list_cycle is None:
            return 0
        return len(self.frame_list_cycle)


__all__ = ["Avatar", "AvatarConfig", "AvatarRuntime", "video2imgs", "osmakedirs"]
