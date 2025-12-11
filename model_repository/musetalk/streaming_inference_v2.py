"""
Clean Streaming Inference for MuseTalk
Uses abstract audio streaming interface - no prior knowledge of audio length
"""

import argparse
import os
import sys
import threading
import time
import queue
from typing import Optional
import numpy as np
import torch
import cv2
from tqdm import tqdm
from transformers import WhisperModel

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_stream import AudioStream, PseudoAudioStream, LiveAudioStream
from avatar import Avatar, AvatarConfig, AvatarRuntime
from whisper_processor import WhisperStreamProcessor

from audio_processor import AudioProcessor
from face_parsing import FaceParsing
from blending import get_image_blending
from utils import load_all_model


class AudioPlaybackController:
    """Chunk-aware audio playback synchronized with generated video frames."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._queue: "queue.Queue[np.ndarray | None]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stream = None
        self._lock = threading.Lock()
        self._enqueued_samples = 0
        self._played_samples = 0
        self.enabled = False

    def start(self) -> bool:
        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:
            print(f"sounddevice playback unavailable ({exc})")
            return False

        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=0,
            )
            self._stream.start()
        except Exception as exc:
            print(f"Failed to open sounddevice stream ({exc})")
            return False

        def worker() -> None:
            while not self._stop_event.is_set():
                try:
                    chunk = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if chunk is None:
                    break
                try:
                    self._stream.write(chunk)
                    with self._lock:
                        self._played_samples += chunk.shape[0]
                except Exception as exc:
                    print(f"Audio playback stream error: {exc}")
                    break

        self._thread = threading.Thread(target=worker, name="audio-playback", daemon=True)
        self._thread.start()
        self.enabled = True
        print("Audio playback stream opened (chunk-synced).")
        return True

    def enqueue_chunk(self, chunk: np.ndarray) -> None:
        if not self.enabled:
            return
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(-1, 1)
        elif chunk.ndim > 1 and chunk.shape[1] != 1:
            chunk = np.mean(chunk, axis=1, keepdims=True)
        frames = chunk.shape[0]
        with self._lock:
            self._enqueued_samples += frames
        self._queue.put(chunk)

    @property
    def scheduled_samples(self) -> int:
        if not self.enabled:
            return 0
        with self._lock:
            return self._enqueued_samples

    @property
    def played_samples(self) -> int:
        if not self.enabled:
            return 0
        with self._lock:
            return self._played_samples

    def wait_until_idle(self, timeout: float = 5.0) -> None:
        if not self.enabled:
            return
        end_time = time.time() + timeout
        while time.time() < end_time:
            with self._lock:
                remaining = self._enqueued_samples - self._played_samples
            if remaining <= 0 and self._queue.empty():
                return
            time.sleep(0.05)

    def stop(self) -> None:
        if not self.enabled:
            return
        self.wait_until_idle(timeout=10.0)
        self._stop_event.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                print(f"Audio stream closure error: {exc}")
        self.enabled = False


def flush_audio_chunks(
    controller: Optional[AudioPlaybackController],
    pending_chunks: list[np.ndarray],
    allowed_time_s: Optional[float],
) -> None:
    if controller is None or not controller.enabled:
        return

    if allowed_time_s is None:
        allowed_samples = float('inf')
    else:
        allowed_samples = max(0.0, allowed_time_s) * controller.sample_rate

    while pending_chunks:
        chunk = pending_chunks[0]
        chunk_samples = len(chunk)
        if controller.scheduled_samples + chunk_samples > allowed_samples:
            break
        controller.enqueue_chunk(chunk)
        pending_chunks.pop(0)


def ensure_avatar_assets(avatar_path: str) -> None:
    required_files = [
        os.path.join(avatar_path, "latents.pt"),
        os.path.join(avatar_path, "coords.pkl"),
        os.path.join(avatar_path, "mask_coords.pkl"),
    ]
    required_dirs = [
        os.path.join(avatar_path, "full_imgs"),
        os.path.join(avatar_path, "mask"),
    ]

    missing = [p for p in required_files if not os.path.isfile(p)]
    missing.extend([d for d in required_dirs if not os.path.isdir(d)])

    if missing:
        report = "\n  ".join(missing)
        raise FileNotFoundError(
            f"Avatar assets are incomplete at '{avatar_path}':\n  {report}\n"
            "Create them using scripts/create_avatar.py in avatar_venv."
        )


@torch.no_grad()
def streaming_inference(args):
    """
    Main streaming inference function.
    Processes audio chunks in real-time without prior knowledge of duration.
    """
    print("\n" + "="*80)
    print("MUSETALK STREAMING INFERENCE")
    print("="*80)
    print(f"Audio: {args.audio_path}")
    print(f"Output: {args.output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Video FPS: {args.fps}")
    print(f"Lookahead chunks: {args.lookahead_chunks}")
    print("="*80 + "\n")

    enforced_chunk_ms = 80
    if args.chunk_duration_ms != enforced_chunk_ms:
        print(f"Overriding chunk duration to {enforced_chunk_ms}ms (requested {args.chunk_duration_ms}ms)")
    args.chunk_duration_ms = enforced_chunk_ms

    # Setup device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load models
    print("Loading models...")
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)

    # Convert to half precision
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    weight_dtype = unet.model.dtype

    print("Models loaded\n")

    # Load Whisper model (using EXISTING approach from inference.py)
    print("Loading Whisper model...")
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    print("Whisper loaded\n")

    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)

    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width,
        )
    else:
        fp = FaceParsing()

    avatar_runtime = AvatarRuntime(
        device=device,
        vae=vae,
        unet=unet,
        pe=pe,
        audio_processor=audio_processor,
        whisper=whisper,
        face_parser=fp,
        timesteps=timesteps,
        weight_dtype=weight_dtype,
    )

    avatar_id = args.avatar_id
    if avatar_id is None:
        raise ValueError("--avatar_id is required when running streaming inference.")

    if args.avatar_root:
        avatar_base_path = os.path.join(args.avatar_root, avatar_id)
    else:
        if args.version == "v15":
            avatar_base_path = os.path.join(args.result_dir, args.version, "avatars", avatar_id)
        else:
            avatar_base_path = os.path.join(args.result_dir, "avatars", avatar_id)

    print(f"Avatar ID: {avatar_id}")
    print(f"Avatar cache: {avatar_base_path}")

    avatar_config = AvatarConfig(
        version=args.version,
        result_root=args.result_dir,
        extra_margin=args.extra_margin,
        parsing_mode=args.parsing_mode,
        skip_save_images=True,
        fps=args.fps,
        batch_size=args.batch_size,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
    )

    bbox_shift = 0 if args.version == "v15" else args.bbox_shift

    ensure_avatar_assets(avatar_base_path)

    avatar = Avatar(
        avatar_id=avatar_id,
        video_path=None,
        bbox_shift=bbox_shift,
        preparation=False,
        config=avatar_config,
        runtime=avatar_runtime,
        interactive=False,
        avatar_path_override=avatar_base_path,
    )
    print(f"Avatar '{avatar_id}' prepared with {avatar.prepared_length()} frames in cycle\n")

    # Initialize audio stream (no prior knowledge of length!)
    print("\n" + "="*80)
    print("INITIALIZING AUDIO STREAM")
    print("="*80)
    audio_stream = PseudoAudioStream(
        audio_path=args.audio_path,
        sample_rate=24000,  # Input at 24kHz
        chunk_duration_ms=args.chunk_duration_ms,
    )
    print(f"Stream initialized: 24kHz, {args.chunk_duration_ms}ms chunks")
    print(f"Samples per chunk: {audio_stream.samples_per_chunk}")
    print("Note: Total duration is NOT known - simulating real-time stream")
    print("="*80 + "\n")

    # Initialize Whisper processor (wraps EXISTING AudioProcessor!)
    print("Initializing Whisper stream processor...")
    whisper_processor = WhisperStreamProcessor(
        whisper_model_path=args.whisper_dir,
        whisper_model=whisper,  # Pass the loaded model instance
        input_sample_rate=24000,
        whisper_sample_rate=16000,
        device=device,
        dtype=weight_dtype,
        chunk_duration_s=args.chunk_duration_ms / 1000.0,
        lookahead_chunks=args.lookahead_chunks,
    )
    print(f"Processor initialized (lookahead={args.lookahead_chunks} chunk(s), base window=0.20s)\n")

    # Initialize display (headless-safe)
    try:
        cv2.namedWindow('Streaming Output', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Streaming Output', 1280, 720)
        display_enabled = True
    except:
        display_enabled = False
        print("Display not available - running in headless mode\n")

    playback_controller: Optional[AudioPlaybackController] = None
    playback_delay = whisper_processor.base_future_context_s + args.lookahead_chunks * (args.chunk_duration_ms / 1000.0)
    if args.no_audio_playback:
        print("Audio playback disabled via --no_audio_playback\n")
    else:
        playback_controller = AudioPlaybackController(sample_rate=audio_stream.sample_rate)
        if playback_controller.start():
            if playback_delay > 0:
                print(f"Audio playback will trail video by {playback_delay:.3f}s to respect lookahead window")
            print("Audio playback running in background (chunk-synced).\n")
        else:
            playback_controller = None
            print("Audio playback unavailable.\n")

    try:
        # Metrics
        total_frames_generated = 0
        total_inference_time = 0
        frame_times = []
        output_frames = []
        frame_idx = 0
        chunk_idx = 0
        pending_audio_chunks: list[np.ndarray] = []

        def consume_whisper_features(whisper_features, chunk_label):
            nonlocal frame_idx, total_frames_generated, total_inference_time, frame_times, output_frames

            if whisper_features is None or len(whisper_features) == 0:
                return

            num_frames = len(whisper_features)
            frame_base_index = frame_idx
            chunk_label_str = str(chunk_label)
            print(f"  emitting {num_frames} frame(s) [{frame_base_index}->{frame_base_index + num_frames - 1}] using chunk {chunk_label_str}")

            for batch_start in range(0, num_frames, args.batch_size):
                batch_end = min(batch_start + args.batch_size, num_frames)
                whisper_batch = whisper_features[batch_start:batch_end]

                latent_batch = []
                for i in range(batch_start, batch_end):
                    frame_data = avatar.get_frame_data(frame_base_index + i)
                    latent_batch.append(frame_data['latent'])
                latent_batch = torch.cat(latent_batch, dim=0)

                inference_start = time.time()

                audio_feature_batch = pe(whisper_batch.to(device))
                latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)

                pred_latents = unet.model(
                    latent_batch,
                    timesteps,
                    encoder_hidden_states=audio_feature_batch,
                ).sample

                pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
                recon = vae.decode_latents(pred_latents)

                inference_time = time.time() - inference_start
                total_inference_time += inference_time
                per_frame_time = inference_time / max(1, len(recon))
                frame_times.append(per_frame_time)

                for res_frame in recon:
                    frame_data = avatar.get_frame_data(frame_idx)

                    x1, y1, x2, y2 = frame_data['bbox']
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))

                    combine_frame = get_image_blending(
                        frame_data['ori_frame'],
                        res_frame,
                        frame_data['bbox'],
                        frame_data['mask'],
                        frame_data['mask_coords'],
                    )

                    if not isinstance(combine_frame, np.ndarray):
                        combine_frame = np.array(combine_frame)
                    combine_frame = np.ascontiguousarray(combine_frame, dtype=np.uint8)

                    current_fps = 1.0 / frame_times[-1] if frame_times and frame_times[-1] > 0 else 0.0
                    avg_fps = len(frame_times) / sum(frame_times) if frame_times and sum(frame_times) > 0 else 0.0

                    cv2.putText(combine_frame, f"Frame: {frame_idx}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(combine_frame, f"Current FPS: {current_fps:.2f}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(combine_frame, f"Avg FPS: {avg_fps:.2f}", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(combine_frame, f"Audio chunk: {chunk_label_str}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if display_enabled:
                        cv2.imshow('Streaming Output', combine_frame)
                        cv2.waitKey(1)

                    output_frames.append(combine_frame)
                    frame_idx += 1
                    total_frames_generated += 1

        print("="*80)
        print("STREAMING STARTED")
        print("="*80)
        print("Processing audio chunks as they arrive...\n")

        start_time = time.time()

        # Process audio stream chunk by chunk
        for audio_chunk in audio_stream:
            chunk_idx += 1

            audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
            pending_audio_chunks.append(audio_chunk)
            whisper_processor.process_chunk(audio_chunk)

            stats = whisper_processor.get_stats(video_fps=args.fps)
            chunk_elapsed = time.time() - start_time
            frames_ready = stats['frames_ready'] if stats['frames_ready'] is not None else 0
            allowed_frames = stats['allowed_frames'] if stats['allowed_frames'] is not None else 0
            available_frames = stats['available_frames']

            print(f"[chunk {chunk_idx:05d} | {chunk_elapsed:7.3f}s] audio={stats['audio_duration_s']:.3f}s "
                  f"buffered={stats['buffer_duration_s']:.3f}s available={available_frames} "
                  f"ready={frames_ready} allowed={allowed_frames} generated={total_frames_generated}")

            whisper_features = whisper_processor.get_features_for_frames(
                num_frames=None,
                video_fps=args.fps,
                audio_padding_left=args.audio_padding_length_left,
                audio_padding_right=args.audio_padding_length_right,
                lookahead_chunks=args.lookahead_chunks,
            )

            if whisper_features is None or len(whisper_features) == 0:
                wait_reason = "lookahead" if args.lookahead_chunks > 0 else "buffer"
                print(f"  waiting for {wait_reason} (lookahead_window={stats['lookahead_duration_s']:.3f}s, "
                      f"min_buffer_ready={stats['min_buffer_ready']}, available={available_frames})")
            else:
                consume_whisper_features(whisper_features, f"{chunk_idx:05d}")

            flush_audio_chunks(
                playback_controller,
                pending_audio_chunks,
                (frame_idx / args.fps) - playback_delay,
            )

        whisper_processor.mark_stream_complete()
        print("\nAudio stream complete - flushing remaining frames without lookahead...")
        while True:
            stats = whisper_processor.get_stats(video_fps=args.fps)
            whisper_features = whisper_processor.get_features_for_frames(
                num_frames=None,
                video_fps=args.fps,
                audio_padding_left=args.audio_padding_length_left,
                audio_padding_right=args.audio_padding_length_right,
                lookahead_chunks=0,
            )

            if whisper_features is None or len(whisper_features) == 0:
                remaining_ready = stats['frames_ready'] if stats['frames_ready'] is not None else 0
                print(f"  no additional frames ready (frames_ready={remaining_ready})")
                break

            consume_whisper_features(whisper_features, "flush")

        flush_audio_chunks(
            playback_controller,
            pending_audio_chunks,
            (frame_idx / args.fps) - playback_delay,
        )
        flush_audio_chunks(
            playback_controller,
            pending_audio_chunks,
            whisper_processor.total_audio_duration_s,
        )
        flush_audio_chunks(
            playback_controller,
            pending_audio_chunks,
            None,
        )

        total_time = time.time() - start_time

        # Close display
        if display_enabled:
            cv2.destroyAllWindows()

        # Print final statistics
        print("\n" + "="*80)
        print("STREAMING COMPLETE")
        print("="*80)
        print(f"Total time: {total_time:.2f}s")
        print(f"Audio chunks processed: {chunk_idx}")
        print(f"Video frames generated: {total_frames_generated}")
        print(f"Total inference time: {total_inference_time:.2f}s")
        print(f"Average FPS: {total_frames_generated / total_inference_time:.2f}")
        if frame_times:
            print(f"Min frame time: {min(frame_times)*1000:.2f}ms")
            print(f"Max frame time: {max(frame_times)*1000:.2f}ms")
            print(f"Avg frame time: {np.mean(frame_times)*1000:.2f}ms")
        print("="*80 + "\n")

        # Save output video
        if args.output_path and len(output_frames) > 0:
            print(f"Saving output video to: {args.output_path}")
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

            height, width = output_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(args.output_path, fourcc, args.fps, (width, height))

            for frame in tqdm(output_frames, desc="Writing video"):
                out.write(frame)
            out.release()

            # Add audio (using EXISTING approach from inference.py)
            output_with_audio = args.output_path.replace('.mp4', '_with_audio.mp4')
            cmd = f"ffmpeg -y -v warning -i {args.output_path} -i {args.audio_path} -c:v copy -c:a aac -shortest {output_with_audio}"
            os.system(cmd)
            print(f"Output with audio: {output_with_audio}\n")
    finally:
        if playback_controller is not None:
            playback_controller.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean streaming inference for MuseTalk")
    parser.add_argument("--audio_path", type=str, default="./data/audio/piradoba.wav")
    parser.add_argument("--output_path", type=str, default="./results/streaming_v2_output.mp4")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"])
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--audio_padding_length_left", type=int, default=2)
    parser.add_argument("--audio_padding_length_right", type=int, default=2)
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper")
    parser.add_argument("--unet_config", type=str, default="./models/musetalk/musetalk.json")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalk/pytorch_model.bin")
    parser.add_argument("--vae_type", type=str, default="sd-vae")
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--extra_margin", type=int, default=10)
    parser.add_argument("--parsing_mode", type=str, default="jaw")
    parser.add_argument("--left_cheek_width", type=int, default=90)
    parser.add_argument("--right_cheek_width", type=int, default=90)
    parser.add_argument("--avatar_id", type=str, required=True,
                        help="Identifier of the prebuilt avatar assets to load")
    parser.add_argument("--avatar_root", type=str, default=None,
                        help="Optional override root directory where avatar caches are stored")
    parser.add_argument("--no_audio_playback", action="store_true",
                        help="Disable local audio playback during streaming")
    parser.add_argument("--lookahead_chunks", type=int, default=0,
                        help="Number of future 80ms audio chunks to buffer before emitting frames")
    parser.add_argument("--chunk_duration_ms", type=int, default=80,
                        help="Audio stream chunk size in milliseconds (enforced to 80ms)")

    args = parser.parse_args()
    streaming_inference(args)
