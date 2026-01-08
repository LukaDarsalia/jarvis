"""
MuseTalk Triton Model - Stateless Lip-Sync Video Generation
Accepts complete audio and generates lip-synced video frames.
"""

import json
import os
import sys
import copy
import glob
import pickle
import traceback
import tempfile
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import torch
import librosa
import soundfile as sf
import triton_python_backend_utils as pb_utils

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import WhisperModel

# Import musetalk components
from audio_processor import AudioProcessor
from vae import VAE
from unet import UNet, PositionalEncoding
from blending import get_image_blending
from face_parsing import FaceParsing


@dataclass
class MuseTalkConfig:
    """Configuration for MuseTalk model"""
    avatar_root: str = "/local_models/musetalk_model/testing_avatar_creation/v15/avatars"
    default_avatar_id: str = "default"
    whisper_dir: str = "/local_models/musetalk_model/whisper"
    unet_config: str = "/local_models/musetalk_model/musetalkV15/musetalk.json"
    unet_model_path: str = "/local_models/musetalk_model/musetalkV15/unet.pth"
    vae_model_path: str = "/local_models/musetalk_model/sd-vae"
    face_parse_resnet_path: str = "/local_models/musetalk_model/face-parse-bisent/resnet18-5c106cde.pth"
    face_parse_model_path: str = "/local_models/musetalk_model/face-parse-bisent/79999_iter.pth"
    unet_onnx_path: str = "/local_models/musetalk_model/musetalkV15/unet.onnx"
    use_onnx_unet: bool = True
    fps: int = 25
    batch_size: int = 4
    audio_padding_length_left: int = 2
    audio_padding_length_right: int = 2
    input_sample_rate: int = 24000
    whisper_sample_rate: int = 16000
    max_audio_duration_s: float = 30.0  # Maximum audio duration (crop from left if exceeded)
    device: str = "cuda"


class TritonPythonModel:
    """Triton Python Backend Model for MuseTalk lip-sync generation (Stateless)"""
    
    def initialize(self, args):
        """Initialize the model and load avatar"""
        try:
            pb_utils.Logger.log_info("=== INITIALIZING MUSETALK (STATELESS DECOUPLED) ===")
            self.model_config = json.loads(args['model_config'])
            params = self.model_config.get('parameters', {})
            
            # Load config from parameters
            self.config = MuseTalkConfig()
            self.config.avatar_root = self._get_param(params, 'avatar_root', self.config.avatar_root)
            self.config.default_avatar_id = self._get_param(params, 'default_avatar_id', self.config.default_avatar_id)
            self.config.whisper_dir = self._get_param(params, 'whisper_dir', self.config.whisper_dir)
            self.config.unet_config = self._get_param(params, 'unet_config', self.config.unet_config)
            self.config.unet_model_path = self._get_param(params, 'unet_model_path', self.config.unet_model_path)
            self.config.vae_model_path = self._get_param(params, 'vae_model_path', self.config.vae_model_path)
            self.config.face_parse_resnet_path = self._get_param(params, 'face_parse_resnet_path', self.config.face_parse_resnet_path)
            self.config.face_parse_model_path = self._get_param(params, 'face_parse_model_path', self.config.face_parse_model_path)
            self.config.unet_onnx_path = self._get_param(params, 'unet_onnx_path', self.config.unet_onnx_path)
            self.config.use_onnx_unet = self._get_bool_param(params, 'use_onnx_unet', self.config.use_onnx_unet)
            self.config.fps = int(self._get_param(params, 'fps', str(self.config.fps)))
            self.config.batch_size = int(self._get_param(params, 'batch_size', str(self.config.batch_size)))
            self.config.audio_padding_length_left = int(self._get_param(params, 'audio_padding_left', str(self.config.audio_padding_length_left)))
            self.config.audio_padding_length_right = int(self._get_param(params, 'audio_padding_right', str(self.config.audio_padding_length_right)))
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            pb_utils.Logger.log_info(f"Device: {self.config.device}")
            pb_utils.Logger.log_info(f"Avatar root: {self.config.avatar_root}")
            pb_utils.Logger.log_info(f"FPS: {self.config.fps}")
            pb_utils.Logger.log_info(f"Batch size: {self.config.batch_size}")
            pb_utils.Logger.log_info(f"Use ONNX UNet: {self.config.use_onnx_unet}")
            
            # Load models
            self._load_models()
            
            # Load default avatar at initialization
            pb_utils.Logger.log_info(f"Loading default avatar '{self.config.default_avatar_id}'...")
            self.avatar_data = self._load_avatar(self.config.default_avatar_id)
            self.num_avatar_frames = len(self.avatar_data['frame_list_cycle'])
            pb_utils.Logger.log_info(f"Avatar loaded with {self.num_avatar_frames} frames")
            
            pb_utils.Logger.log_info("MuseTalk Triton model initialized successfully (stateless)")
            
        except Exception as e:
            msg = f"Failed to initialize MuseTalk: {e}\n{traceback.format_exc()}"
            pb_utils.Logger.log_error(msg)
            raise
    
    def _get_param(self, params, key, default):
        """Get parameter value from config"""
        if key in params:
            value = params[key]['string_value']
        else:
            value = default
        pb_utils.Logger.log_info(f"Parameter {key}: {value}")
        return value

    def _get_bool_param(self, params, key, default: bool) -> bool:
        value = self._get_param(params, key, str(default))
        return str(value).lower() in {"1", "true", "yes", "y", "on"}
    
    def _load_models(self):
        """Load all required models"""
        device = torch.device(self.config.device)
        
        pb_utils.Logger.log_info(f"Loading VAE model from {self.config.vae_model_path}...")
        self.vae = VAE(model_path=self.config.vae_model_path, use_float16=True)
        
        pb_utils.Logger.log_info(f"Loading UNet model from {self.config.unet_model_path}...")
        self.unet = UNet(
            unet_config=self.config.unet_config,
            model_path=self.config.unet_model_path,
            device=device,
            use_float16=True
        )
        
        pb_utils.Logger.log_info("Loading Positional Encoding...")
        self.pe = PositionalEncoding(d_model=384)
        
        pb_utils.Logger.log_info(f"Loading Whisper model from {self.config.whisper_dir}...")
        self.whisper = WhisperModel.from_pretrained(self.config.whisper_dir)
        
        # Convert to half precision and move to device
        self.pe = self.pe.half()
        # self.vae.vae = self.vae.vae.half()
        # self.unet.model = self.unet.model.half()
        self.whisper = self.whisper.to(device=device, dtype=torch.float16).eval()
        self.whisper.requires_grad_(False)
        
        self.weight_dtype = self.unet.model.dtype
        self.device = device
        self.timesteps = torch.tensor([0], device=device)
        self.unet_onnx_session = None
        self.unet_onnx_dtype = None
        self.unet_onnx_timestep = None

        if self.config.use_onnx_unet:
            try:
                import onnxruntime as ort

                available = ort.get_available_providers()
                providers = []
                if "CUDAExecutionProvider" in available and self.config.device == "cuda":
                    providers.append("CUDAExecutionProvider")
                providers.append("CPUExecutionProvider")

                pb_utils.Logger.log_info(
                    f"Loading UNet ONNX from {self.config.unet_onnx_path} with providers {providers}"
                )
                self.unet_onnx_session = ort.InferenceSession(
                    self.config.unet_onnx_path,
                    providers=providers,
                )
                input_type = self.unet_onnx_session.get_inputs()[0].type
                if "float16" in input_type:
                    self.unet_onnx_dtype = np.float16
                else:
                    self.unet_onnx_dtype = np.float32
                self.unet_onnx_timestep = np.array([0], dtype=np.int64)
            except Exception as exc:
                pb_utils.Logger.log_error(f"Failed to initialize UNet ONNX: {exc}")
                self.unet_onnx_session = None
                if self.config.use_onnx_unet:
                    raise
        
        pb_utils.Logger.log_info("Loading Audio Processor...")
        self.audio_processor = AudioProcessor(feature_extractor_path=self.config.whisper_dir)
        
        pb_utils.Logger.log_info(f"Loading Face Parser from {self.config.face_parse_model_path}...")
        self.face_parser = FaceParsing(
            resnet_path=self.config.face_parse_resnet_path,
            model_pth=self.config.face_parse_model_path
        )
        
        pb_utils.Logger.log_info("All models loaded successfully")
    
    def _load_avatar(self, avatar_id: str) -> Dict[str, Any]:
        """Load avatar assets from disk"""
        avatar_path = os.path.join(self.config.avatar_root, avatar_id)
        
        if not os.path.exists(avatar_path):
            raise FileNotFoundError(f"Avatar not found: {avatar_path}")
        
        pb_utils.Logger.log_info(f"Loading avatar assets from {avatar_path}")
        
        # Load latents
        latents_path = os.path.join(avatar_path, "latents.pt")
        if not os.path.exists(latents_path):
            raise FileNotFoundError(f"Latents not found: {latents_path}")
        input_latent_list_cycle = torch.load(latents_path)
        
        # Load coordinates
        coords_path = os.path.join(avatar_path, "coords.pkl")
        with open(coords_path, "rb") as f:
            coord_list_cycle = pickle.load(f)
        
        # Load mask coordinates
        mask_coords_path = os.path.join(avatar_path, "mask_coords.pkl")
        with open(mask_coords_path, "rb") as f:
            mask_coords_list_cycle = pickle.load(f)
        
        # Load frames
        full_imgs_path = os.path.join(avatar_path, "full_imgs")
        input_img_list = glob.glob(os.path.join(full_imgs_path, "*.[jpJP][pnPN]*[gG]"))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        frame_list_cycle = []
        for img_path in input_img_list:
            frame = cv2.imread(img_path)
            if frame is not None:
                frame_list_cycle.append(frame)
        
        # Load masks
        mask_path = os.path.join(avatar_path, "mask")
        input_mask_list = glob.glob(os.path.join(mask_path, "*.[jpJP][pnPN]*[gG]"))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        mask_list_cycle = []
        for mask_img_path in input_mask_list:
            mask = cv2.imread(mask_img_path)
            if mask is not None:
                mask_list_cycle.append(mask)
        
        pb_utils.Logger.log_info(f"Avatar loaded: {len(frame_list_cycle)} frames, {len(mask_list_cycle)} masks")
        
        if len(frame_list_cycle) == 0:
            raise FileNotFoundError(f"No frames found in {full_imgs_path}")
        
        if len(mask_list_cycle) == 0:
            raise FileNotFoundError(f"No masks found in {mask_path}")
        
        return {
            'input_latent_list_cycle': input_latent_list_cycle,
            'coord_list_cycle': coord_list_cycle,
            'frame_list_cycle': frame_list_cycle,
            'mask_list_cycle': mask_list_cycle,
            'mask_coords_list_cycle': mask_coords_list_cycle,
        }
    
    def _get_int_input(self, request, name: str, default: Optional[int] = None) -> Optional[int]:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        arr = tensor.as_numpy()
        if arr.size == 0:
            return default
        return int(arr.squeeze())
    
    def _get_audio_input(self, request) -> Optional[np.ndarray]:
        tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO")
        if tensor is None:
            return None
        arr = tensor.as_numpy()
        if arr.size == 0:
            return None
        return arr.astype(np.float32)

    def _run_unet_onnx(
        self,
        latent_batch: torch.Tensor,
        audio_feature_batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.unet_onnx_session is None:
            raise RuntimeError("UNet ONNX session is not initialized")
        sample = latent_batch.detach().cpu().numpy()
        encoder = audio_feature_batch.detach().cpu().numpy()
        if self.unet_onnx_dtype is not None:
            sample = sample.astype(self.unet_onnx_dtype)
            encoder = encoder.astype(self.unet_onnx_dtype)
        outputs = self.unet_onnx_session.run(
            None,
            {
                "sample": sample,
                "timestep": self.unet_onnx_timestep,
                "encoder_hidden_states": encoder,
            },
        )
        return torch.from_numpy(outputs[0]).to(self.device)
    
    def _send_frame(self, sender, frame_data: np.ndarray, frame_index: int, timestamp_ms: float, is_final: bool = False):
        """Send a video frame response"""
        # Encode frame as JPEG for efficient transmission
        _, jpeg_data = cv2.imencode('.jpg', frame_data, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpeg_bytes = jpeg_data.tobytes()
        
        frame_tensor = pb_utils.Tensor("VIDEO_FRAME", np.frombuffer(jpeg_bytes, dtype=np.uint8))
        index_tensor = pb_utils.Tensor("FRAME_INDEX", np.array([frame_index], dtype=np.int32))
        timestamp_tensor = pb_utils.Tensor("TIMESTAMP_MS", np.array([timestamp_ms], dtype=np.float32))
        
        response = pb_utils.InferenceResponse(output_tensors=[frame_tensor, index_tensor, timestamp_tensor])
        
        flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL if is_final else 0
        sender.send(response, flags=flags)
    
    def _send_final(self, sender):
        """Send empty final response"""
        frame_tensor = pb_utils.Tensor("VIDEO_FRAME", np.zeros(0, dtype=np.uint8))
        index_tensor = pb_utils.Tensor("FRAME_INDEX", np.array([-1], dtype=np.int32))
        timestamp_tensor = pb_utils.Tensor("TIMESTAMP_MS", np.array([0.0], dtype=np.float32))
        
        response = pb_utils.InferenceResponse(output_tensors=[frame_tensor, index_tensor, timestamp_tensor])
        sender.send(response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
    
    def _resample_audio(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio from input rate to whisper rate"""
        if self.config.input_sample_rate == self.config.whisper_sample_rate:
            return audio.astype(np.float32)
        
        resampled = librosa.resample(
            audio,
            orig_sr=self.config.input_sample_rate,
            target_sr=self.config.whisper_sample_rate
        )
        return resampled.astype(np.float32)
    
    def _process_audio_to_whisper(self, audio: np.ndarray) -> Optional[torch.Tensor]:
        """Process audio through Whisper and return whisper chunks"""
        # Crop audio from left if too long (keep most recent audio)
        max_samples = int(self.config.max_audio_duration_s * self.config.input_sample_rate)
        if len(audio) > max_samples:
            pb_utils.Logger.log_info(
                f"Audio too long ({len(audio)/self.config.input_sample_rate:.2f}s), "
                f"cropping to {self.config.max_audio_duration_s}s"
            )
            audio = audio[-max_samples:]
        
        # Create temp file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
            # Resample and save
            resampled = self._resample_audio(audio)
            sf.write(temp_file.name, resampled, self.config.whisper_sample_rate)
            
            try:
                # Get audio features using existing AudioProcessor
                whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(
                    temp_file.name,
                    weight_dtype=self.weight_dtype
                )
                
                if not whisper_input_features:
                    return None
                
                # Get whisper chunks
                whisper_chunks = self.audio_processor.get_whisper_chunk(
                    whisper_input_features=whisper_input_features,
                    device=self.device,
                    weight_dtype=self.weight_dtype,
                    whisper=self.whisper,
                    librosa_length=librosa_length,
                    fps=self.config.fps,
                    audio_padding_length_left=self.config.audio_padding_length_left,
                    audio_padding_length_right=self.config.audio_padding_length_right
                )
                
                return whisper_chunks
                
            except Exception as e:
                pb_utils.Logger.log_error(f"Whisper processing failed: {e}\n{traceback.format_exc()}")
                return None
    
    @torch.no_grad()
    def _generate_frames(self, whisper_chunks: torch.Tensor, start_frame_index: int) -> List[tuple]:
        """
        Generate video frames from whisper features.
        
        Args:
            whisper_chunks: Whisper audio features [num_frames, ...]
            start_frame_index: Starting frame index in avatar cycle (for video continuity)
        
        Returns:
            List of (frame_data, frame_index, timestamp_ms) tuples
        """
        num_frames = len(whisper_chunks)
        if num_frames == 0:
            return []
        
        results = []
        
        # Process in batches
        for batch_start in range(0, num_frames, self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, num_frames)
            
            # Get whisper features for batch
            whisper_batch = whisper_chunks[batch_start:batch_end]
            
            # Get latents for batch (use start_frame_index for avatar cycle position)
            latent_batch = []
            for i in range(batch_start, batch_end):
                # Cycle through avatar frames starting from start_frame_index
                cycle_idx = (start_frame_index + i) % self.num_avatar_frames
                latent_batch.append(self.avatar_data['input_latent_list_cycle'][cycle_idx])
            latent_batch = torch.cat(latent_batch, dim=0)
            
            # Forward pass through models
            audio_feature_batch = self.pe(whisper_batch.to(self.device))
            latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)

            if self.unet_onnx_session is not None:
                pred_latents = self._run_unet_onnx(latent_batch, audio_feature_batch)
            else:
                pred_latents = self.unet.model(
                    latent_batch,
                    self.timesteps,
                    encoder_hidden_states=audio_feature_batch,
                ).sample
            
            pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
            recon = self.vae.decode_latents(pred_latents)
            
            # Blend each frame
            for i, res_frame in enumerate(recon):
                frame_num = batch_start + i
                cycle_idx = (start_frame_index + frame_num) % self.num_avatar_frames
                
                frame_data = {
                    'bbox': self.avatar_data['coord_list_cycle'][cycle_idx],
                    'ori_frame': copy.deepcopy(self.avatar_data['frame_list_cycle'][cycle_idx]),
                    'mask': self.avatar_data['mask_list_cycle'][cycle_idx],
                    'mask_coords': self.avatar_data['mask_coords_list_cycle'][cycle_idx],
                }
                
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
                
                # Output frame index continues from start_frame_index
                output_frame_idx = start_frame_index + frame_num
                timestamp_ms = frame_num * (1000.0 / self.config.fps)
                
                results.append((combine_frame, output_frame_idx, timestamp_ms))
        
        return results
    
    def execute(self, requests):
        """Process inference requests"""
        for request in requests:
            sender = request.get_response_sender()
            self._handle_request(request, sender)
    
    def _handle_request(self, request, sender):
        """Handle a single request (stateless)"""
        try:
            # Get inputs
            audio = self._get_audio_input(request)
            frame_index = self._get_int_input(request, "FRAME_INDEX", 0)
            
            if audio is None or len(audio) == 0:
                pb_utils.Logger.log_error("AUDIO input is required and must not be empty")
                self._send_final(sender)
                return
            
            audio_duration_s = len(audio) / self.config.input_sample_rate
            pb_utils.Logger.log_info(f"Processing audio: {audio_duration_s:.3f}s, start_frame_index: {frame_index}")
            
            # Process audio through Whisper
            whisper_chunks = self._process_audio_to_whisper(audio)
            
            if whisper_chunks is None or len(whisper_chunks) == 0:
                pb_utils.Logger.log_error("Failed to process audio through Whisper")
                self._send_final(sender)
                return
            
            pb_utils.Logger.log_info(f"Generated {len(whisper_chunks)} whisper chunks")
            
            # Generate all frames
            frames = self._generate_frames(whisper_chunks, frame_index)
            
            if len(frames) == 0:
                pb_utils.Logger.log_warning("No frames generated")
                self._send_final(sender)
                return
            
            pb_utils.Logger.log_info(f"Sending {len(frames)} frames")
            
            # Send all frames
            for i, (frame_data, frame_idx, timestamp_ms) in enumerate(frames):
                is_last = (i == len(frames) - 1)
                self._send_frame(sender, frame_data, frame_idx, timestamp_ms, is_final=is_last)
        
        except Exception as e:
            pb_utils.Logger.log_error(f"Error processing request: {e}\n{traceback.format_exc()}")
            self._send_final(sender)
    
    def finalize(self):
        """Cleanup on model unload"""
        try:
            pb_utils.Logger.log_info("Finalizing MuseTalk model...")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            pb_utils.Logger.log_error(f"Error in finalize: {e}")
