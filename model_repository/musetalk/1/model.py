"""
MuseTalk Triton Model - Streaming Lip-Sync Video Generation
Accepts audio chunks and generates lip-synced video frames.
"""

import json
import os
import sys
import copy
import glob
import pickle
import traceback
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from collections import deque
import math

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
    fps: int = 25
    batch_size: int = 1
    audio_padding_length_left: int = 2
    audio_padding_length_right: int = 2
    input_sample_rate: int = 24000
    whisper_sample_rate: int = 16000
    chunk_duration_ms: int = 80
    device: str = "cuda"
    max_sessions: int = 4


@dataclass
class SessionState:
    """State for a single MuseTalk session"""
    session_id: int
    avatar_id: str
    
    # Audio buffering
    audio_buffer: deque = field(default_factory=deque)
    original_audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    total_audio_duration_s: float = 0.0
    
    # Whisper processing state
    whisper_input_features: Optional[Any] = None
    whisper_chunks: Optional[torch.Tensor] = None
    librosa_length: int = 0
    
    # Frame generation state
    frame_index: int = 0
    generated_frame_count: int = 0
    total_available_frames: int = 0
    
    # Avatar data (loaded once per session)
    input_latent_list_cycle: Optional[List] = None
    coord_list_cycle: Optional[List] = None
    frame_list_cycle: Optional[List] = None
    mask_list_cycle: Optional[List] = None
    mask_coords_list_cycle: Optional[List] = None
    
    # Temp file for audio processing
    temp_audio_file: Optional[Any] = None
    
    # Stream state
    stream_complete: bool = False
    is_initialized: bool = False


class TritonPythonModel:
    """Triton Python Backend Model for MuseTalk lip-sync generation"""
    
    def initialize(self, args):
        """Initialize the model"""
        try:
            pb_utils.Logger.log_info("=== INITIALIZING MUSETALK (DECOUPLED) ===")
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
            self.config.fps = int(self._get_param(params, 'fps', str(self.config.fps)))
            self.config.batch_size = int(self._get_param(params, 'batch_size', str(self.config.batch_size)))
            self.config.audio_padding_length_left = int(self._get_param(params, 'audio_padding_left', str(self.config.audio_padding_length_left)))
            self.config.audio_padding_length_right = int(self._get_param(params, 'audio_padding_right', str(self.config.audio_padding_length_right)))
            self.config.max_sessions = int(self._get_param(params, 'max_sessions', str(self.config.max_sessions)))
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            pb_utils.Logger.log_info(f"Device: {self.config.device}")
            pb_utils.Logger.log_info(f"Avatar root: {self.config.avatar_root}")
            pb_utils.Logger.log_info(f"FPS: {self.config.fps}")
            
            # Load models
            self._load_models()
            
            # Session management
            self.sessions: Dict[int, SessionState] = {}
            self.session_lock = None  # Will use simple dict access (Triton handles threading)
            # Track loop position per avatar so new sessions continue from previous frame offset
            self.avatar_cycle_index: Dict[str, int] = {}
            
            # Pre-compute constants
            self.samples_per_chunk = int(self.config.input_sample_rate * self.config.chunk_duration_ms / 1000)
            self.min_buffer_chunks = 3  # Minimum chunks before processing
            
            pb_utils.Logger.log_info("MuseTalk Triton model initialized successfully")
            
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
    
    def _load_models(self):
        """Load all required models"""
        device = torch.device(self.config.device)
        
        pb_utils.Logger.log_info(f"Loading VAE model from {self.config.vae_model_path}...")
        self.vae = VAE(
            model_path=self.config.vae_model_path,
        )
        
        pb_utils.Logger.log_info(f"Loading UNet model from {self.config.unet_model_path}...")
        self.unet = UNet(
            unet_config=self.config.unet_config,
            model_path=self.config.unet_model_path,
            device=device
        )
        
        pb_utils.Logger.log_info("Loading Positional Encoding...")
        self.pe = PositionalEncoding(d_model=384)
        
        pb_utils.Logger.log_info(f"Loading Whisper model from {self.config.whisper_dir}...")
        self.whisper = WhisperModel.from_pretrained(self.config.whisper_dir)
        
        # Convert to half precision and move to device
        self.pe = self.pe.half().to(device)
        self.vae.vae = self.vae.vae.half().to(device)
        self.unet.model = self.unet.model.half().to(device)
        self.whisper = self.whisper.to(device=device, dtype=torch.float16).eval()
        self.whisper.requires_grad_(False)
        
        self.weight_dtype = self.unet.model.dtype
        self.device = device
        self.timesteps = torch.tensor([0], device=device)
        
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
    
    def _get_bool_input(self, request, name: str, default: bool = False) -> bool:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        arr = tensor.as_numpy()
        if arr.size == 0:
            return default
        return bool(arr.squeeze())
    
    def _get_int_input(self, request, name: str, default: Optional[int] = None) -> Optional[int]:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        arr = tensor.as_numpy()
        if arr.size == 0:
            return default
        return int(arr.squeeze())
    
    def _get_string_input(self, request, name: str, default: str = "") -> str:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        arr = tensor.as_numpy()
        if arr.size == 0:
            return default
        val = arr[0]
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return str(val)
    
    def _get_audio_input(self, request) -> Optional[np.ndarray]:
        tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_CHUNK")
        if tensor is None:
            return None
        arr = tensor.as_numpy()
        if arr.size == 0:
            return None
        return arr.astype(np.float32)
    
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
    
    def _process_audio_to_whisper(self, session: SessionState) -> bool:
        """Process accumulated audio through Whisper"""
        if len(session.original_audio) < self.samples_per_chunk * self.min_buffer_chunks:
            return False
        
        # Create temp file if needed
        if session.temp_audio_file is None:
            session.temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        
        # Resample and save
        resampled = self._resample_audio(session.original_audio)
        sf.write(session.temp_audio_file.name, resampled, self.config.whisper_sample_rate)
        
        try:
            # Get audio features using existing AudioProcessor
            whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(
                session.temp_audio_file.name,
                weight_dtype=self.weight_dtype
            )
            
            if not whisper_input_features:
                return False
            
            session.whisper_input_features = whisper_input_features
            session.librosa_length = librosa_length
            
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
            
            session.whisper_chunks = whisper_chunks
            session.total_available_frames = len(whisper_chunks)
            
            return True
            
        except Exception as e:
            pb_utils.Logger.log_error(f"Whisper processing failed: {e}")
            return False
    
    @torch.no_grad()
    def _generate_frames(self, session: SessionState, num_frames: int) -> List[np.ndarray]:
        """Generate video frames from whisper features"""
        if session.whisper_chunks is None or session.total_available_frames == 0:
            return []
        
        frames = []
        start_idx = session.generated_frame_count
        end_idx = min(start_idx + num_frames, session.total_available_frames)
        
        if start_idx >= end_idx:
            return []
        
        # Process in batches
        for batch_start in range(start_idx, end_idx, self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, end_idx)
            
            # Get whisper features for batch
            whisper_batch = session.whisper_chunks[batch_start:batch_end]
            
            # Get latents for batch
            latent_batch = []
            for i in range(batch_start, batch_end):
                frame_idx = i % len(session.input_latent_list_cycle)
                latent_batch.append(session.input_latent_list_cycle[frame_idx])
            latent_batch = torch.cat(latent_batch, dim=0)
            
            # Forward pass through models
            audio_feature_batch = self.pe(whisper_batch.to(self.device))
            latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
            
            pred_latents = self.unet.model(
                latent_batch,
                self.timesteps,
                encoder_hidden_states=audio_feature_batch,
            ).sample
            
            pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
            recon = self.vae.decode_latents(pred_latents)
            
            # Blend each frame
            for i, res_frame in enumerate(recon):
                frame_idx = batch_start + i
                cycle_idx = frame_idx % len(session.frame_list_cycle)
                
                frame_data = {
                    'latent': session.input_latent_list_cycle[cycle_idx],
                    'bbox': session.coord_list_cycle[cycle_idx],
                    'ori_frame': copy.deepcopy(session.frame_list_cycle[cycle_idx]),
                    'mask': session.mask_list_cycle[cycle_idx],
                    'mask_coords': session.mask_coords_list_cycle[cycle_idx],
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
                
                frames.append(combine_frame)
        
        session.generated_frame_count = end_idx
        return frames
    
    def _initialize_session(self, session_id: int, avatar_id: str) -> bool:
        """Initialize a new session"""
        if len(self.sessions) >= self.config.max_sessions:
            pb_utils.Logger.log_error(f"Max sessions ({self.config.max_sessions}) reached")
            return False
        
        if session_id in self.sessions:
            pb_utils.Logger.log_warning(f"Session {session_id} already exists, reinitializing")
            self._cleanup_session(session_id)
        
        try:
            # Load avatar
            avatar_data = self._load_avatar(avatar_id)
            
            # Create session
            session = SessionState(session_id=session_id, avatar_id=avatar_id)
            session.input_latent_list_cycle = avatar_data['input_latent_list_cycle']
            session.coord_list_cycle = avatar_data['coord_list_cycle']
            session.frame_list_cycle = avatar_data['frame_list_cycle']
            session.mask_list_cycle = avatar_data['mask_list_cycle']
            session.mask_coords_list_cycle = avatar_data['mask_coords_list_cycle']
            session.total_available_frames = len(session.frame_list_cycle)
            # Continue avatar loop from last remembered index for this avatar
            start_idx = self.avatar_cycle_index.get(avatar_id, 0) % session.total_available_frames
            session.frame_index = start_idx
            session.is_initialized = True
            
            self.sessions[session_id] = session
            pb_utils.Logger.log_info(f"Session {session_id} initialized with avatar '{avatar_id}'")
            return True
            
        except Exception as e:
            pb_utils.Logger.log_error(f"Failed to initialize session {session_id}: {e}")
            return False
    
    def _cleanup_session(self, session_id: int):
        """Cleanup a session"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        # Persist loop position for this avatar so next session resumes from there
        try:
            if session.frame_list_cycle:
                self.avatar_cycle_index[session.avatar_id] = session.frame_index % len(session.frame_list_cycle)
        except Exception:
            pass
        
        # Clean up temp file
        if session.temp_audio_file is not None:
            try:
                os.unlink(session.temp_audio_file.name)
            except:
                pass
        
        del self.sessions[session_id]
        pb_utils.Logger.log_info(f"Session {session_id} cleaned up")
    
    def execute(self, requests):
        """Process inference requests"""
        for request in requests:
            sender = request.get_response_sender()
            self._handle_request(request, sender)
    
    def _handle_request(self, request, sender):
        """Handle a single request"""
        try:
            # Get inputs
            is_start = self._get_bool_input(request, "START", False)
            is_end = self._get_bool_input(request, "END", False)
            session_id = self._get_int_input(request, "CORRID", None)
            avatar_id = self._get_string_input(request, "AVATAR_ID", self.config.default_avatar_id)
            audio_chunk = self._get_audio_input(request)
            
            if session_id is None:
                pb_utils.Logger.log_error("CORRID is required")
                self._send_final(sender)
                return
            
            # Handle session start
            if is_start:
                pb_utils.Logger.log_info(f"Initializing session {session_id} with avatar '{avatar_id}'")
                success = self._initialize_session(session_id, avatar_id)
                if success:
                    # Send first frame as idle frame
                    session = self.sessions[session_id]
                    if session.frame_list_cycle:
                        start_idx = session.frame_index % len(session.frame_list_cycle)
                        pb_utils.Logger.log_info(
                            f"Sending idle frame for session {session_id} starting at index {start_idx} "
                            f"({len(session.frame_list_cycle)} frames available)"
                        )
                        timestamp = start_idx * (1000.0 / self.config.fps)
                        self._send_frame(sender, session.frame_list_cycle[start_idx], start_idx, timestamp, is_final=True)
                    else:
                        pb_utils.Logger.log_error(f"No frames available for session {session_id}")
                        self._send_final(sender)
                else:
                    pb_utils.Logger.log_error(f"Session {session_id} initialization failed")
                    self._send_final(sender)
                return
            
            # Handle session end
            if is_end:
                if session_id in self.sessions:
                    session = self.sessions[session_id]
                    session.stream_complete = True
                    
                    # Process any remaining audio
                    self._process_audio_to_whisper(session)
                    
                    # Generate remaining frames
                    remaining = session.total_available_frames - session.generated_frame_count
                    if remaining > 0:
                        frames = self._generate_frames(session, remaining)
                        for i, frame in enumerate(frames):
                            frame_idx = session.frame_index + i
                            timestamp = frame_idx * (1000.0 / self.config.fps)
                            is_last = (i == len(frames) - 1)
                            self._send_frame(sender, frame, frame_idx, timestamp, is_final=is_last)
                        session.frame_index += len(frames)
                    else:
                        self._send_final(sender)
                    
                    self._cleanup_session(session_id)
                else:
                    self._send_final(sender)
                return
            
            # Handle audio chunk
            if session_id not in self.sessions:
                pb_utils.Logger.log_error(f"Session {session_id} not found")
                self._send_final(sender)
                return
            
            session = self.sessions[session_id]
            
            if audio_chunk is not None and len(audio_chunk) > 0:
                # Buffer audio
                session.audio_buffer.append(audio_chunk)
                session.original_audio = np.concatenate([session.original_audio, audio_chunk])
                session.total_audio_duration_s = len(session.original_audio) / self.config.input_sample_rate
                
                # Process through whisper
                if self._process_audio_to_whisper(session):
                    # Calculate how many frames we can safely generate
                    # Need some lookahead for better quality
                    lookahead_s = 0.2  # 200ms lookahead
                    safe_duration_s = session.total_audio_duration_s - lookahead_s
                    
                    if safe_duration_s > 0:
                        allowed_frames = int(safe_duration_s * self.config.fps)
                        frames_to_generate = min(
                            allowed_frames - session.generated_frame_count,
                            session.total_available_frames - session.generated_frame_count
                        )
                        
                        if frames_to_generate > 0:
                            frames = self._generate_frames(session, frames_to_generate)
                            
                            for i, frame in enumerate(frames):
                                frame_idx = session.frame_index + i
                                timestamp = frame_idx * (1000.0 / self.config.fps)
                                self._send_frame(sender, frame, frame_idx, timestamp, is_final=False)
                            
                            session.frame_index += len(frames)
            
            # Always send final to mark end of this chunk's processing
            self._send_final(sender)
            
        except Exception as e:
            pb_utils.Logger.log_error(f"Error processing request: {e}\n{traceback.format_exc()}")
            self._send_final(sender)
    
    def finalize(self):
        """Cleanup on model unload"""
        try:
            pb_utils.Logger.log_info("Finalizing MuseTalk model...")
            
            # Cleanup all sessions
            for session_id in list(self.sessions.keys()):
                self._cleanup_session(session_id)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            pb_utils.Logger.log_error(f"Error in finalize: {e}")
