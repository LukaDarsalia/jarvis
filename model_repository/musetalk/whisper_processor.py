"""
Whisper Stream Processor - Uses Existing AudioProcessor
Wraps the proven AudioProcessor from musetalk.utils.audio_processor
"""

import os
import tempfile
import math
import torch
import numpy as np
from collections import deque
from typing import Optional, Dict
import librosa
import soundfile as sf

# Import the existing AudioProcessor
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from audio_processor import AudioProcessor


class WhisperStreamProcessor:
    """
    Streaming wrapper around the existing AudioProcessor.
    Uses the EXACT same code path as the working MuseTalk implementation.

    Key: Buffers audio chunks and writes to temp file, then uses AudioProcessor
    to get features exactly as the original code does.
    """

    def __init__(
        self,
        whisper_model_path: str,
        whisper_model,
        input_sample_rate: int = 24000,
        whisper_sample_rate: int = 16000,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float16,
        chunk_duration_s: float = 0.08,
        lookahead_chunks: int = 0,
    ):
        """
        Initialize processor using existing AudioProcessor.

        Args:
            whisper_model_path: Path to Whisper model (for AudioProcessor)
            whisper_model: The actual Whisper model instance
            input_sample_rate: Input audio sample rate (24kHz)
            whisper_sample_rate: Whisper's required rate (16kHz)
            device: Device to run on
            dtype: Data type for inference
        """
        self.input_sample_rate = input_sample_rate
        self.whisper_sample_rate = whisper_sample_rate
        self.device = device
        self.dtype = dtype
        self.chunk_duration_s = chunk_duration_s
        self.lookahead_chunks = max(0, lookahead_chunks)
        self.base_future_context_s = 0.200  # 10 whisper frames @ 50 FPS (~200ms)
        self.stream_complete = False
        self.whisper_model = whisper_model

        # Use existing AudioProcessor - PROVEN CODE!
        self.audio_processor = AudioProcessor(feature_extractor_path=whisper_model_path)

        # Buffering
        self.audio_buffer = deque()
        self.min_buffer_duration_s = 0.200  # 200ms minimum
        self.min_buffer_chunks = max(1, math.ceil(self.min_buffer_duration_s / self.chunk_duration_s))

        # Feature storage
        self.whisper_features = None  # Stores processed Whisper features
        self.librosa_length = 0
        self.processed_chunks_count = 0
        self.generated_frame_count = 0  # Track how many frames we've already generated
        self.total_audio_duration_s = 0.0

        # Temp file for audio and cumulative buffer
        self.temp_audio_file = None
        self._original_audio = np.zeros(0, dtype=np.float32)
        self.whisper_chunks = None
        self.total_available_frames = 0
        self._cache_params = None

    def resample_audio(self, audio_samples: np.ndarray) -> np.ndarray:
        """Resample full audio from input sample rate to Whisper sample rate."""
        if self.input_sample_rate == self.whisper_sample_rate:
            return audio_samples.astype(np.float32)

        resampled = librosa.resample(
            audio_samples,
            orig_sr=self.input_sample_rate,
            target_sr=self.whisper_sample_rate
        )
        return resampled.astype(np.float32)

    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Process audio chunk using EXISTING AudioProcessor methods.

        Args:
            audio_chunk: Audio at input_sample_rate

        Returns:
            bool: True if new features computed
        """
        audio_chunk = np.asarray(audio_chunk)
        if audio_chunk.ndim > 1:
            audio_chunk = np.mean(audio_chunk, axis=0)
        audio_chunk = audio_chunk.astype(np.float32)

        # Buffer
        self.audio_buffer.append(audio_chunk)
        self._original_audio = np.concatenate((self._original_audio, audio_chunk))
        self.total_audio_duration_s = len(self._original_audio) / float(self.input_sample_rate)

        if len(self.audio_buffer) < self.min_buffer_chunks:
            return False

        if self.temp_audio_file is None:
            self.temp_audio_file = tempfile.NamedTemporaryFile(
                suffix='.wav',
                delete=False
            )

        resampled_audio = self.resample_audio(self._original_audio)

        sf.write(
            self.temp_audio_file.name,
            resampled_audio,
            self.whisper_sample_rate
        )

        try:
            whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(
                self.temp_audio_file.name,
                weight_dtype=self.dtype
            )

            if not whisper_input_features:
                return False

            self.whisper_input_features = whisper_input_features
            self.librosa_length = librosa_length
            self.processed_chunks_count += 1
            self.whisper_chunks = None
            self.total_available_frames = 0
            self._cache_params = None

            return True

        except Exception as e:
            print(f"Warning: Audio processing failed: {e}")
            return False

    def get_features_for_frames(
        self,
        num_frames: Optional[int] = None,
        video_fps: int = 25,
        audio_padding_left: int = 2,
        audio_padding_right: int = 2,
        lookahead_chunks: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Extract features for frames using EXISTING AudioProcessor.get_whisper_chunk().

        This uses the EXACT same code as the working implementation!

        IMPORTANT: This method tracks which frames have been generated to avoid
        returning the same frames twice. Call this with the TOTAL expected frames,
        and it will return only the NEW frames since last call.

        Args:
            num_frames: TOTAL number of frames expected so far (not delta!). If None,
                all available frames will be considered.
            video_fps: Video FPS
            audio_padding_left: Left padding
            audio_padding_right: Right padding

        Returns:
            Tensor with NEW frames only (from last position to num_frames)
        """
        if not hasattr(self, 'whisper_input_features') or self.whisper_input_features is None:
            return None

        cache_key = (video_fps, audio_padding_left, audio_padding_right)

        if self.whisper_chunks is None or self._cache_params != cache_key:
            try:
                whisper_chunks = self.audio_processor.get_whisper_chunk(
                    whisper_input_features=self.whisper_input_features,
                    device=self.device,
                    weight_dtype=self.dtype,
                    whisper=self.whisper_model,
                    librosa_length=self.librosa_length,
                    fps=video_fps,
                    audio_padding_length_left=audio_padding_left,
                    audio_padding_length_right=audio_padding_right
                )
            except Exception as e:
                print(f"Warning: Feature extraction failed: {e}")
                return None

            self.whisper_chunks = whisper_chunks
            self.total_available_frames = len(whisper_chunks)
            self._cache_params = cache_key

            if self.generated_frame_count > self.total_available_frames:
                self.generated_frame_count = self.total_available_frames

        if self.whisper_chunks is None or self.total_available_frames == 0:
            return None

        effective_lookahead = self.lookahead_chunks if lookahead_chunks is None else max(0, lookahead_chunks)
        allowed_frames = self._frames_allowed_with_context(video_fps, effective_lookahead)

        candidate_frames = self.total_available_frames if num_frames is None else min(num_frames, self.total_available_frames)
        target_frames = min(candidate_frames, allowed_frames) if allowed_frames > 0 else 0

        if target_frames < self.generated_frame_count:
            target_frames = self.generated_frame_count

        frames_to_generate = target_frames - self.generated_frame_count
        if frames_to_generate <= 0:
            return None

        start_idx = self.generated_frame_count
        end_idx = min(target_frames, self.total_available_frames)

        if start_idx >= end_idx:
            return None

        new_frames = self.whisper_chunks[start_idx:end_idx]
        self.generated_frame_count = end_idx

        return new_frames

    def reset(self):
        """Reset processor state"""
        self.audio_buffer.clear()
        self.whisper_features = None
        self.librosa_length = 0
        self.processed_chunks_count = 0
        self.generated_frame_count = 0
        self._original_audio = np.zeros(0, dtype=np.float32)
        self.whisper_chunks = None
        self.total_available_frames = 0
        self._cache_params = None

        if hasattr(self, 'whisper_input_features'):
            del self.whisper_input_features
        self.whisper_input_features = None

        # Clean up temp file
        if self.temp_audio_file is not None:
            try:
                os.unlink(self.temp_audio_file.name)
            except:
                pass
            self.temp_audio_file = None

    def get_stats(self, video_fps: Optional[int] = None) -> Dict[str, float]:
        """Get processing statistics"""
        buffer_duration_s = len(self.audio_buffer) * self.chunk_duration_s
        audio_duration_s = self.librosa_length / float(self.whisper_sample_rate) if self.librosa_length else self.total_audio_duration_s
        lookahead_duration_s = self._effective_future_context_s(self.lookahead_chunks)

        frames_ready = None
        allowed_frames = None
        if video_fps is not None:
            allowed_frames = self._frames_allowed_with_context(video_fps, self.lookahead_chunks)
            frames_ready = max(0, allowed_frames - self.generated_frame_count)

        return {
            'buffer_chunks': len(self.audio_buffer),
            'buffer_duration_s': buffer_duration_s,
            'processed_chunks': self.processed_chunks_count,
            'has_features': hasattr(self, 'whisper_input_features'),
            'min_buffer_ready': len(self.audio_buffer) >= self.min_buffer_chunks,
            'available_frames': self.total_available_frames,
            'generated_frames': self.generated_frame_count,
            'frames_ready': frames_ready,
            'allowed_frames': allowed_frames,
            'audio_duration_s': audio_duration_s,
            'lookahead_chunks': self.lookahead_chunks,
            'lookahead_duration_s': lookahead_duration_s,
            'stream_complete': self.stream_complete,
        }

    def available_frames(self) -> int:
        """Return total number of frames currently available."""
        return self.total_available_frames

    def __del__(self):
        """Cleanup on deletion"""
        if self.temp_audio_file is not None:
            try:
                os.unlink(self.temp_audio_file.name)
            except:
                pass

    def _effective_future_context_s(self, lookahead_chunks: int) -> float:
        if self.stream_complete:
            return 0.0
        return self.base_future_context_s + max(0, lookahead_chunks) * self.chunk_duration_s

    def _frames_allowed_with_context(self, video_fps: int, lookahead_chunks: int) -> int:
        if self.total_available_frames == 0 or video_fps <= 0:
            return 0
        audio_duration_s = self.librosa_length / float(self.whisper_sample_rate) if self.librosa_length else self.total_audio_duration_s
        safe_duration_s = audio_duration_s - self._effective_future_context_s(lookahead_chunks)
        if safe_duration_s <= 0:
            return 0
        allowed_frames = math.floor(safe_duration_s * video_fps)
        return max(0, min(allowed_frames, self.total_available_frames))

    def mark_stream_complete(self) -> None:
        """Mark the input audio stream as complete to relax lookahead constraints."""
        self.stream_complete = True
