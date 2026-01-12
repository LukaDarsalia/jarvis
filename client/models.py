"""
Data models and types for the voice assistant client.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any
import numpy as np


# ============================================================================
# Enums
# ============================================================================

class VADStatus(Enum):
    """Voice Activity Detection status."""
    LISTENING = "listening"
    SPEECH_START = "speech_start"
    SPEAKING = "speaking"
    SPEECH_CONTINUE = "speech_continue"
    SPEECH_RESUMED = "speech_resumed"  # Speech resumed after early silence - cancel speculative
    EARLY_SILENCE = "early_silence"  # Early silence detected, can start speculative STT/LLM
    UTTERANCE_COMPLETE = "utterance_complete"


class PipelineState(Enum):
    """State of the voice pipeline."""
    IDLE = auto()
    LISTENING = auto()
    PROCESSING_VAD = auto()
    TRANSCRIBING = auto()
    GENERATING_LLM = auto()
    SYNTHESIZING_TTS = auto()
    GENERATING_VIDEO = auto()
    STREAMING_AV = auto()
    ERROR = auto()


class MessageType(Enum):
    """WebSocket message types."""
    # Client -> Server
    RECORDING_START = "recording_start"
    RECORDING_STOP = "recording_stop"
    STOP_GENERATION = "stop_generation"

    # Server -> Client
    CONNECTED = "connected"
    VAD_STATUS = "vad_status"
    STT_START = "stt_start"
    STT_COMPLETE = "stt_complete"
    LLM_START = "llm_start"
    LLM_TOKEN = "llm_token"
    LLM_COMPLETE = "llm_complete"
    TTS_START = "tts_start"
    TTS_COMPLETE = "tts_complete"
    SYNCED_AV_FRAME = "synced_av_frame"
    VIDEO_COMPLETE = "video_complete"
    TTS_CACHE_READY = "tts_cache_ready"
    MUSETALK_READY = "musetalk_ready"
    ERROR = "error"


# ============================================================================
# TTS Related Models
# ============================================================================

@dataclass
class TTSMetrics:
    """Metrics for TTS generation."""
    generation_time_ms: float = 0.0
    audio_duration_ms: float = 0.0
    rtf: float = 0.0  # Real-time factor
    words_generated: List[str] = field(default_factory=list)

    def copy(self) -> "TTSMetrics":
        """Create a copy of the metrics."""
        return TTSMetrics(
            generation_time_ms=self.generation_time_ms,
            audio_duration_ms=self.audio_duration_ms,
            rtf=self.rtf,
            words_generated=list(self.words_generated),
        )


@dataclass
class AudioFrame:
    """
    A single audio frame with metadata for AV synchronization.

    In the pipeline:
    - TTS generates audio at 24kHz
    - Each frame contains samples_per_frame samples (960 @ 25fps = 40ms)
    - Frame index is used for video synchronization
    """
    index: int
    samples: np.ndarray
    word: str = ""
    metrics: Optional[TTSMetrics] = None
    timestamp_ms: float = 0.0

    @property
    def duration_ms(self) -> float:
        """Duration of this frame in milliseconds."""
        if self.samples is None or len(self.samples) == 0:
            return 0.0
        # Assuming 24kHz sample rate
        return len(self.samples) / 24.0


@dataclass
class VideoFrame:
    """A video frame from MuseTalk."""
    index: int
    jpeg_bytes: bytes
    timestamp_ms: float


@dataclass
class AVFrame:
    """
    Combined audio-video frame for synchronized streaming.

    This represents a single 40ms frame (at 25fps) with both
    audio samples and corresponding video.
    """
    frame_index: int
    audio_samples: np.ndarray
    video_jpeg: Optional[bytes]
    word: str = ""
    timestamp_ms: float = 0.0
    metrics: Optional[TTSMetrics] = None

    def to_websocket_payload(self) -> Dict[str, Any]:
        """Convert to WebSocket payload format."""
        import base64

        payload = {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "word": self.word,
            "audio": "",
            "frame": "",
        }

        if self.audio_samples is not None and len(self.audio_samples) > 0:
            payload["audio"] = base64.b64encode(self.audio_samples.tobytes()).decode("utf-8")

        if self.video_jpeg is not None:
            payload["frame"] = base64.b64encode(self.video_jpeg).decode("utf-8")

        if self.metrics and self.metrics.audio_duration_ms > 0:
            payload["rtf"] = round(self.metrics.rtf, 3)
            payload["generation_time_ms"] = round(self.metrics.generation_time_ms, 1)
            payload["audio_duration_ms"] = round(self.metrics.audio_duration_ms, 1)

        return payload


# ============================================================================
# Streaming Metrics
# ============================================================================

@dataclass
class StreamingStats:
    """Statistics for a single metric type."""
    samples: List[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    window_size: int = 50

    def add_sample(self, value: float) -> None:
        """Add a sample to the rolling window."""
        if value > 0:
            self.samples.append(value)
            if len(self.samples) > self.window_size:
                self.samples.pop(0)
            self._update_stats()

    def _update_stats(self) -> None:
        """Update mean and std."""
        if not self.samples:
            self.mean = 0.0
            self.std = 0.0
            return

        self.mean = sum(self.samples) / len(self.samples)
        if len(self.samples) < 2:
            self.std = 0.0
        else:
            variance = sum((x - self.mean) ** 2 for x in self.samples) / (len(self.samples) - 1)
            self.std = variance ** 0.5


@dataclass
class BufferConfig:
    """Buffer configuration for adaptive streaming."""
    buffer_ms: float = 0.0
    frame_buffer: int = 0
    buffer_source: str = "adaptive"  # "adaptive" or "manual"
    manual_buffer_ms: Optional[float] = None
    is_calibrated: bool = False

    # Component breakdown
    tts_rtf_mean: float = 0.0
    tts_rtf_std: float = 0.0
    tts_overage_ms: float = 0.0
    musetalk_fps_mean: float = 25.0
    musetalk_fps_std: float = 0.0
    musetalk_overage_ms: float = 0.0
    network_latency_mean_ms: float = 0.0
    network_latency_std_ms: float = 0.0
    network_buffer_ms: float = 40.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "buffer_ms": round(self.buffer_ms, 1),
            "frame_buffer": self.frame_buffer,
            "buffer_source": self.buffer_source,
            "manual_buffer_ms": self.manual_buffer_ms,
            "is_calibrated": self.is_calibrated,
            "tts_rtf_mean": round(self.tts_rtf_mean, 3),
            "tts_rtf_std": round(self.tts_rtf_std, 3),
            "tts_overage_ms": round(self.tts_overage_ms, 1),
            "musetalk_fps_mean": round(self.musetalk_fps_mean, 1),
            "musetalk_fps_std": round(self.musetalk_fps_std, 1),
            "musetalk_overage_ms": round(self.musetalk_overage_ms, 1),
            "network_latency_mean_ms": round(self.network_latency_mean_ms, 1),
            "network_latency_std_ms": round(self.network_latency_std_ms, 1),
            "network_buffer_ms": round(self.network_buffer_ms, 1),
        }


# ============================================================================
# Conversation Management
# ============================================================================

@dataclass
class ConversationMessage:
    """A single message in the conversation."""
    role: str  # "user", "assistant", or "system"
    content: str


class ConversationHistory:
    """Manages conversation history for the LLM."""

    def __init__(self, max_messages: int = 20):
        self.messages: List[ConversationMessage] = []
        self.max_messages = max_messages

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.messages.append(ConversationMessage(role="user", content=content))
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message."""
        self.messages.append(ConversationMessage(role="assistant", content=content))
        self._trim()

    def _trim(self) -> None:
        """Trim history to max size."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_history(self) -> List[Dict[str, str]]:
        """Get history as list of dicts for LLM."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def clear(self) -> None:
        """Clear all history."""
        self.messages.clear()


# ============================================================================
# Error Types
# ============================================================================

class VoiceAssistantError(Exception):
    """Base exception for voice assistant errors."""
    pass


class TritonConnectionError(VoiceAssistantError):
    """Error connecting to Triton server."""
    pass


class TTSSessionError(VoiceAssistantError):
    """Error with TTS session management."""
    pass


class PipelineError(VoiceAssistantError):
    """Error in the voice pipeline."""
    pass
