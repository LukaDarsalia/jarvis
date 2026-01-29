"""
Configuration management for the voice assistant client.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    speech_threshold_ms: float = 100.0
    silence_threshold_ms: float = 1200.0
    # Early silence threshold for speculative STT/LLM processing
    # When silence reaches this threshold, we start STT speculatively to reduce latency
    early_silence_threshold_ms: float = 400.0
    enable_speculative: bool = False
    prob_threshold: float = 0.5
    sample_rate: int = 16000
    chunk_samples: int = 512


@dataclass
class LLMConfig:
    """Large Language Model configuration."""
    max_new_tokens: int = 192
    temperature: float = 0.1
    top_p: float = 0.95
    system_prompt: str = (
        "You are the TBC Bank digital assistant. Help users with banking questions.\n\n"
        "Keep responses concise (2-4 sentences). Be clear and polite."
    )


@dataclass
class TTSConfig:
    """Text-to-Speech configuration."""
    backbone_temperature: float = 0.01
    backbone_top_p: float = 0.999
    depth_temperature: float = 0.01
    depth_top_p: float = 0.999
    sample_rate: int = 24000
    # TTS outputs 1920 samples per chunk (80ms @ 24kHz)
    chunk_samples: int = 1920
    # Lookahead: TTS generates audio for word[i-2] when receiving word[i]
    lookahead_words: int = 2
    # Pocket-TTS predefined voice (used if no voice prompt is provided)
    voice_id: str = "alba"


@dataclass
class MuseTalkConfig:
    """MuseTalk avatar generation configuration."""
    avatar_id: str = "default"
    fps: int = 25
    # Number of video frames to generate per Triton batch call
    # Higher = more efficient but higher latency, Lower = lower latency but less efficient
    # With 600ms Triton round-trip: batch_size=24 gives ~40fps sustained (24 frames / 600ms)
    batch_size: int = 24
    # Number of TTS chunks (80ms each) to keep as lookahead for Whisper context
    # This provides future audio context for better lip-sync accuracy
    # 1 chunk = 80ms lookahead
    lookahead_chunks: int = 1
    # Samples per video frame (24kHz / 25fps = 960 samples per frame)
    samples_per_frame: int = 960


@dataclass
class StreamingConfig:
    """Streaming and buffering configuration."""
    # Audio processor settings
    # Reduced batch size and timeout for more responsive VAD
    audio_batch_size: int = 2  # Reduced from 5 - process fewer chunks per batch
    audio_batch_timeout_s: float = 0.02  # Reduced from 0.05 - 20ms timeout
    audio_queue_max_size: int = 200

    # Adaptive buffer settings
    buffer_window_size: int = 50
    buffer_k_std: float = 1.645  # 95% one-sided confidence
    manual_buffer_ms: Optional[float] = None

    # Timeouts
    llm_timeout_s: float = 120.0
    tts_timeout_s: float = 120.0
    tts_init_timeout_s: float = 35.0
    musetalk_timeout_s: float = 60.0
    # Extra silence padding appended after TTS generation (ms)
    tts_silence_padding_ms: float = 500.0


@dataclass
class AppConfig:
    """Main application configuration."""
    triton_url: str = field(default_factory=lambda: os.environ.get("TRITON_URL", "localhost:8001"))
    host: str = "0.0.0.0"
    port: int = 8080

    vad: VADConfig = field(default_factory=VADConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    musetalk: MuseTalkConfig = field(default_factory=MuseTalkConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)


def load_config() -> AppConfig:
    """Load configuration from environment and defaults."""
    return AppConfig()
