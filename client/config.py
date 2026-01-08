"""
Configuration management for the voice assistant client.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    speech_threshold_ms: float = 200.0
    silence_threshold_ms: float = 1500.0
    prob_threshold: float = 0.5
    sample_rate: int = 16000
    chunk_samples: int = 512


@dataclass
class LLMConfig:
    """Large Language Model configuration."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: str = (
        "თქვენ ხართ თიბისი ბანკის ციფრული ასისტენტი, "
        "რომლის მოვალეობაცაა დაეხმაროს მომხმარებლებს საბანკო თემებში"
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


@dataclass
class MuseTalkConfig:
    """MuseTalk avatar generation configuration."""
    avatar_id: str = "default"
    fps: int = 25
    # How many TTS chunks to wait before starting MuseTalk
    start_after_chunks: int = 3
    # How many TTS chunks to keep as lookahead buffer
    lookahead_chunks: int = 2
    # Samples per video frame (24kHz / 25fps = 960 samples per frame)
    samples_per_frame: int = 960


@dataclass
class StreamingConfig:
    """Streaming and buffering configuration."""
    # Audio processor settings
    audio_batch_size: int = 5
    audio_batch_timeout_s: float = 0.05
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