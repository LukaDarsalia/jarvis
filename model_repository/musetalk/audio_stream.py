"""
Abstract Audio Streaming Interface
No assumptions about audio length or format - only chunk-based streaming
"""

from abc import ABC, abstractmethod
from typing import Optional, Iterator
import numpy as np


class AudioStream(ABC):
    """
    Abstract base class for audio streaming.
    Represents a continuous stream of audio chunks with no prior knowledge of duration.
    """

    def __init__(self, sample_rate: int = 24000, chunk_duration_ms: int = 80):
        """
        Initialize audio stream.

        Args:
            sample_rate: Sample rate in Hz (e.g., 24000 for 24kHz)
            chunk_duration_ms: Duration of each chunk in milliseconds
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.samples_per_chunk = int((chunk_duration_ms / 1000.0) * sample_rate)

    @abstractmethod
    def get_next_chunk(self) -> Optional[np.ndarray]:
        """
        Get the next audio chunk from the stream.

        Returns:
            np.ndarray: Audio chunk of shape (samples_per_chunk,) with float32 values in [-1, 1]
                       Returns None when stream ends
        """
        pass

    @abstractmethod
    def is_streaming(self) -> bool:
        """
        Check if the stream is still active.

        Returns:
            bool: True if more chunks are available, False otherwise
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the stream to the beginning (if supported).
        """
        pass

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Make the stream iterable.

        Yields:
            Audio chunks until stream ends
        """
        while self.is_streaming():
            chunk = self.get_next_chunk()
            if chunk is not None:
                yield chunk
            else:
                break


class PseudoAudioStream(AudioStream):
    """
    Pseudo-streaming implementation that loads audio from a file but
    streams it chunk-by-chunk without revealing the total length.

    Simulates real-time audio streaming behavior for testing/development.
    """

    def __init__(self, audio_path: str, sample_rate: int = 24000, chunk_duration_ms: int = 80):
        """
        Initialize pseudo audio stream from a file.

        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate in Hz (will resample if needed)
            chunk_duration_ms: Duration of each chunk in milliseconds
        """
        super().__init__(sample_rate, chunk_duration_ms)

        self.audio_path = audio_path
        self._audio_data = None
        self._current_position = 0
        self._is_loaded = False

        # Load audio on initialization (but don't expose length)
        self._load_audio()

    def _load_audio(self) -> None:
        """
        Internal method to load audio file.
        This simulates receiving audio data from an external source.
        """
        import librosa

        # Load audio at target sample rate
        audio, sr = librosa.load(self.audio_path, sr=self.sample_rate, mono=True)

        # Normalize to [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure values are in [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        self._audio_data = audio
        self._is_loaded = True
        self._current_position = 0

    def get_next_chunk(self) -> Optional[np.ndarray]:
        """
        Get the next audio chunk.
        No information about remaining chunks is provided.

        Returns:
            Audio chunk or None if stream has ended
        """
        if not self._is_loaded or self._audio_data is None:
            return None

        # Check if we've reached the end
        if self._current_position >= len(self._audio_data):
            return None

        # Extract chunk
        end_position = self._current_position + self.samples_per_chunk
        chunk = self._audio_data[self._current_position:end_position]

        # If chunk is shorter than expected (at the end), pad with zeros
        if len(chunk) < self.samples_per_chunk:
            chunk = np.pad(chunk, (0, self.samples_per_chunk - len(chunk)), mode='constant')

        self._current_position = end_position

        return chunk

    def is_streaming(self) -> bool:
        """
        Check if more chunks are available.

        Returns:
            True if stream has more data
        """
        if not self._is_loaded or self._audio_data is None:
            return False

        return self._current_position < len(self._audio_data)

    def reset(self) -> None:
        """
        Reset stream to the beginning.
        """
        self._current_position = 0


class LiveAudioStream(AudioStream):
    """
    Placeholder for real-time audio streaming from microphone.
    Can be implemented using PyAudio, sounddevice, etc.
    """

    def __init__(self, sample_rate: int = 24000, chunk_duration_ms: int = 80):
        super().__init__(sample_rate, chunk_duration_ms)
        # TODO: Initialize audio input device
        raise NotImplementedError("LiveAudioStream not yet implemented")

    def get_next_chunk(self) -> Optional[np.ndarray]:
        # TODO: Read from microphone
        raise NotImplementedError()

    def is_streaming(self) -> bool:
        # TODO: Check if microphone is active
        raise NotImplementedError()

    def reset(self) -> None:
        # Not applicable for live streaming
        pass
