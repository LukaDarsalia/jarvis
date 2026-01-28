"""
Triton gRPC client services for VAD, STT, LLM, and MuseTalk.

Each service provides a clean interface for interacting with its
corresponding Triton model.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional, Generator, Tuple, List, Callable

import numpy as np
import tritonclient.grpc as grpc_client
from tritonclient.utils import InferenceServerException

from config import VADConfig, LLMConfig, TTSConfig, MuseTalkConfig
from models import VADStatus, TTSMetrics

logger = logging.getLogger(__name__)


# ============================================================================
# Base Triton Client
# ============================================================================

class TritonClientBase:
    """Base class for Triton client services."""

    # gRPC keepalive settings - conservative to avoid server rejection
    # Server may enforce minimum ping intervals, so we use longer intervals
    GRPC_KEEPALIVE_OPTIONS = [
        ('grpc.keepalive_time_ms', 60000),  # Send keepalive ping every 60s
        ('grpc.keepalive_timeout_ms', 20000),  # Wait 20s for ping ack
        ('grpc.keepalive_permit_without_calls', True),  # Allow keepalive without active calls
        ('grpc.http2.min_time_between_pings_ms', 60000),  # Min 60s between pings
        ('grpc.http2.max_pings_without_data', 0),  # Unlimited pings without data
    ]

    def __init__(self, triton_url: str):
        self.triton_url = triton_url
        self._client: Optional[grpc_client.InferenceServerClient] = None
        self._lock = threading.Lock()

    @property
    def client(self) -> grpc_client.InferenceServerClient:
        """Get or create the Triton client."""
        with self._lock:
            if self._client is None:
                self._client = grpc_client.InferenceServerClient(
                    url=self.triton_url,
                    channel_args=self.GRPC_KEEPALIVE_OPTIONS,
                )
            return self._client

    def is_healthy(self) -> bool:
        """Check if Triton server is healthy."""
        try:
            return self.client.is_server_live() and self.client.is_server_ready()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def is_model_ready(self, model_name: str) -> bool:
        """Check if a specific model is ready."""
        try:
            return self.client.is_model_ready(model_name)
        except Exception as e:
            logger.error(f"Model check failed for {model_name}: {e}")
            return False


# ============================================================================
# VAD Service
# ============================================================================

class VADService(TritonClientBase):
    """Voice Activity Detection service."""

    MODEL_NAME = "vad"

    def __init__(self, triton_url: str, config: VADConfig):
        super().__init__(triton_url)
        self.config = config
        logger.info(f"[VAD] Initialized with early_silence={config.early_silence_threshold_ms}ms, "
                   f"silence={config.silence_threshold_ms}ms, speech={config.speech_threshold_ms}ms")

        # VAD state machine
        self._is_speaking = False
        self._speech_start_time: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        self._accumulated_audio: List[np.ndarray] = []
        self._early_silence_triggered = False  # Track if we already sent EARLY_SILENCE

    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, float, bool]:
        """
        Process a single audio chunk through VAD.

        Args:
            audio_chunk: Audio samples (float32, 512 samples at 16kHz)

        Returns:
            Tuple of (is_speech, probability, end_of_utterance)
        """
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        inputs = [grpc_client.InferInput("AUDIO_PCM", audio_chunk.shape, "FP32")]
        inputs[0].set_data_from_numpy(audio_chunk)

        outputs = [
            grpc_client.InferRequestedOutput("IS_SPEECH"),
            grpc_client.InferRequestedOutput("PROB"),
            grpc_client.InferRequestedOutput("END_OF_UTTERANCE"),
        ]

        result = self.client.infer(self.MODEL_NAME, inputs, outputs=outputs)

        is_speech = bool(result.as_numpy("IS_SPEECH")[0])
        prob = float(result.as_numpy("PROB")[0])
        end_of_utt = bool(result.as_numpy("END_OF_UTTERANCE")[0])

        return is_speech, prob, end_of_utt

    def process_with_state(
        self,
        audio_chunk: np.ndarray,
        current_time_ms: float,
    ) -> Tuple[VADStatus, Optional[np.ndarray]]:
        """
        Process VAD with state management.

        Args:
            audio_chunk: Audio samples
            current_time_ms: Current time in milliseconds

        Returns:
            Tuple of (status, audio_if_available)
            - EARLY_SILENCE: returns audio so far for speculative processing
            - UTTERANCE_COMPLETE: returns final complete audio
            - Others: returns None
        """
        is_speech, prob, _ = self.process_chunk(audio_chunk)

        if is_speech:
            # Check if this is a brand new utterance (after full reset)
            if self._speech_start_time is None:
                self._speech_start_time = current_time_ms
                self._accumulated_audio = []
                self._early_silence_triggered = False
                logger.debug(f"[VAD] New utterance started at {current_time_ms:.0f}ms")
            elif not self._is_speaking:
                # Speech resuming after a pause within the same utterance
                # Reset early silence flag so it can trigger again with updated audio
                was_early_triggered = self._early_silence_triggered
                logger.info(f"[VAD] Speech resumed at {current_time_ms:.0f}ms, "
                           f"resetting early_silence_triggered from {self._early_silence_triggered} to False")
                self._early_silence_triggered = False
                
                # If early silence was triggered, we need to signal speech resumed
                # so speculative processing can be cancelled
                if was_early_triggered:
                    self._is_speaking = True
                    self._last_speech_time = current_time_ms
                    self._accumulated_audio.append(audio_chunk)
                    return VADStatus.SPEECH_RESUMED, None

            self._is_speaking = True
            self._last_speech_time = current_time_ms
            self._accumulated_audio.append(audio_chunk)

            # Calculate total speech duration from original start
            total_speech_duration = self._last_speech_time - self._speech_start_time
            if total_speech_duration >= self.config.speech_threshold_ms:
                return VADStatus.SPEAKING, None
            return VADStatus.LISTENING, None

        else:
            # No speech detected
            if self._speech_start_time is not None:
                # We're in an utterance (may or may not be currently speaking)
                silence_duration = current_time_ms - self._last_speech_time
                self._accumulated_audio.append(audio_chunk)
                
                # Total speech duration from the original utterance start
                total_speech_duration = self._last_speech_time - self._speech_start_time

                # Check for full utterance completion first
                if silence_duration >= self.config.silence_threshold_ms:
                    if total_speech_duration >= self.config.speech_threshold_ms:
                        complete_audio = np.concatenate(self._accumulated_audio)
                        self.reset_state()
                        return VADStatus.UTTERANCE_COMPLETE, complete_audio

                    # Speech too short, reset
                    self.reset_state()
                    return VADStatus.LISTENING, None

                # Check for early silence (speculative processing trigger)
                # Can trigger multiple times if speech resumes and then goes silent again
                early_conditions = (
                    not self._early_silence_triggered,
                    silence_duration >= self.config.early_silence_threshold_ms,
                    total_speech_duration >= self.config.speech_threshold_ms
                )
                
                if silence_duration >= 400:  # Log when approaching threshold
                    logger.info(f"[VAD] Silence check: duration={silence_duration:.0f}ms, "
                               f"early_triggered={self._early_silence_triggered}, "
                               f"conditions={early_conditions}, "
                               f"early_thresh={self.config.early_silence_threshold_ms}ms")
                
                if all(early_conditions):
                    self._early_silence_triggered = True
                    # Return copy of audio so far for speculative processing
                    early_audio = np.concatenate(self._accumulated_audio)
                    logger.info(f"[VAD] EARLY_SILENCE triggered: silence_duration={silence_duration:.0f}ms, "
                               f"early_threshold={self.config.early_silence_threshold_ms}ms, "
                               f"full_threshold={self.config.silence_threshold_ms}ms")
                    return VADStatus.EARLY_SILENCE, early_audio

                # Mark as not actively speaking (but still in utterance)
                self._is_speaking = False
                return VADStatus.SPEAKING, None

            return VADStatus.LISTENING, None

    def reset_state(self) -> None:
        """Reset the VAD state machine."""
        self._speech_start_time = None
        self._last_speech_time = None
        self._accumulated_audio = []
        self._is_speaking = False
        self._early_silence_triggered = False


# ============================================================================
# STT Service
# ============================================================================

class STTService(TritonClientBase):
    """Speech-to-Text service."""

    MODEL_NAME = "stt"

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples (float32, 16kHz)

        Returns:
            Transcribed text
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        inputs = [grpc_client.InferInput("AUDIO_PCM", audio.shape, "FP32")]
        inputs[0].set_data_from_numpy(audio)

        outputs = [grpc_client.InferRequestedOutput("TRANSCRIPT")]

        result = self.client.infer(self.MODEL_NAME, inputs, outputs=outputs)
        transcript = result.as_numpy("TRANSCRIPT")[0]

        if isinstance(transcript, bytes):
            transcript = transcript.decode("utf-8")

        return transcript.strip()


# ============================================================================
# LLM Service
# ============================================================================

class LLMService(TritonClientBase):
    """Large Language Model service with streaming support."""

    MODEL_NAME = "llm"

    def __init__(self, triton_url: str, config: LLMConfig):
        super().__init__(triton_url)
        self.config = config

    def build_prompt(
        self,
        user_message: str,
        conversation_history: Optional[List[dict]] = None,
    ) -> str:
        """
        Build chat prompt with system message and history.

        Args:
            user_message: The user's current message
            conversation_history: Previous messages

        Returns:
            Formatted prompt string
        """
        logger.debug(f"[LLM] Building prompt with system_prompt: {self.config.system_prompt[:50]}...")
        messages = [{"role": "system", "content": self.config.system_prompt}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_message})

        prompt_parts = ["<s>"]
        for msg in messages:
            prompt_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        prompt_parts.append("<|im_start|>assistant\n")

        return "".join(prompt_parts)

    def generate_stream(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
        timeout: float = 120.0,
    ) -> Generator[str, None, None]:
        """
        Stream LLM generation token by token.

        Args:
            prompt: The input prompt
            on_token: Optional callback for each token
            timeout: Timeout for the stream

        Yields:
            Generated tokens
        """
        prompt_bytes = np.array([prompt.encode("utf-8")], dtype=object)

        inputs = [
            grpc_client.InferInput("PROMPT", [1], "BYTES"),
            grpc_client.InferInput("MAX_NEW_TOKENS", [1], "INT32"),
            grpc_client.InferInput("TEMPERATURE", [1], "FP32"),
            grpc_client.InferInput("TOP_P", [1], "FP32"),
        ]

        inputs[0].set_data_from_numpy(prompt_bytes)
        inputs[1].set_data_from_numpy(np.array([self.config.max_new_tokens], dtype=np.int32))
        inputs[2].set_data_from_numpy(np.array([self.config.temperature], dtype=np.float32))
        inputs[3].set_data_from_numpy(np.array([self.config.top_p], dtype=np.float32))

        outputs = [
            grpc_client.InferRequestedOutput("TEXT_CHUNK"),
            grpc_client.InferRequestedOutput("FINISHED"),
        ]

        result_queue: queue.Queue = queue.Queue()
        stream_done = threading.Event()

        def callback(result, error):
            if error:
                result_queue.put(("error", str(error)))
                stream_done.set()
            elif result:
                result_queue.put(("result", result))
            else:
                stream_done.set()

        # Create a new client for streaming
        stream_client = grpc_client.InferenceServerClient(url=self.triton_url)
        stream_client.start_stream(callback=callback)

        first_token_time: Optional[float] = None
        token_count = 0

        try:
            stream_client.async_stream_infer(
                model_name=self.MODEL_NAME,
                inputs=inputs,
                outputs=outputs,
            )

            while not stream_done.is_set():
                try:
                    msg_type, data = result_queue.get(timeout=1.0)

                    if msg_type == "error":
                        logger.error(f"LLM stream error: {data}")
                        break

                    elif msg_type == "result":
                        try:
                            chunk = data.as_numpy("TEXT_CHUNK")[0]
                            if isinstance(chunk, bytes):
                                chunk = chunk.decode("utf-8")

                            finished = bool(data.as_numpy("FINISHED")[0])

                            if chunk:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                token_count += 1

                                if on_token:
                                    on_token(chunk)
                                yield chunk

                            if finished:
                                break

                        except Exception as e:
                            logger.error(f"Error processing LLM response: {e}")
                            break

                except queue.Empty:
                    continue

        finally:
            stream_client.stop_stream()

            if first_token_time is not None and token_count > 1:
                duration = time.time() - first_token_time
                tokens_per_sec = (token_count - 1) / duration if duration > 0 else 0
                logger.info(
                    f"LLM stream completed: {token_count} tokens in {duration*1000:.0f}ms "
                    f"({tokens_per_sec:.1f} tok/s)"
                )


# ============================================================================
# MuseTalk Service
# ============================================================================

class MuseTalkService(TritonClientBase):
    """MuseTalk video generation service (stateless)."""

    MODEL_NAME = "musetalk"

    def __init__(self, triton_url: str, config: MuseTalkConfig):
        super().__init__(triton_url)
        self.config = config

    def generate_frames(
        self,
        audio: np.ndarray,
        frame_index: int = 0,
        on_frame: Optional[Callable[[bytes, int, float], None]] = None,
        timeout: float = 60.0,
    ) -> Generator[Tuple[bytes, int, float], None, None]:
        """
        Generate video frames from audio (stateless).

        Args:
            audio: Audio samples at 24kHz as float32
            frame_index: Starting frame index for avatar cycle
            on_frame: Optional callback for each frame
            timeout: Timeout for waiting for frames

        Yields:
            Tuple of (jpeg_bytes, frame_index, timestamp_ms)
        """
        if audio is None or len(audio) == 0:
            logger.error("MuseTalk: Audio is required and must not be empty")
            return

        audio_duration_s = len(audio) / 24000.0
        logger.info(f"MuseTalk: Processing {audio_duration_s:.3f}s audio, start_frame_index={frame_index}")

        result_queue: queue.Queue = queue.Queue()
        start_time = time.time()
        frames_received = 0

        def callback(result, error):
            if error:
                result_queue.put(("error", error))
            else:
                result_queue.put(("result", result))

        # Create client for this request
        client = grpc_client.InferenceServerClient(url=self.triton_url)

        try:
            client.start_stream(callback=callback)

            audio = np.asarray(audio, dtype=np.float32)

            inputs = [
                grpc_client.InferInput("AUDIO", list(audio.shape), "FP32"),
                grpc_client.InferInput("FRAME_INDEX", [1], "INT32"),
            ]
            inputs[0].set_data_from_numpy(audio)
            inputs[1].set_data_from_numpy(np.array([frame_index], dtype=np.int32))

            outputs = [
                grpc_client.InferRequestedOutput("VIDEO_FRAME"),
                grpc_client.InferRequestedOutput("FRAME_INDEX"),
                grpc_client.InferRequestedOutput("TIMESTAMP_MS"),
            ]

            client.async_stream_infer(
                model_name=self.MODEL_NAME,
                inputs=inputs,
                outputs=outputs,
                enable_empty_final_response=True,
            )

            while True:
                try:
                    msg_type, data = result_queue.get(timeout=timeout)

                    if msg_type == "error":
                        logger.error(f"MuseTalk error: {data}")
                        break

                    response = data.get_response()
                    is_final = response.parameters.get("triton_final_response").bool_param

                    frame_data = data.as_numpy("VIDEO_FRAME")
                    output_frame_index = int(data.as_numpy("FRAME_INDEX")[0])
                    timestamp_ms = float(data.as_numpy("TIMESTAMP_MS")[0])

                    if len(frame_data) > 0:
                        frames_received += 1
                        frame_bytes = frame_data.tobytes()

                        if on_frame:
                            on_frame(frame_bytes, output_frame_index, timestamp_ms)

                        yield frame_bytes, output_frame_index, timestamp_ms

                    if is_final:
                        break

                except queue.Empty:
                    logger.warning("Timeout waiting for MuseTalk frame")
                    break

            duration = time.time() - start_time
            logger.info(f"MuseTalk: Generated {frames_received} frames in {duration:.3f}s")

        except Exception as e:
            logger.error(f"MuseTalk generation error: {e}")
            import traceback
            logger.error(traceback.format_exc())

        finally:
            try:
                client.stop_stream()
            except Exception:
                pass

    def get_idle_frame(self, timeout: float = 30.0) -> Optional[bytes]:
        """
        Get a single idle frame from MuseTalk.

        Returns:
            JPEG bytes of the idle frame, or None if failed
        """
        # Generate minimal audio (240ms of silence)
        silent_audio = np.zeros(5760, dtype=np.float32)

        try:
            for frame_bytes, _, _ in self.generate_frames(
                audio=silent_audio,
                frame_index=0,
                timeout=timeout,
            ):
                logger.info(f"MuseTalk: Got idle frame ({len(frame_bytes)} bytes)")
                return frame_bytes

        except Exception as e:
            logger.error(f"Failed to get MuseTalk idle frame: {e}")

        return None


# ============================================================================
# Unified Triton Client
# ============================================================================

class TritonClient:
    """
    Unified client for all Triton services.

    Provides a single interface for health checks and model status.
    """

    MODELS = ["vad", "stt", "llm", "tts", "musetalk"]

    def __init__(
        self,
        triton_url: str,
        vad_config: VADConfig,
        llm_config: LLMConfig,
        tts_config: TTSConfig,
        musetalk_config: MuseTalkConfig,
    ):
        self.triton_url = triton_url

        # Create individual services
        self.vad = VADService(triton_url, vad_config)
        self.stt = STTService(triton_url)
        self.llm = LLMService(triton_url, llm_config)
        self.musetalk = MuseTalkService(triton_url, musetalk_config)

        # Configs for TTS (TTS service is separate due to session management)
        self.tts_config = tts_config

        logger.info(f"TritonClient initialized with URL: {triton_url}")

    def is_healthy(self) -> bool:
        """Check if Triton server is healthy."""
        return self.vad.is_healthy()

    def check_models_ready(self) -> dict:
        """Check if all models are ready."""
        status = {}
        for model in self.MODELS:
            status[model] = self.vad.is_model_ready(model)
        return status
