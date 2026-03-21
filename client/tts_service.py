"""
TTS Service with session management.

The TTS model uses sequence batching and requires a persistent gRPC stream
for the entire session lifecycle. This service manages:
- Session creation and initialization (KV cache allocation)
- Streaming text-to-speech generation
- Session cleanup
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional, Generator, Tuple, List, Callable

import numpy as np
import tritonclient.grpc as grpc_client

from config import TTSConfig
from models import TTSMetrics, TTSSessionError

logger = logging.getLogger(__name__)


class TTSSession:
    """
    Manages a persistent TTS session with Triton.

    The TTS model uses sequence batching which requires the SAME gRPC stream
    for the entire sequence lifecycle (sequence_start -> data -> sequence_end).

    Lifecycle:
        1. Create session with unique ID
        2. Call initialize() to allocate KV cache
        3. Call generate() multiple times to stream audio
        4. Call close() to release resources
    """

    MODEL_NAME = "tts"
    MAX_IDLE_SECONDS = 300  # 5 minutes

    def __init__(
        self,
        triton_url: str,
        session_id: int,
        config: TTSConfig,
    ):
        self.triton_url = triton_url
        self.session_id = session_id
        self.config = config

        self._client: Optional[grpc_client.InferenceServerClient] = None
        self._result_queue: Optional[queue.Queue] = None
        self._is_initialized = False
        self._is_closed = False
        self._lock = threading.Lock()
        self._last_used: float = time.time()
        self._audio_frame_count = 0
        self._audio_log_every = 50

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    @property
    def is_stale(self) -> bool:
        """Check if session has been idle too long."""
        return (time.time() - self._last_used) > self.MAX_IDLE_SECONDS

    @property
    def idle_seconds(self) -> float:
        """Get seconds since last use."""
        return time.time() - self._last_used

    def _callback(self, result, error):
        """Callback for stream responses."""
        if self._result_queue is not None:
            if error:
                self._result_queue.put(("error", error))
            else:
                self._result_queue.put(("result", result))

    def initialize(self, timeout: float = 30.0) -> bool:
        """
        Initialize the TTS session and allocate KV cache.

        Returns:
            True if initialization was successful
        """
        with self._lock:
            if self._is_closed:
                logger.warning(f"Cannot initialize closed session {self.session_id}")
                return False

            if self._is_initialized:
                logger.info(f"Session {self.session_id} already initialized")
                return True

            try:
                logger.info(f"TTS session {self.session_id} initializing...")

                # Create client and start stream
                self._result_queue = queue.Queue()
                self._client = grpc_client.InferenceServerClient(url=self.triton_url)
                self._client.start_stream(callback=self._callback)

                # Send init request (START=True)
                inputs = [
                    grpc_client.InferInput("START", [1], "BOOL"),
                    grpc_client.InferInput("CORRID", [1], "INT64"),
                ]
                inputs[0].set_data_from_numpy(np.array([True], dtype=bool))
                inputs[1].set_data_from_numpy(np.array([self.session_id], dtype=np.int64))

                outputs = [grpc_client.InferRequestedOutput("AUDIO_FRAME")]

                self._client.async_stream_infer(
                    model_name=self.MODEL_NAME,
                    inputs=inputs,
                    outputs=outputs,
                    sequence_id=self.session_id,
                    sequence_start=True,
                    sequence_end=False,
                    enable_empty_final_response=True,
                )

                # Wait for response
                try:
                    msg_type, data = self._result_queue.get(timeout=timeout)

                    if msg_type == "error":
                        logger.error(f"TTS session {self.session_id} init error: {data}")
                        self._cleanup_stream()
                        return False

                    self._is_initialized = True
                    logger.info(
                        "TTS session %s params: backbone_temp=%.3f backbone_top_p=%.3f "
                        "depth_temp=%.3f depth_top_p=%.3f",
                        self.session_id,
                        self.config.backbone_temperature,
                        self.config.backbone_top_p,
                        self.config.depth_temperature,
                        self.config.depth_top_p,
                    )
                    logger.info(f"TTS session {self.session_id} initialized successfully")
                    return True

                except queue.Empty:
                    logger.warning(
                        f"Timeout waiting for TTS init response for session {self.session_id}"
                    )
                    self._cleanup_stream()
                    return False

            except Exception as e:
                logger.error(f"TTS session {self.session_id} init failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self._cleanup_stream()
                return False

    def generate(
        self,
        text_chunks: List[str],
        on_audio: Optional[Callable[[np.ndarray, str, TTSMetrics], None]] = None,
    ) -> Generator[Tuple[np.ndarray, str, TTSMetrics], None, None]:
        """
        Generate audio from text chunks.

        The TTS model uses 2-word lookahead:
        - When sending text chunk[i], audio for word[i-2] is generated
        - First 3 words are sent together
        - Subsequent words are sent one at a time with leading space
        - Two empty strings at end to flush remaining words

        Args:
            text_chunks: Pre-split text chunks for streaming TTS
            on_audio: Optional callback for each audio chunk

        Yields:
            Tuple of (audio_array, word, metrics)
        """
        with self._lock:
            if self._is_closed:
                logger.error(f"Cannot generate on closed session {self.session_id}")
                return

            if not self._is_initialized:
                logger.error(f"Session {self.session_id} not initialized")
                return

            self._last_used = time.time()

        metrics = TTSMetrics()
        start_time = time.time()
        total_samples = 0

        # Build word list for tracking (one entry per chunk, matching word_audio_index)
        all_words = []
        for i, chunk in enumerate(text_chunks):
            stripped = chunk.strip()
            if stripped:
                all_words.append(stripped)
            else:
                all_words.append("")

        logger.info(
            f"TTS session {self.session_id}: generating {len(all_words)} words "
            f"from {len(text_chunks)} chunks"
        )

        word_audio_index = 0

        try:
            for i, chunk in enumerate(text_chunks):
                expected_word = all_words[word_audio_index] if word_audio_index < len(all_words) else ""

                logger.debug(f"TTS chunk {i}: '{chunk}' -> expecting word '{expected_word}'")

                texts = np.array([chunk.encode("utf-8")], dtype=object)

                inputs = [
                    grpc_client.InferInput("TEXTS", [1], "BYTES"),
                    grpc_client.InferInput("CORRID", [1], "INT64"),
                ]
                inputs[0].set_data_from_numpy(texts)
                inputs[1].set_data_from_numpy(np.array([self.session_id], dtype=np.int64))

                # Add optional temperature/top_p parameters
                if self.config.backbone_temperature is not None:
                    inp = grpc_client.InferInput("BACKBONE_TEMPERATURE", [1], "FP32")
                    inp.set_data_from_numpy(np.array([self.config.backbone_temperature], dtype=np.float32))
                    inputs.append(inp)

                if self.config.backbone_top_p is not None:
                    inp = grpc_client.InferInput("BACKBONE_TOP_P", [1], "FP32")
                    inp.set_data_from_numpy(np.array([self.config.backbone_top_p], dtype=np.float32))
                    inputs.append(inp)

                if self.config.depth_temperature is not None:
                    inp = grpc_client.InferInput("DEPTH_TEMPERATURE", [1], "FP32")
                    inp.set_data_from_numpy(np.array([self.config.depth_temperature], dtype=np.float32))
                    inputs.append(inp)

                if self.config.depth_top_p is not None:
                    inp = grpc_client.InferInput("DEPTH_TOP_P", [1], "FP32")
                    inp.set_data_from_numpy(np.array([self.config.depth_top_p], dtype=np.float32))
                    inputs.append(inp)

                outputs = [grpc_client.InferRequestedOutput("AUDIO_FRAME")]

                self._client.async_stream_infer(
                    model_name=self.MODEL_NAME,
                    inputs=inputs,
                    outputs=outputs,
                    sequence_id=self.session_id,
                    sequence_start=False,
                    sequence_end=False,
                    enable_empty_final_response=True,
                )

                # Collect audio for this chunk
                chunk_audio_count = 0
                while True:
                    try:
                        msg_type, data = self._result_queue.get(timeout=120.0)

                        if msg_type == "error":
                            logger.error(f"TTS Error: {data}")
                            return

                        response = data.get_response()

                        # Check for final response marker
                        if response.parameters.get("triton_final_response").bool_param:
                            logger.debug(
                                f"Chunk {i} complete, generated {chunk_audio_count} audio frames"
                            )
                            break

                        audio = np.asarray(data.as_numpy("AUDIO_FRAME"))
                        self._audio_frame_count += 1

                        if audio.size > 0:
                            log_frame = (
                                self._audio_frame_count <= 3
                                or self._audio_frame_count % self._audio_log_every == 0
                                or (audio.size != self.config.chunk_samples)
                                or (audio.dtype != np.float32)
                                or (audio.ndim != 1)
                            )
                            if log_frame:
                                min_val = float(np.min(audio))
                                max_val = float(np.max(audio))
                                rms_val = float(np.sqrt(np.mean(audio ** 2))) if audio.size > 0 else 0.0
                                nan_count = int(np.isnan(audio).sum())
                                logger.info(
                                    f"TTS session {self.session_id} AUDIO_FRAME {self._audio_frame_count}: "
                                    f"samples={audio.size} expected={self.config.chunk_samples} "
                                    f"dtype={audio.dtype} ndim={audio.ndim} min={min_val:.4f} "
                                    f"max={max_val:.4f} rms={rms_val:.4f} nan={nan_count}"
                                )

                        if len(audio) > 0:
                            chunk_audio_count += 1
                            total_samples += len(audio)

                            # Update metrics
                            elapsed = time.time() - start_time
                            metrics.generation_time_ms = elapsed * 1000
                            metrics.audio_duration_ms = (total_samples / self.config.sample_rate) * 1000
                            metrics.rtf = (
                                elapsed / (total_samples / self.config.sample_rate)
                                if total_samples > 0
                                else 0
                            )

                            current_word = expected_word
                            if current_word and current_word not in metrics.words_generated:
                                metrics.words_generated.append(current_word)

                            if on_audio:
                                on_audio(audio, current_word, metrics)

                            yield audio, current_word, metrics

                    except queue.Empty:
                        logger.warning(f"Timeout waiting for TTS response on chunk {i}")
                        return

                word_audio_index += 1

                if metrics.audio_duration_ms > 0:
                    logger.info(
                        f"TTS chunk {i} ({expected_word}): "
                        f"{metrics.audio_duration_ms:.1f}ms audio in "
                        f"{metrics.generation_time_ms:.1f}ms (RTF {metrics.rtf:.3f})"
                    )

        except Exception as e:
            logger.error(f"TTS session {self.session_id} generate error: {e}")
            import traceback
            logger.error(traceback.format_exc())

        logger.info(f"TTS session {self.session_id} generation complete. RTF: {metrics.rtf:.3f}")

    def close(self, timeout: float = 10.0) -> bool:
        """
        Close the TTS session, releasing resources on the server.

        Returns:
            True if session was closed successfully
        """
        with self._lock:
            if self._is_closed:
                return True

            if self._client is None:
                self._is_closed = True
                return True

            try:
                logger.info(f"Closing TTS session {self.session_id}...")

                # Send end request (END=True)
                inputs = [
                    grpc_client.InferInput("END", [1], "BOOL"),
                    grpc_client.InferInput("CORRID", [1], "INT64"),
                ]
                inputs[0].set_data_from_numpy(np.array([True], dtype=bool))
                inputs[1].set_data_from_numpy(np.array([self.session_id], dtype=np.int64))

                outputs = [grpc_client.InferRequestedOutput("AUDIO_FRAME")]

                self._client.async_stream_infer(
                    model_name=self.MODEL_NAME,
                    inputs=inputs,
                    outputs=outputs,
                    sequence_id=self.session_id,
                    sequence_start=False,
                    sequence_end=True,
                    enable_empty_final_response=True,
                )

                # Wait for confirmation
                try:
                    msg_type, data = self._result_queue.get(timeout=timeout)

                    if msg_type == "error":
                        logger.warning(
                            f"TTS session {self.session_id} end received error: {data}"
                        )
                    else:
                        logger.info(f"TTS session {self.session_id} ended successfully")

                except queue.Empty:
                    logger.warning(
                        f"Timeout waiting for TTS session {self.session_id} end confirmation"
                    )

            except Exception as e:
                logger.error(f"Error ending TTS session {self.session_id}: {e}")

            finally:
                self._cleanup_stream()
                self._is_closed = True
                self._is_initialized = False

            return True

    def _cleanup_stream(self):
        """Clean up the gRPC stream."""
        if self._client is not None:
            try:
                self._client.stop_stream()
            except Exception as e:
                logger.debug(f"Error stopping stream: {e}")
            self._client = None
        self._result_queue = None

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class TTSService:
    """
    TTS Service that manages multiple TTS sessions.

    Provides:
    - Session lifecycle management
    - Text splitting for the 2-word lookahead protocol
    - Convenience methods for synthesis
    """

    def __init__(self, triton_url: str, config: TTSConfig):
        self.triton_url = triton_url
        self.config = config

        self._session_counter = 100
        self._session_lock = threading.Lock()
        self._active_sessions: dict[int, TTSSession] = {}

    def _get_next_session_id(self) -> int:
        """Get a unique session ID."""
        with self._session_lock:
            self._session_counter += 1
            return self._session_counter

    def split_text_for_streaming(self, text: str) -> List[str]:
        """
        Split text for TTS streaming with 2-word lookahead.

        The TTS model generates audio for word[i-2] when receiving word[i].

        Example: "Hello how are you today?"
        Returns: ["Hello how are", " you", " today?", "", ""]

        Generation sequence:
        - Send chunk[0] ("Hello how are") -> generates "Hello"
        - Send chunk[1] (" you") -> generates "how"
        - Send chunk[2] (" today?") -> generates "are"
        - Send chunk[3] ("") -> generates "you"
        - Send chunk[4] ("") -> generates "today?"
        """
        text = text.replace("\n", " ").strip()
        words = text.split()

        if not words:
            return ["", ""]

        if len(words) <= 3:
            return [text, "", ""]

        # First chunk: first 3 words
        result = [" ".join(words[:3])]

        # Subsequent chunks: one word each with leading space
        for w in words[3:]:
            result.append(" " + w)

        # Two empty strings to flush the 2-word lookahead buffer
        result.extend(["", ""])

        return result

    def create_session(self) -> TTSSession:
        """
        Create a new TTS session with a unique ID.

        Returns:
            A new TTSSession instance (not yet initialized)
        """
        session_id = self._get_next_session_id()
        session = TTSSession(self.triton_url, session_id, self.config)

        with self._session_lock:
            self._active_sessions[session_id] = session

        return session

    def get_session(self, session_id: int) -> Optional[TTSSession]:
        """Get an existing TTS session by ID."""
        with self._session_lock:
            return self._active_sessions.get(session_id)

    def init_session(self, session_id: int) -> bool:
        """
        Initialize a TTS session (creates if needed).

        Args:
            session_id: Session ID to initialize

        Returns:
            True if initialized successfully
        """
        existing = self.get_session(session_id)
        if existing is not None:
            logger.info(f"Closing existing TTS session {session_id} before reinitializing")
            existing.close()
            with self._session_lock:
                self._active_sessions.pop(session_id, None)

        session = TTSSession(self.triton_url, session_id, self.config)
        success = session.initialize()

        if success:
            with self._session_lock:
                self._active_sessions[session_id] = session

        return success

    def close_session(self, session_id: int) -> bool:
        """
        Close and cleanup a TTS session.

        Args:
            session_id: Session ID to close

        Returns:
            True if closed successfully
        """
        with self._session_lock:
            session = self._active_sessions.pop(session_id, None)

        if session is not None:
            return session.close()
        return True

    def is_session_stale(self, session_id: int) -> bool:
        """
        Check if a TTS session is stale.

        Returns:
            True if session is stale or doesn't exist
        """
        session = self.get_session(session_id)
        if session is None:
            return True
        if session.is_closed:
            return True
        if not session.is_initialized:
            return True
        if session.is_stale:
            logger.warning(
                f"TTS session {session_id} is stale (idle for {session.idle_seconds:.1f}s)"
            )
            return True
        return False

    def generate_stream(
        self,
        text_chunks: List[str],
        session_id: int,
        on_audio: Optional[Callable[[np.ndarray, str, TTSMetrics], None]] = None,
    ) -> Generator[Tuple[np.ndarray, str, TTSMetrics], None, None]:
        """
        Stream TTS generation.

        Args:
            text_chunks: Pre-split text chunks
            session_id: Session ID to use (must be initialized)
            on_audio: Optional callback for audio chunks

        Yields:
            Tuple of (audio_array, word, metrics)
        """
        session = self.get_session(session_id)

        if session is None:
            logger.error(
                f"[TTS_STREAM] Session {session_id} not found. "
                "Did you call init_session first?"
            )
            return

        if not session.is_initialized:
            logger.error(f"[TTS_STREAM] Session {session_id} not initialized")
            return

        if session.is_closed:
            logger.error(f"[TTS_STREAM] Session {session_id} is already closed")
            return

        logger.info(
            f"[TTS_STREAM] Session {session_id}: "
            f"text='{' '.join(text_chunks)}', chunks={len(text_chunks)}"
        )

        yield from session.generate(text_chunks, on_audio=on_audio)

        logger.info(f"[TTS_STREAM] Session {session_id} COMPLETE")

    def synthesize_text(self, text: str) -> Tuple[np.ndarray, TTSMetrics]:
        """
        Synthesize complete audio from text (non-streaming).

        Handles the full session lifecycle: create -> init -> generate -> close
        """
        audio_chunks = []
        final_metrics = TTSMetrics()

        chunks = self.split_text_for_streaming(text)

        session_id = self._get_next_session_id()
        if not self.init_session(session_id):
            logger.error("Failed to initialize TTS session for synthesize_text")
            return np.array([], dtype=np.float32), final_metrics

        try:
            for audio, word, metrics in self.generate_stream(chunks, session_id=session_id):
                audio_chunks.append(audio)
                final_metrics = metrics
        finally:
            self.close_session(session_id)

        if audio_chunks:
            return np.concatenate(audio_chunks), final_metrics

        return np.array([], dtype=np.float32), final_metrics
