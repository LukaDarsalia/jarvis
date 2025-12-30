"""
Triton Client for Voice Assistant Pipeline
Handles VAD, STT, LLM, TTS, and MuseTalk model interactions
"""

import math
import numpy as np
import tritonclient.grpc as grpc_client
from tritonclient.utils import InferenceServerException
import threading
import queue
import time
from typing import Optional, Callable, Generator, List, Tuple
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VADParams:
    """VAD configuration parameters"""
    speech_threshold_ms: float = 200
    silence_threshold_ms: float = 1500
    prob_threshold: float = 0.5


@dataclass
class LLMParams:
    """LLM configuration parameters"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: str = "თქვენ ხართ თიბისი ბანკის ციფრული ასისტენტი, რომლის მოვალეობაცაა დაეხმაროს მომხმარებლებს საბანკო თემებში"


@dataclass
class TTSParams:
    """TTS configuration parameters"""
    backbone_temperature: float = 0.8
    backbone_top_p: float = 0.9
    depth_temperature: float = 0.8
    depth_top_p: float = 0.9
    target_sample_rate: int = 24000


@dataclass
class TTSMetrics:
    """TTS performance metrics"""
    generation_time_ms: float = 0
    audio_duration_ms: float = 0
    rtf: float = 0
    words_generated: List[str] = field(default_factory=list)


@dataclass
class MuseTalkParams:
    """MuseTalk configuration parameters"""
    avatar_id: str = "default"
    fps: int = 25
    start_after_chunks: int = 3
    lookahead_chunks: int = 2


@dataclass
class MuseTalkMetrics:
    """MuseTalk performance metrics"""
    frame_index: int = 0
    timestamp_ms: float = 0
    frames_generated: int = 0
    generation_time_ms: float = 0


class StreamingMetrics:
    """
    Tracks generation timing statistics for adaptive buffering.
    
    Maintains rolling windows of generation times and calculates
    optimal buffer sizes based on mean + k*std to handle jitter.
    """
    
    def __init__(self, window_size: int = 50, k_std: float = 1.645):
        """
        Args:
            window_size: Number of samples to keep in rolling window
            k_std: Number of standard deviations to add to mean for buffer calculation
                  (1.645 ~= one-sided 95% confidence)
        """
        self.window_size = window_size
        self.k_std = k_std
        
        # Rolling windows for different metrics
        self._llm_tokens_per_sec: List[float] = []
        self._tts_rtf: List[float] = []  # Real-time factor (< 1 means faster than real-time)
        self._musetalk_fps: List[float] = []
        self._network_latency_ms: List[float] = []
        
        # Calibration results
        self._is_calibrated = False
        self._calibration_llm_tokens_per_sec: float = 0
        self._calibration_tts_rtf: float = 0
        self._calibration_musetalk_fps: float = 0
        
        # Manual override for buffer size (ms)
        self._manual_buffer_ms: Optional[float] = None
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def _add_sample(self, samples: List[float], value: float):
        """Add a sample to a rolling window"""
        samples.append(value)
        if len(samples) > self.window_size:
            samples.pop(0)
    
    def _calc_stats(self, samples: List[float]) -> Tuple[float, float]:
        """Calculate mean and std of samples"""
        if not samples:
            return 0.0, 0.0
        mean = sum(samples) / len(samples)
        if len(samples) < 2:
            return mean, 0.0
        variance = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
        std = variance ** 0.5
        return mean, std
    
    def record_llm_generation(self, tokens: int, duration_sec: float):
        """Record LLM token generation timing"""
        if duration_sec > 0 and tokens > 0:
            tokens_per_sec = tokens / duration_sec
            with self._lock:
                self._add_sample(self._llm_tokens_per_sec, tokens_per_sec)
    
    def record_tts_generation(self, audio_duration_ms: float, generation_time_ms: float):
        """Record TTS generation timing"""
        if generation_time_ms > 0 and audio_duration_ms > 0:
            rtf = generation_time_ms / audio_duration_ms
            with self._lock:
                self._add_sample(self._tts_rtf, rtf)
    
    def record_musetalk_generation(self, frames: int, duration_sec: float):
        """Record MuseTalk frame generation timing"""
        if duration_sec > 0 and frames > 0:
            fps = frames / duration_sec
            with self._lock:
                self._add_sample(self._musetalk_fps, fps)
    
    def record_network_latency(self, latency_ms: float):
        """Record network round-trip latency"""
        if latency_ms > 0:
            with self._lock:
                self._add_sample(self._network_latency_ms, latency_ms)

    def set_manual_buffer_ms(self, buffer_ms: Optional[float]):
        """Set or clear a manual buffer override (ms)"""
        with self._lock:
            if buffer_ms is None:
                self._manual_buffer_ms = None
                logger.info("Cleared manual buffer override; using adaptive buffer.")
            else:
                self._manual_buffer_ms = max(0.0, float(buffer_ms))
                logger.info(f"Manual buffer override set to {self._manual_buffer_ms:.1f} ms")
    
    def get_llm_stats(self) -> Tuple[float, float]:
        """Get LLM tokens/sec mean and std"""
        with self._lock:
            return self._calc_stats(self._llm_tokens_per_sec)
    
    def get_tts_stats(self) -> Tuple[float, float]:
        """Get TTS RTF mean and std"""
        with self._lock:
            return self._calc_stats(self._tts_rtf)
    
    def get_musetalk_stats(self) -> Tuple[float, float]:
        """Get MuseTalk FPS mean and std"""
        with self._lock:
            return self._calc_stats(self._musetalk_fps)
    
    def get_network_stats(self) -> Tuple[float, float]:
        """Get network latency mean and std in ms"""
        with self._lock:
            return self._calc_stats(self._network_latency_ms)

    def _compute_buffer_components(self) -> dict:
        """
        Compute buffer size and contributing components using 95% (one-sided) confidence.
        Returns a dict so we can log the breakdown without recalculating.
        """
        # Expectation is 40ms frames (25 FPS) and 80ms TTS/MuseTalk processing chunks
        target_frame_ms = 40.0
        tts_chunk_ms = 80.0

        tts_mean, tts_std = self._calc_stats(self._tts_rtf)
        net_mean, net_std = self._calc_stats(self._network_latency_ms)
        musetalk_mean, musetalk_std = self._calc_stats(self._musetalk_fps)

        # Fallbacks if we have no data yet
        if musetalk_mean == 0:
            musetalk_mean = 25.0  # default target FPS
        if musetalk_std == 0 and musetalk_mean > 0:
            musetalk_std = 1.0  # small jitter baseline

        worst_case_rtf = max(0.0, tts_mean + self.k_std * tts_std) if tts_mean > 0 else 0.0
        tts_overage_ms = 0.0
        if worst_case_rtf > 0:
            tts_overage_ms = max(0.0, (tts_chunk_ms * worst_case_rtf) - tts_chunk_ms)

        # Lower FPS is worse, so subtract std
        worst_case_fps = max(0.0, musetalk_mean - self.k_std * musetalk_std) if musetalk_mean > 0 else 0.0
        musetalk_overage_ms = 0.0
        if worst_case_fps > 0:
            musetalk_frame_ms = 1000.0 / worst_case_fps
            musetalk_overage_ms = max(0.0, (musetalk_frame_ms - target_frame_ms) * 2)  # 2 frames per 80ms audio

        # Always keep a modest baseline for network/scheduling jitter even if we lack stats
        network_buffer_ms = max(0.0, net_mean + self.k_std * net_std) if net_mean > 0 else 40.0

        auto_buffer_ms = max(0.0, tts_overage_ms + musetalk_overage_ms + network_buffer_ms)

        buffer_ms = self._manual_buffer_ms if self._manual_buffer_ms is not None else auto_buffer_ms
        buffer_source = "manual" if self._manual_buffer_ms is not None else "adaptive"

        return {
            "buffer_ms": buffer_ms,
            "buffer_source": buffer_source,
            "tts_overage_ms": tts_overage_ms,
            "worst_case_rtf": worst_case_rtf,
            "musetalk_overage_ms": musetalk_overage_ms,
            "worst_case_fps": worst_case_fps,
            "network_buffer_ms": network_buffer_ms,
            "tts_mean": tts_mean,
            "tts_std": tts_std,
            "musetalk_mean": musetalk_mean,
            "musetalk_std": musetalk_std,
            "network_mean": net_mean,
            "network_std": net_std,
            "manual_buffer_ms": self._manual_buffer_ms,
        }

    def calculate_optimal_buffer_ms(self) -> float:
        """Calculate optimal buffer size in milliseconds."""
        with self._lock:
            components = self._compute_buffer_components()
            if components["buffer_ms"] <= 0:
                logger.warning(
                    "Adaptive buffer computed as 0ms; check calibration data. "
                    f"Components: {components}"
                )
            return components["buffer_ms"]
    
    def calculate_optimal_frame_buffer(self) -> int:
        """
        Calculate optimal number of video frames to buffer.
        
        Returns:
            Number of frames to buffer before starting playback
        """
        buffer_ms = self.calculate_optimal_buffer_ms()
        return max(0, int(math.ceil(buffer_ms / 40.0)))  # 40ms per frame @25fps
    
    def set_calibration_results(
        self,
        llm_tokens_per_sec: float,
        tts_rtf: float,
        musetalk_fps: float,
        llm_samples: Optional[List[float]] = None,
        tts_samples: Optional[List[float]] = None,
        musetalk_samples: Optional[List[float]] = None,
    ):
        """Set calibration results from initial measurement and seed rolling windows."""
        with self._lock:
            if llm_samples is None and llm_tokens_per_sec > 0:
                llm_samples = [llm_tokens_per_sec]
            if tts_samples is None and tts_rtf > 0:
                tts_samples = [tts_rtf]
            if musetalk_samples is None and musetalk_fps > 0:
                musetalk_samples = [musetalk_fps]

            if llm_samples:
                self._llm_tokens_per_sec = llm_samples[-self.window_size:]
                self._calibration_llm_tokens_per_sec = sum(llm_samples) / len(llm_samples)
            if tts_samples:
                self._tts_rtf = tts_samples[-self.window_size:]
                self._calibration_tts_rtf = sum(tts_samples) / len(tts_samples)
            if musetalk_samples:
                self._musetalk_fps = musetalk_samples[-self.window_size:]
                self._calibration_musetalk_fps = sum(musetalk_samples) / len(musetalk_samples)

            if llm_samples or tts_samples or musetalk_samples:
                self._is_calibrated = True
    
    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated
    
    def get_buffer_config(self) -> dict:
        """Get current buffer configuration as a dictionary"""
        with self._lock:
            components = self._compute_buffer_components()
            llm_mean, llm_std = self._calc_stats(self._llm_tokens_per_sec)
            frame_buffer = max(0, int(math.ceil(components["buffer_ms"] / 40.0)))
            calibrated = self._is_calibrated
        
        return {
            "buffer_ms": round(components["buffer_ms"], 1),
            "frame_buffer": frame_buffer,
            "buffer_source": components["buffer_source"],
            "manual_buffer_ms": components["manual_buffer_ms"],
            "tts_rtf_mean": round(components["tts_mean"], 3),
            "tts_rtf_std": round(components["tts_std"], 3),
            "worst_case_rtf": round(components["worst_case_rtf"], 3),
            "musetalk_fps_mean": round(components["musetalk_mean"], 1),
            "musetalk_fps_std": round(components["musetalk_std"], 1),
            "worst_case_fps": round(components["worst_case_fps"], 1) if components["worst_case_fps"] else 0.0,
            "network_latency_mean_ms": round(components["network_mean"], 1),
            "network_latency_std_ms": round(components["network_std"], 1),
            "network_buffer_ms": round(components["network_buffer_ms"], 1),
            "tts_overage_ms": round(components["tts_overage_ms"], 1),
            "musetalk_overage_ms": round(components["musetalk_overage_ms"], 1),
            "llm_tokens_per_sec_mean": round(llm_mean, 2),
            "llm_tokens_per_sec_std": round(llm_std, 2),
            "k_std": self.k_std,
            "is_calibrated": calibrated,
        }


class TTSSession:
    """
    Manages a persistent TTS session with Triton.
    
    The key insight is that Triton's sequence management requires the SAME gRPC stream
    to be used for the entire sequence lifecycle (sequence_start -> data -> sequence_end).
    
    This class maintains that persistent stream and handles:
    - Session initialization (cache allocation)
    - Streaming generation
    - Session cleanup
    """
    
    # Sessions older than this are considered stale and should be refreshed
    MAX_IDLE_SECONDS = 300  # 5 minutes
    
    def __init__(self, triton_url: str, session_id: int, tts_params: TTSParams):
        self.triton_url = triton_url
        self.session_id = session_id
        self.tts_params = tts_params
        
        self._client: Optional[grpc_client.InferenceServerClient] = None
        self._result_queue: Optional[queue.Queue] = None
        self._is_initialized = False
        self._is_closed = False
        self._lock = threading.Lock()
        self._last_used: float = time.time()  # Track last usage time
        
    @property
    def is_stale(self) -> bool:
        """Check if session has been idle too long and may be invalid."""
        return (time.time() - self._last_used) > self.MAX_IDLE_SECONDS
    
    @property
    def idle_seconds(self) -> float:
        """Get seconds since last use."""
        return time.time() - self._last_used
        
    def _callback(self, result, error):
        """Callback for stream responses"""
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
                
                # Send init request
                inputs = [
                    grpc_client.InferInput("START", [1], "BOOL"),
                    grpc_client.InferInput("CORRID", [1], "INT64"),
                ]
                inputs[0].set_data_from_numpy(np.array([True], dtype=bool))
                inputs[1].set_data_from_numpy(np.array([self.session_id], dtype=np.int64))
                
                outputs = [grpc_client.InferRequestedOutput("AUDIO_FRAME")]
                
                self._client.async_stream_infer(
                    model_name="tts",
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
                    logger.info(f"TTS session {self.session_id} initialized successfully")
                    return True
                    
                except queue.Empty:
                    logger.warning(f"Timeout waiting for TTS init response for session {self.session_id}")
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
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        decoder_temperature: Optional[float] = None,
        decoder_top_p: Optional[float] = None,
        on_audio: Optional[Callable[[np.ndarray, str, TTSMetrics], None]] = None,
    ) -> Generator[Tuple[np.ndarray, str, TTSMetrics], None, None]:
        """
        Generate audio from text chunks on this session.
        
        Args:
            text_chunks: Pre-split text chunks for streaming TTS
            temperature: Backbone temperature override
            top_p: Backbone top_p override
            decoder_temperature: Depth decoder temperature override  
            decoder_top_p: Depth decoder top_p override
            on_audio: Callback for each audio chunk
            
        Yields:
            Tuple of (audio_array, word, metrics)
        """
        with self._lock:
            if self._is_closed:
                logger.error(f"Cannot generate on closed session {self.session_id}")
                return
            
            if not self._is_initialized:
                logger.error(f"Session {self.session_id} not initialized, cannot generate")
                return
            
            # Update last used time
            self._last_used = time.time()
        
        metrics = TTSMetrics()
        start_time = time.time()
        total_samples = 0
        
        # Build word list for tracking
        all_words = []
        for i, chunk in enumerate(text_chunks):
            stripped = chunk.strip()
            if stripped:
                if i == 0:
                    all_words.extend(stripped.split())
                else:
                    all_words.append(stripped)
        
        logger.info(f"TTS session {self.session_id}: generating {len(all_words)} words from {len(text_chunks)} chunks")
        
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
                
                if temperature is not None:
                    inp = grpc_client.InferInput("BACKBONE_TEMPERATURE", [1], "FP32")
                    inp.set_data_from_numpy(np.array([temperature], dtype=np.float32))
                    inputs.append(inp)
                if top_p is not None:
                    inp = grpc_client.InferInput("BACKBONE_TOP_P", [1], "FP32")
                    inp.set_data_from_numpy(np.array([top_p], dtype=np.float32))
                    inputs.append(inp)
                if decoder_temperature is not None:
                    inp = grpc_client.InferInput("DEPTH_TEMPERATURE", [1], "FP32")
                    inp.set_data_from_numpy(np.array([decoder_temperature], dtype=np.float32))
                    inputs.append(inp)
                if decoder_top_p is not None:
                    inp = grpc_client.InferInput("DEPTH_TOP_P", [1], "FP32")
                    inp.set_data_from_numpy(np.array([decoder_top_p], dtype=np.float32))
                    inputs.append(inp)
                
                outputs = [grpc_client.InferRequestedOutput("AUDIO_FRAME")]
                
                self._client.async_stream_infer(
                    model_name="tts",
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
                            logger.debug(f"Chunk {i} complete, generated {chunk_audio_count} audio frames")
                            break
                        
                        audio = data.as_numpy("AUDIO_FRAME")
                        
                        if len(audio) > 0:
                            chunk_audio_count += 1
                            total_samples += len(audio)
                            
                            # Update metrics
                            elapsed = time.time() - start_time
                            metrics.generation_time_ms = elapsed * 1000
                            metrics.audio_duration_ms = (total_samples / self.tts_params.target_sample_rate) * 1000
                            metrics.rtf = elapsed / (total_samples / self.tts_params.target_sample_rate) if total_samples > 0 else 0
                            
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
                
                # Send end request
                inputs = [
                    grpc_client.InferInput("END", [1], "BOOL"),
                    grpc_client.InferInput("CORRID", [1], "INT64"),
                ]
                inputs[0].set_data_from_numpy(np.array([True], dtype=bool))
                inputs[1].set_data_from_numpy(np.array([self.session_id], dtype=np.int64))
                
                outputs = [grpc_client.InferRequestedOutput("AUDIO_FRAME")]
                
                self._client.async_stream_infer(
                    model_name="tts",
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
                        logger.warning(f"TTS session {self.session_id} end received error: {data}")
                    else:
                        logger.info(f"TTS session {self.session_id} ended successfully")
                        
                except queue.Empty:
                    logger.warning(f"Timeout waiting for TTS session {self.session_id} end confirmation")
                
            except Exception as e:
                logger.error(f"Error ending TTS session {self.session_id}: {e}")
            finally:
                self._cleanup_stream()
                self._is_closed = True
                self._is_initialized = False
            
            return True
    
    def _cleanup_stream(self):
        """Clean up the gRPC stream"""
        if self._client is not None:
            try:
                self._client.stop_stream()
            except Exception as e:
                logger.debug(f"Error stopping stream: {e}")
            self._client = None
        self._result_queue = None
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    @property
    def is_closed(self) -> bool:
        return self._is_closed
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# MuseTalkSession class removed - MuseTalk is now stateless
# Use TritonVoiceClient.generate_musetalk_frames() directly


class TritonVoiceClient:
    """Client for the Voice Assistant Triton pipeline"""
    
    def __init__(
        self,
        triton_url: str = "localhost:8001",
        vad_params: Optional[VADParams] = None,
        llm_params: Optional[LLMParams] = None,
        tts_params: Optional[TTSParams] = None,
        musetalk_params: Optional[MuseTalkParams] = None,
    ):
        self.triton_url = triton_url
        self.vad_params = vad_params or VADParams()
        self.llm_params = llm_params or LLMParams()
        self.tts_params = tts_params or TTSParams()
        self.musetalk_params = musetalk_params or MuseTalkParams()
        
        # Create Triton client
        self.client = grpc_client.InferenceServerClient(url=triton_url)
        
        # VAD state
        self.vad_sample_rate = 16000
        self.vad_chunk_samples = 512
        self.speech_start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None
        self.accumulated_audio: List[np.ndarray] = []
        self.is_speaking = False
        
        # TTS session management
        self._tts_session_counter = 100
        self._tts_session_lock = threading.Lock()
        self._active_tts_sessions: dict[int, TTSSession] = {}
        
        # Streaming metrics for adaptive buffering
        self.streaming_metrics = StreamingMetrics()
        
        logger.info(f"TritonVoiceClient initialized with URL: {triton_url}")
    
    def check_health(self) -> bool:
        """Check if Triton server is healthy"""
        try:
            return self.client.is_server_live() and self.client.is_server_ready()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def check_models_ready(self) -> dict:
        """Check if all models are ready"""
        models = ["vad", "stt", "llm", "tts", "musetalk"]
        status = {}
        for model in models:
            try:
                status[model] = self.client.is_model_ready(model)
            except Exception as e:
                logger.error(f"Model {model} check failed: {e}")
                status[model] = False
        return status

    # =========== Calibration Methods ===========
    
    def calibrate_generation_speed(
        self,
        test_prompt: str = "გამარჯობა, როგორ ხარ?",
        tts_session_id: Optional[int] = None,
        musetalk_session_id: Optional[int] = None,
        num_samples: int = 3,
    ) -> dict:
        """
        Calibrate generation speeds for adaptive buffering.
        
        Measures the actual generation speed of LLM, TTS, and MuseTalk
        to calculate optimal buffer sizes. Excludes initialization time.
        
        Args:
            test_prompt: Short test prompt for LLM
            tts_session_id: TTS session to use (must be already initialized)
            musetalk_session_id: MuseTalk session to use (must be already initialized)
            num_samples: Number of samples to collect for each measurement
            
        Returns:
            Dictionary with calibration results and recommended buffer config
        """
        logger.info(f"Starting generation speed calibration (samples={num_samples})...")
        
        results = {
            "llm_tokens_per_sec": 0.0,
            "llm_tokens_per_sec_std": 0.0,
            "tts_rtf": 0.0,
            "tts_rtf_std": 0.0,
            "musetalk_fps": 0.0,
            "musetalk_fps_std": 0.0,
            "samples_collected": 0,
            "buffer_config": {},
        }
        
        llm_samples: List[float] = []
        tts_samples: List[float] = []
        musetalk_samples: List[float] = []
        
        def _mean_std(samples: List[float]) -> Tuple[float, float]:
            if not samples:
                return 0.0, 0.0
            mean = sum(samples) / len(samples)
            if len(samples) < 2:
                return mean, 0.0
            variance = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
            return mean, variance ** 0.5
        
        try:
            # Calibrate LLM speed
            logger.info(f"Calibrating LLM generation speed with prompt '{test_prompt}'")
            for i in range(num_samples):
                token_count = 0
                start_time = None
                
                for token in self.generate_llm_stream(test_prompt):
                    if start_time is None:
                        # Skip initialization time - start timing from first token
                        start_time = time.time()
                    token_count += 1
                
                if start_time and token_count > 1:
                    duration = time.time() - start_time
                    tokens_per_sec = (token_count - 1) / duration if duration > 0 else 0
                    if tokens_per_sec > 0:
                        llm_samples.append(tokens_per_sec)
                        logger.info(
                            f"[Calibration] LLM sample {i+1}: {token_count} tokens in {duration*1000:.0f}ms "
                            f"({tokens_per_sec:.1f} tok/s)"
                        )
            
            results["llm_tokens_per_sec"], results["llm_tokens_per_sec_std"] = _mean_std(llm_samples)
            
            # Calibrate TTS speed (if session provided)
            if tts_session_id is not None:
                logger.info("Calibrating TTS generation speed...")
                test_texts = ["გამარჯობა", "როგორ ხარ", "კარგად ვარ"]
                
                for i, text in enumerate(test_texts[:num_samples]):
                    final_metrics: Optional[TTSMetrics] = None
                    for _, _, metrics in self.generate_tts_stream([text], session_id=tts_session_id):
                        if metrics.audio_duration_ms > 0 and metrics.generation_time_ms > 0:
                            final_metrics = metrics
                    
                    if final_metrics and final_metrics.audio_duration_ms > 0:
                        rtf = final_metrics.generation_time_ms / final_metrics.audio_duration_ms
                        tts_samples.append(rtf)
                        logger.info(
                            f"[Calibration] TTS sample {i+1} '{text}': "
                            f"{final_metrics.audio_duration_ms:.1f}ms audio in "
                            f"{final_metrics.generation_time_ms:.1f}ms (RTF {rtf:.3f})"
                        )
            
            results["tts_rtf"], results["tts_rtf_std"] = _mean_std(tts_samples)
            
            # Calibrate MuseTalk speed (if session provided)
            if musetalk_session_id is not None:
                logger.info("Calibrating MuseTalk generation speed...")
                # Generate some test audio chunks
                chunk_size = 1920  # 80ms at 24kHz
                test_audio = np.zeros(chunk_size, dtype=np.float32)
                
                for i in range(num_samples):
                    start_time = time.time()
                    frame_count = 0
                    
                    for frame_bytes, frame_idx, timestamp_ms, metrics in self.send_musetalk_audio(
                        test_audio, session_id=musetalk_session_id
                    ):
                        frame_count += 1
                    
                    duration = time.time() - start_time
                    if duration > 0 and frame_count > 0:
                        fps = frame_count / duration
                        musetalk_samples.append(fps)
                        logger.info(
                            f"[Calibration] MuseTalk sample {i+1}: "
                            f"{frame_count} frames in {duration*1000:.0f}ms ({fps:.1f} FPS)"
                        )
            
            results["musetalk_fps"], results["musetalk_fps_std"] = _mean_std(musetalk_samples)
            
            # Update streaming metrics with calibration results
            results["samples_collected"] = len(llm_samples) + len(tts_samples) + len(musetalk_samples)
            
            self.streaming_metrics.set_calibration_results(
                llm_tokens_per_sec=results["llm_tokens_per_sec"],
                tts_rtf=results["tts_rtf"],
                musetalk_fps=results["musetalk_fps"] if results["musetalk_fps"] > 0 else 25.0,
                llm_samples=llm_samples,
                tts_samples=tts_samples,
                musetalk_samples=musetalk_samples if musetalk_samples else [25.0],
            )
            
            results["buffer_config"] = self.streaming_metrics.get_buffer_config()
            logger.info(
                "Calibration complete. "
                f"RTF {results['tts_rtf']:.3f}±{results['tts_rtf_std']:.3f}, "
                f"MuseTalk {results['musetalk_fps']:.1f}±{results['musetalk_fps_std']:.1f} FPS, "
                f"LLM {results['llm_tokens_per_sec']:.1f} tok/s"
            )
            logger.info(
                "Buffer breakdown (95%%): "
                f"buffer {results['buffer_config']['buffer_ms']:.0f}ms "
                f"({results['buffer_config'].get('buffer_source','adaptive')}) | "
                f"TTS overage {results['buffer_config'].get('tts_overage_ms', 0):.0f}ms | "
                f"MuseTalk overage {results['buffer_config'].get('musetalk_overage_ms', 0):.0f}ms | "
                f"network {results['buffer_config'].get('network_buffer_ms', 0):.0f}ms"
            )
            
        except Exception as e:
            logger.error(f"Calibration error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return results
    
    def get_buffer_config(self) -> dict:
        """Get current adaptive buffer configuration"""
        return self.streaming_metrics.get_buffer_config()
    
    def set_manual_buffer_ms(self, buffer_ms: Optional[float]):
        """Manually override adaptive buffer size (ms). Pass None to return to adaptive."""
        self.streaming_metrics.set_manual_buffer_ms(buffer_ms)

    # =========== VAD Methods ===========
    
    def process_vad_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, float, bool]:
        """Process a single audio chunk through VAD"""
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        inputs = [grpc_client.InferInput("AUDIO_PCM", audio_chunk.shape, "FP32")]
        inputs[0].set_data_from_numpy(audio_chunk)
        
        outputs = [
            grpc_client.InferRequestedOutput("IS_SPEECH"),
            grpc_client.InferRequestedOutput("PROB"),
            grpc_client.InferRequestedOutput("END_OF_UTTERANCE")
        ]
        
        result = self.client.infer("vad", inputs, outputs=outputs)
        
        is_speech = bool(result.as_numpy("IS_SPEECH")[0])
        prob = float(result.as_numpy("PROB")[0])
        end_of_utt = bool(result.as_numpy("END_OF_UTTERANCE")[0])
        
        return is_speech, prob, end_of_utt
    
    def process_vad_with_state(self, audio_chunk: np.ndarray, current_time_ms: float) -> Tuple[str, Optional[np.ndarray]]:
        """Process VAD with state management"""
        is_speech, prob, _ = self.process_vad_chunk(audio_chunk)
        
        if is_speech:
            if not self.is_speaking:
                self.speech_start_time = current_time_ms
                self.accumulated_audio = []
            
            self.is_speaking = True
            self.last_speech_time = current_time_ms
            self.accumulated_audio.append(audio_chunk)
            
            speech_duration = current_time_ms - self.speech_start_time
            if speech_duration >= self.vad_params.speech_threshold_ms:
                return "speaking", None
            return "listening", None
        else:
            if self.is_speaking:
                silence_duration = current_time_ms - self.last_speech_time
                self.accumulated_audio.append(audio_chunk)
                
                if silence_duration >= self.vad_params.silence_threshold_ms:
                    speech_duration = self.last_speech_time - self.speech_start_time
                    
                    if speech_duration >= self.vad_params.speech_threshold_ms:
                        complete_audio = np.concatenate(self.accumulated_audio)
                        self.reset_vad_state()
                        return "utterance_complete", complete_audio
                    
                    self.reset_vad_state()
                    return "listening", None
                return "speaking", None
            return "listening", None
    
    def reset_vad_state(self):
        """Reset VAD state machine"""
        self.speech_start_time = None
        self.last_speech_time = None
        self.accumulated_audio = []
        self.is_speaking = False
    
    # =========== STT Methods ===========
    
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text using STT model"""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        inputs = [grpc_client.InferInput("AUDIO_PCM", audio.shape, "FP32")]
        inputs[0].set_data_from_numpy(audio)
        
        outputs = [grpc_client.InferRequestedOutput("TRANSCRIPT")]
        
        result = self.client.infer("stt", inputs, outputs=outputs)
        transcript = result.as_numpy("TRANSCRIPT")[0]
        
        if isinstance(transcript, bytes):
            transcript = transcript.decode("utf-8")
        
        return transcript.strip()
    
    # =========== LLM Methods ===========
    
    def build_prompt(self, user_message: str, conversation_history: List[dict] = None) -> str:
        """Build chat prompt with system message and history"""
        messages = [{"role": "system", "content": self.llm_params.system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_message})
        
        prompt_parts = ["<s>"]
        for msg in messages:
            prompt_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        prompt_parts.append("<|im_start|>assistant\n")
        
        return "".join(prompt_parts)
    
    def generate_llm_stream(self, prompt: str, on_token: Optional[Callable[[str], None]] = None) -> Generator[str, None, None]:
        """Stream LLM generation token by token"""
        prompt_bytes = np.array([prompt.encode("utf-8")], dtype=object)

        inputs = [
            grpc_client.InferInput("PROMPT", [1], "BYTES"),
            grpc_client.InferInput("MAX_NEW_TOKENS", [1], "INT32"),
            grpc_client.InferInput("TEMPERATURE", [1], "FP32"),
            grpc_client.InferInput("TOP_P", [1], "FP32"),
        ]

        inputs[0].set_data_from_numpy(prompt_bytes)
        inputs[1].set_data_from_numpy(np.array([self.llm_params.max_new_tokens], dtype=np.int32))
        inputs[2].set_data_from_numpy(np.array([self.llm_params.temperature], dtype=np.float32))
        inputs[3].set_data_from_numpy(np.array([self.llm_params.top_p], dtype=np.float32))

        outputs = [
            grpc_client.InferRequestedOutput("TEXT_CHUNK"),
            grpc_client.InferRequestedOutput("FINISHED"),
        ]

        full_response = ""
        result_queue = queue.Queue()
        stream_done = threading.Event()
        
        # Track generation timing for metrics (excluding initialization)
        first_token_time = None
        token_count = 0

        def callback(result, error):
            if error:
                result_queue.put(("error", str(error)))
                stream_done.set()
            elif result:
                result_queue.put(("result", result))
            else:
                stream_done.set()

        stream_client = grpc_client.InferenceServerClient(url=self.triton_url)
        stream_client.start_stream(callback=callback)

        try:
            stream_client.async_stream_infer(
                model_name="llm",
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
                                # Start timing from first token (excludes initialization)
                                if first_token_time is None:
                                    first_token_time = time.time()
                                token_count += 1
                                
                                full_response += chunk
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
            
            # Record LLM metrics for adaptive buffering (excluding first token as it includes init time)
            if first_token_time is not None and token_count > 1:
                duration = time.time() - first_token_time
                self.streaming_metrics.record_llm_generation(token_count - 1, duration)
                tokens_per_sec = (token_count - 1) / duration if duration > 0 else 0
                logger.info(
                    f"LLM stream completed: {token_count} tokens in {duration*1000:.0f}ms "
                    f"({tokens_per_sec:.1f} tok/s)"
                )

        logger.info(f"LLM generation complete: {len(full_response)} chars")
    
    # =========== TTS Methods ===========
    
    def _get_next_session_id(self) -> int:
        """Get a unique session ID for TTS"""
        with self._tts_session_lock:
            self._tts_session_counter += 1
            return self._tts_session_counter
    
    def _split_text_for_streaming(self, text: str) -> List[str]:
        """
        Split text for TTS streaming with 2-word lookahead.
        
        The TTS model generates audio for word[i-2] when receiving word[i].
        
        Example: "გამარჯობა! როგორ შემიძლია დაგეხმაროთ დღეს?"
        Returns: ["გამარჯობა! როგორ შემიძლია", " დაგეხმაროთ", " დღეს?", "", ""]
        
        Generation sequence:
        - Send chunk[0] ("გამარჯობა! როგორ შემიძლია") → generates "გამარჯობა!"
        - Send chunk[1] (" დაგეხმაროთ") → generates "როგორ"
        - Send chunk[2] (" დღეს?") → generates "შემიძლია"
        - Send chunk[3] ("") → generates "დაგეხმაროთ"
        - Send chunk[4] ("") → generates "დღეს?"
        """
        text = text.replace("\n", " ").strip()
        words = text.split()
        
        if not words:
            return ["", ""]
        
        if len(words) <= 3:
            return [text, "", ""]
        
        result = [' '.join(words[:3])]
        for w in words[3:]:
            result.append(' ' + w)
        result.extend(["", ""])
        
        return result
    
    def create_tts_session(self) -> TTSSession:
        """
        Create a new TTS session with a unique ID.
        
        Returns:
            A new TTSSession instance (not yet initialized)
        """
        session_id = self._get_next_session_id()
        session = TTSSession(self.triton_url, session_id, self.tts_params)
        
        with self._tts_session_lock:
            self._active_tts_sessions[session_id] = session
        
        return session
    
    def get_tts_session(self, session_id: int) -> Optional[TTSSession]:
        """Get an existing TTS session by ID"""
        with self._tts_session_lock:
            return self._active_tts_sessions.get(session_id)
    
    def is_tts_session_stale(self, session_id: int) -> bool:
        """
        Check if a TTS session is stale (idle for too long).
        
        Args:
            session_id: The session ID to check
            
        Returns:
            True if session is stale or doesn't exist
        """
        session = self.get_tts_session(session_id)
        if session is None:
            return True
        if session.is_closed:
            return True
        if not session.is_initialized:
            return True
        if session.is_stale:
            logger.warning(f"TTS session {session_id} is stale (idle for {session.idle_seconds:.1f}s)")
            return True
        return False
    
    def close_tts_session(self, session_id: int) -> bool:
        """
        Close and cleanup a TTS session.
        
        Args:
            session_id: The session ID to close
            
        Returns:
            True if closed successfully
        """
        with self._tts_session_lock:
            session = self._active_tts_sessions.pop(session_id, None)
        
        if session is not None:
            return session.close()
        return True
    
    def init_tts_session(self, session_id: int) -> bool:
        """
        Initialize TTS Session and kv cache.
        
        This creates a new TTSSession and initializes it.
        The session remains open and ready for generation.
        
        Args:
            session_id: Unique session identifier for cache allocation
            
        Returns:
            True if cache initialized successfully
        """
        # Check if there's an existing session and close it first
        existing = self.get_tts_session(session_id)
        if existing is not None:
            logger.info(f"Closing existing TTS session {session_id} before reinitializing")
            existing.close()
            with self._tts_session_lock:
                self._active_tts_sessions.pop(session_id, None)
        
        # Create new session
        session = TTSSession(self.triton_url, session_id, self.tts_params)
        success = session.initialize()
        
        if success:
            with self._tts_session_lock:
                self._active_tts_sessions[session_id] = session
        
        return success
    
    def end_tts_session(self, session_id: int) -> bool:
        """
        End a TTS session and release resources.
        
        Args:
            session_id: The session ID to end
            
        Returns:
            True if ended successfully
        """
        return self.close_tts_session(session_id)

    def generate_tts_stream(
        self,
        text: List[str],
        session_id: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        decoder_temperature: Optional[float] = None,
        decoder_top_p: Optional[float] = None,
        on_audio: Optional[Callable[[np.ndarray, str, TTSMetrics], None]] = None,
        
    ) -> Generator[Tuple[np.ndarray, str, TTSMetrics], None, None]:
        """
        Stream TTS generation with word-level synchronization.
        
        The TTS model uses 2-word lookahead:
        - When sending text chunk[i], the model generates audio for words that were sent 2 chunks ago
        - First 3 words are sent together, then individual words with leading space
        - Two empty strings at end to flush the remaining 2 words
        
        Args:
            text: Text to synthesize. It is already correctly splitted text. It can be intermediate chunks too!
            session_id: session ID to use (must be initialized first via init_tts_session)
            temperature: Backbone temperature override
            top_p: Backbone top_p override
            decoder_temperature: Depth decoder temperature override
            decoder_top_p: Depth decoder top_p override
            on_audio: Optional callback for audio chunks
        """
        session = self.get_tts_session(session_id)
        
        if session is None:
            logger.error(f"[TTS_STREAM] Session {session_id} not found. Did you call init_tts_session first?")
            return
        
        if not session.is_initialized:
            logger.error(f"[TTS_STREAM] Session {session_id} not initialized. is_initialized={session.is_initialized}")
            return
        
        if session.is_closed:
            logger.error(f"[TTS_STREAM] Session {session_id} is already closed. is_closed={session.is_closed}")
            return
        
        logger.info(f"[TTS_STREAM] Session {session_id}: text='{' '.join(text)}', chunks={len(text)}")
        
        # Wrap generator to record metrics
        for audio, word, metrics in session.generate(
            text_chunks=text,
            temperature=temperature,
            top_p=top_p,
            decoder_temperature=decoder_temperature,
            decoder_top_p=decoder_top_p,
            on_audio=on_audio,
        ):
            # Record TTS metrics for adaptive buffering
            if metrics.audio_duration_ms > 0 and metrics.generation_time_ms > 0:
                self.streaming_metrics.record_tts_generation(
                    audio_duration_ms=metrics.audio_duration_ms,
                    generation_time_ms=metrics.generation_time_ms
                )
            
            yield audio, word, metrics
        
        logger.info(f"[TTS_STREAM] Session {session_id} COMPLETE")
    
    def synthesize_text(self, text: str) -> Tuple[np.ndarray, TTSMetrics]:
        """
        Synthesize complete audio from text (non-streaming).
        
        This is a convenience method that handles the full session lifecycle:
        create -> init -> generate -> close
        """
        audio_chunks = []
        final_metrics = TTSMetrics()
        
        # Split text into chunks
        chunks = self._split_text_for_streaming(text)
        
        # Create and initialize session
        session_id = self._get_next_session_id()
        if not self.init_tts_session(session_id):
            logger.error("Failed to initialize TTS session for synthesize_text")
            return np.array([], dtype=np.float32), final_metrics
        
        try:
            for audio, word, metrics in self.generate_tts_stream(chunks, session_id=session_id):
                audio_chunks.append(audio)
                final_metrics = metrics
        finally:
            # Always close the session
            self.end_tts_session(session_id)
        
        if audio_chunks:
            return np.concatenate(audio_chunks), final_metrics
        
        return np.array([], dtype=np.float32), final_metrics
    
    # =========== MuseTalk Methods (Stateless) ===========
    
    def generate_musetalk_frames(
        self,
        audio: np.ndarray,
        frame_index: int = 0,
        on_frame: Optional[Callable[[bytes, int, float, MuseTalkMetrics], None]] = None,
        timeout: float = 60.0,
    ) -> Generator[Tuple[bytes, int, float, MuseTalkMetrics], None, None]:
        """
        Generate video frames from audio (stateless).
        
        Sends complete audio to MuseTalk and receives all generated frames.
        The model is stateless - each call processes the audio independently.
        
        Args:
            audio: Complete audio samples at 24kHz as float32
            frame_index: Starting frame index in avatar cycle (for video continuity)
            on_frame: Optional callback for each frame
            timeout: Timeout for waiting for frames
            
        Yields:
            Tuple of (frame_jpeg_bytes, frame_index, timestamp_ms, metrics)
        """
        if audio is None or len(audio) == 0:
            logger.error("MuseTalk: Audio is required and must not be empty")
            return
        
        audio_duration_s = len(audio) / 24000  # 24kHz
        logger.info(f"MuseTalk: Processing {audio_duration_s:.3f}s audio, start_frame_index={frame_index}")
        
        metrics = MuseTalkMetrics()
        start_time = time.time()
        frames_received = 0
        
        result_queue: queue.Queue = queue.Queue()
        
        def callback(result, error):
            if error:
                result_queue.put(("error", error))
            else:
                result_queue.put(("result", result))
        
        try:
            # Create client and start stream for this request
            client = grpc_client.InferenceServerClient(url=self.triton_url)
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
                model_name="musetalk",
                inputs=inputs,
                outputs=outputs,
                enable_empty_final_response=True,
            )
            
            # Collect all frames
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
                        
                        # Update metrics
                        elapsed = time.time() - start_time
                        metrics.frame_index = output_frame_index
                        metrics.timestamp_ms = timestamp_ms
                        metrics.frames_generated = frames_received
                        metrics.generation_time_ms = elapsed * 1000
                        
                        if on_frame:
                            on_frame(frame_bytes, output_frame_index, timestamp_ms, metrics)
                        
                        yield frame_bytes, output_frame_index, timestamp_ms, metrics
                    
                    if is_final:
                        break
                        
                except queue.Empty:
                    logger.warning("Timeout waiting for MuseTalk frame")
                    break
            
            # Record metrics for adaptive buffering
            if frames_received > 0:
                duration = time.time() - start_time
                if duration > 0:
                    self.streaming_metrics.record_musetalk_generation(frames_received, duration)
            
            logger.info(f"MuseTalk: Generated {frames_received} frames in {time.time() - start_time:.3f}s")
            
        except Exception as e:
            logger.error(f"MuseTalk generation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            try:
                client.stop_stream()
            except Exception:
                pass

    def get_musetalk_idle_frame(self, timeout: float = 30.0) -> Optional[bytes]:
        """
        Get a single idle frame from MuseTalk (stateless).
        
        Generates one frame with minimal silent audio to get the avatar's idle pose.
        This is used to display the avatar before any speech is generated.
        
        Args:
            timeout: Timeout for waiting for the frame
            
        Returns:
            JPEG bytes of the idle frame, or None if failed
        """
        # Generate minimal audio (240ms of silence - minimum required)
        # 24kHz * 0.24s = 5760 samples
        silent_audio = np.zeros(5760, dtype=np.float32)
        
        try:
            for frame_bytes, frame_idx, timestamp_ms, metrics in self.generate_musetalk_frames(
                audio=silent_audio,
                frame_index=0,
                timeout=timeout,
            ):
                # Return first frame only
                logger.info(f"MuseTalk: Got idle frame ({len(frame_bytes)} bytes)")
                return frame_bytes
        except Exception as e:
            logger.error(f"Failed to get MuseTalk idle frame: {e}")
        
        return None


class ConversationManager:
    """Manages conversation state and history"""
    
    def __init__(self, max_history: int = 10):
        self.history: List[dict] = []
        self.max_history = max_history
    
    def add_user_message(self, text: str):
        self.history.append({"role": "user", "content": text})
        self._trim_history()
    
    def add_assistant_message(self, text: str):
        self.history.append({"role": "assistant", "content": text})
        self._trim_history()
    
    def _trim_history(self):
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]
    
    def get_history(self) -> List[dict]:
        return self.history.copy()
    
    def clear(self):
        self.history = []


if __name__ == "__main__":
    client = TritonVoiceClient()
    print("Health check:", client.check_health())
    print("Models status:", client.check_models_ready())
    
    # Test text splitting
    test_text = "გამარჯობა! როგორ შემიძლია დაგეხმაროთ დღეს?"
    chunks = client._split_text_for_streaming(test_text)
    print(f"\nText: {test_text}")
    print(f"Chunks: {chunks}")
    
    # Show expected word generation order
    print("\nExpected generation sequence:")
    words = []
    for i, c in enumerate(chunks):
        s = c.strip()
        if s:
            if i == 0:
                words.extend(s.split())
            else:
                words.append(s)
    
    for i, chunk in enumerate(chunks):
        word = words[i] if i < len(words) else "(flush)"
        print(f"  Chunk {i}: '{chunk}' → generates audio for: '{word}'")
