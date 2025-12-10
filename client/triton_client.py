"""
Triton Client for Voice Assistant Pipeline
Handles VAD, STT, LLM, and TTS model interactions
"""

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
    
    def __init__(self, triton_url: str, session_id: int, tts_params: TTSParams):
        self.triton_url = triton_url
        self.session_id = session_id
        self.tts_params = tts_params
        
        self._client: Optional[grpc_client.InferenceServerClient] = None
        self._result_queue: Optional[queue.Queue] = None
        self._is_initialized = False
        self._is_closed = False
        self._lock = threading.Lock()
        
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


class TritonVoiceClient:
    """Client for the Voice Assistant Triton pipeline"""
    
    def __init__(
        self,
        triton_url: str = "localhost:8001",
        vad_params: Optional[VADParams] = None,
        llm_params: Optional[LLMParams] = None,
        tts_params: Optional[TTSParams] = None
    ):
        self.triton_url = triton_url
        self.vad_params = vad_params or VADParams()
        self.llm_params = llm_params or LLMParams()
        self.tts_params = tts_params or TTSParams()
        
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
        models = ["vad", "stt", "llm", "tts"]
        status = {}
        for model in models:
            try:
                status[model] = self.client.is_model_ready(model)
            except Exception as e:
                logger.error(f"Model {model} check failed: {e}")
                status[model] = False
        return status
    
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
            logger.error(f"TTS session {session_id} not found. Did you call init_tts_session first?")
            return
        
        if not session.is_initialized:
            logger.error(f"TTS session {session_id} not initialized")
            return
        
        if session.is_closed:
            logger.error(f"TTS session {session_id} is already closed")
            return
        
        logger.info(f"=" * 60)
        logger.info(f"TTS SESSION {session_id}")
        logger.info(f"Full text: '{' '.join(text)}'")
        logger.info(f"Split into {len(text)} chunks: {text}")
        logger.info(f"=" * 60)
        
        yield from session.generate(
            text_chunks=text,
            temperature=temperature,
            top_p=top_p,
            decoder_temperature=decoder_temperature,
            decoder_top_p=decoder_top_p,
            on_audio=on_audio,
        )
        
        logger.info(f"=" * 60)
        logger.info(f"TTS SESSION {session_id} COMPLETE")
        logger.info(f"=" * 60)
    
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
