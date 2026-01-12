"""
WebSocket handler and connection state management.

Handles:
- WebSocket connection lifecycle
- Message routing (text/binary)
- Connection state management
- Audio processing loop
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np
from fastapi import WebSocket

from config import AppConfig, StreamingConfig
from models import (
    VADStatus,
    MessageType,
    ConversationHistory,
    AVFrame,
)
from triton_services import TritonClient
from tts_service import TTSService
from pipeline import VoiceToVoicePipeline, StreamingMetricsManager, PipelineConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Connection State
# ============================================================================

@dataclass
class ConnectionState:
    """
    State for a single WebSocket connection.

    Tracks:
    - Connection lifecycle (connected, recording, generating)
    - TTS session management
    - VAD timing
    - Conversation history
    - Audio processing queue
    - Speculative processing state
    """
    websocket: WebSocket
    connection_id: str

    # Lifecycle flags
    is_connected: bool = True
    is_recording: bool = False
    is_generating: bool = False

    # Conversation
    conversation: ConversationHistory = field(default_factory=ConversationHistory)

    # TTS session
    tts_session_id: Optional[int] = None
    tts_session_ready: asyncio.Event = field(default_factory=asyncio.Event)

    # VAD timing
    vad_time_ms: float = 0.0

    # Audio processing
    audio_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=200))
    audio_processor_task: Optional[asyncio.Task] = None

    # MuseTalk frame tracking
    musetalk_frame_index: int = 0
    last_video_frame: Optional[bytes] = None  # Last generated video frame for idle display

    # Speculative processing state (early STT/LLM before full silence)
    speculative_task: Optional[asyncio.Task] = None
    speculative_stt_result: Optional[str] = None
    speculative_llm_result: Optional[str] = None
    speculative_cancelled: bool = False

    # Serialize websocket sends to preserve message order
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    # Key latency metric: time from utterance end to first AV frame
    utterance_end_time: Optional[float] = None  # When VAD detected utterance complete
    first_av_frame_sent: bool = False  # Track if we've sent first frame this generation


# ============================================================================
# Message Helpers
# ============================================================================

async def send_message(
    state_or_ws: ConnectionState | WebSocket,
    msg_type: str,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """Send a JSON message to the WebSocket."""
    try:
        send_lock = None
        if isinstance(state_or_ws, ConnectionState):
            if not state_or_ws.is_connected:
                return
            ws = state_or_ws.websocket
            send_lock = state_or_ws.send_lock
        else:
            ws = state_or_ws

        payload = {"type": msg_type}
        if data:
            payload.update(data)

        if send_lock is not None:
            async with send_lock:
                await ws.send_json(payload)
        else:
            await ws.send_json(payload)

    except Exception as exc:
        logger.error(f"Failed to send message '{msg_type}': {exc}")


# ============================================================================
# WebSocket Handler
# ============================================================================

class WebSocketHandler:
    """
    Handles WebSocket connections and message routing.

    Responsibilities:
    - Connection lifecycle management
    - Message type routing
    - Audio processing loop
    - Voice-to-voice pipeline coordination
    """

    def __init__(
        self,
        triton_client: TritonClient,
        tts_service: TTSService,
        pipeline: VoiceToVoicePipeline,
        config: AppConfig,
    ):
        self.triton_client = triton_client
        self.tts_service = tts_service
        self.pipeline = pipeline
        self.config = config

        self.active_connections: Dict[str, ConnectionState] = {}
        
        # Global last video frame - persists across connections for idle display
        self._global_last_video_frame: Optional[bytes] = None
        # Global frame index - continues avatar animation across connections
        self._global_frame_index: int = 0

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Handle a new WebSocket connection."""
        await websocket.accept()

        connection_id = f"conn_{int(time.time() * 1000)}"
        state = ConnectionState(
            websocket=websocket,
            connection_id=connection_id,
            musetalk_frame_index=self._global_frame_index,  # Continue from global index
        )
        self.active_connections[connection_id] = state

        logger.info(f"WebSocket connected: {connection_id}")

        try:
            # Send connection confirmation
            await send_message(state, "connected", {
                "connection_id": connection_id,
                "message": "Connected to Voice Assistant",
            })

            # Check MuseTalk availability and get idle frame
            musetalk_available = await self._check_musetalk_available()
            # Use global last frame (persists across connections) or connection-specific
            idle_frame = self._global_last_video_frame or state.last_video_frame

            # Only fetch initial idle frame if we don't have one from a previous generation
            if musetalk_available and idle_frame is None:
                loop = asyncio.get_event_loop()
                idle_frame = await loop.run_in_executor(
                    None,
                    self.triton_client.musetalk.get_idle_frame,
                )

            await send_message(state, "musetalk_ready", {
                "success": musetalk_available,
                "session_id": None,
                "idle_frame": base64.b64encode(idle_frame).decode("utf-8") if idle_frame else None,
            })

            # Start audio processor
            state.audio_processor_task = asyncio.create_task(
                self._audio_processor_loop(state)
            )

            # Message loop
            while True:
                message = await websocket.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                if "text" in message:
                    await self._handle_text_message(state, json.loads(message["text"]))

                elif "bytes" in message:
                    await self._handle_audio_data(state, message["bytes"])

        except Exception as exc:
            logger.error(f"WebSocket error: {exc}")
            import traceback
            logger.error(traceback.format_exc())

        finally:
            await self._cleanup_connection(state)

    async def _check_musetalk_available(self) -> bool:
        """Check if MuseTalk model is available."""
        try:
            models_status = self.triton_client.check_models_ready()
            return models_status.get("musetalk", False)
        except Exception as exc:
            logger.warning(f"Error checking MuseTalk availability: {exc}")
            return False

    async def _handle_audio_data(self, state: ConnectionState, audio_bytes: bytes) -> None:
        """Handle incoming audio data."""
        try:
            state.audio_queue.put_nowait(audio_bytes)
        except asyncio.QueueFull:
            # Drop oldest and add new
            try:
                state.audio_queue.get_nowait()
                state.audio_queue.put_nowait(audio_bytes)
            except asyncio.QueueEmpty:
                pass

    async def _handle_text_message(self, state: ConnectionState, data: Dict[str, Any]) -> None:
        """Handle incoming text message."""
        msg_type = data.get("type", "")

        if msg_type == "stop_generation":
            await self._handle_stop_generation(state)

        elif msg_type == "recording_start":
            await self._handle_recording_start(state)

        elif msg_type == "recording_stop":
            await self._handle_recording_stop(state)

    async def _handle_stop_generation(self, state: ConnectionState) -> None:
        """Handle stop generation request."""
        state.is_generating = False
        state.vad_time_ms = 0.0

        self.triton_client.vad.reset_state()

        await send_message(state, "vad_status", {"status": "listening"})
        await self._close_tts_session(state)

    async def _handle_recording_start(self, state: ConnectionState) -> None:
        """Handle recording start request."""
        state.is_recording = True
        state.vad_time_ms = 0.0

        self.triton_client.vad.reset_state()

        await send_message(state, "vad_status", {"status": "listening"})
        await self._init_tts_session(state)

    async def _handle_recording_stop(self, state: ConnectionState) -> None:
        """Handle recording stop request."""
        state.is_recording = False
        state.vad_time_ms = 0.0

        self.triton_client.vad.reset_state()

        # Clear audio queue
        while not state.audio_queue.empty():
            try:
                state.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        await send_message(state, "vad_status", {"status": "listening"})

        if not state.is_generating:
            await self._close_tts_session(state)

    async def _audio_processor_loop(self, state: ConnectionState) -> None:
        """Process incoming audio chunks."""
        buffer: List[np.ndarray] = []
        batch_size = self.config.streaming.audio_batch_size
        batch_timeout = self.config.streaming.audio_batch_timeout_s

        logger.info(f"[AUDIO_PROC] Started for {state.connection_id}")

        try:
            while state.is_connected:
                # Collect batch
                batch_start = time.time()
                while len(buffer) < batch_size:
                    remaining = batch_timeout - (time.time() - batch_start)
                    if remaining <= 0:
                        break
                    try:
                        audio_bytes = await asyncio.wait_for(
                            state.audio_queue.get(),
                            timeout=remaining,
                        )
                        audio = np.frombuffer(audio_bytes, dtype=np.float32)
                        buffer.append(audio)
                    except asyncio.TimeoutError:
                        break
                    except Exception as exc:
                        logger.error(f"Failed to decode audio: {exc}")

                if not buffer:
                    continue

                if not state.is_recording or state.is_generating:
                    buffer.clear()
                    continue

                combined_audio = np.concatenate(buffer)
                buffer.clear()

                state.vad_time_ms += len(combined_audio) / 16000.0 * 1000.0

                # Process through VAD
                loop = asyncio.get_event_loop()
                status, audio_data = await loop.run_in_executor(
                    None,
                    self.triton_client.vad.process_with_state,
                    combined_audio,
                    state.vad_time_ms,
                )

                # Log VAD status for debugging
                if status in {VADStatus.SPEAKING, VADStatus.EARLY_SILENCE, VADStatus.UTTERANCE_COMPLETE, VADStatus.SPEECH_RESUMED}:
                    logger.debug(f"[VAD_WS] Status: {status.value}, speculative_task={'running' if state.speculative_task else 'none'}")

                # Handle speech resumption - cancel speculative processing so we can re-run with more audio
                # Only cancel on actual SPEECH_RESUMED, not on SPEAKING (which is just "in utterance")
                if status == VADStatus.SPEECH_RESUMED and state.speculative_task is not None:
                    logger.info(f"[SPECULATIVE] Speech resumed, cancelling speculative (will re-trigger on next silence)")
                    state.speculative_cancelled = True
                    state.speculative_task.cancel()
                    state.speculative_task = None
                    state.speculative_stt_result = None
                    state.speculative_llm_result = None
                    # Don't send cancellation to frontend - it's internal re-triggering

                if status in {VADStatus.SPEAKING, VADStatus.SPEECH_START, VADStatus.SPEECH_CONTINUE, VADStatus.SPEECH_RESUMED, VADStatus.UTTERANCE_COMPLETE, VADStatus.EARLY_SILENCE}:
                    await send_message(state, "vad_status", {"status": status.value})

                # Handle early silence - start/restart speculative STT/LLM with latest audio
                if status == VADStatus.EARLY_SILENCE and audio_data is not None:
                    # Cancel any existing speculative task and start fresh with new audio
                    if state.speculative_task is not None:
                        state.speculative_task.cancel()
                        state.speculative_task = None
                    
                    logger.info(f"[SPECULATIVE] Early silence detected, starting speculative STT/LLM ({len(audio_data)} samples)")
                    state.speculative_cancelled = False
                    state.speculative_stt_result = None
                    state.speculative_llm_result = None
                    state.speculative_task = asyncio.create_task(
                        self._process_speculative_stt_llm(state, audio_data)
                    )

                # Handle full utterance completion
                if status == VADStatus.UTTERANCE_COMPLETE and audio_data is not None:
                    # Record utterance end time for latency tracking
                    state.utterance_end_time = time.time()
                    state.first_av_frame_sent = False
                    logger.info(f"[LATENCY] Utterance complete - starting pipeline")
                    
                    await self._process_voice_to_voice(state, audio_data)
                    state.vad_time_ms = 0.0
                    self.triton_client.vad.reset_state()
                    await send_message(state, "vad_status", {"status": "listening"})

        except asyncio.CancelledError:
            logger.info(f"[AUDIO_PROC] Cancelled for {state.connection_id}")
        except Exception as exc:
            logger.error(f"[AUDIO_PROC] Error: {exc}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            logger.info(f"[AUDIO_PROC] Stopped for {state.connection_id}")

    async def _process_speculative_stt_llm(
        self,
        state: ConnectionState,
        audio: np.ndarray,
    ) -> None:
        """
        Process audio speculatively through STT and LLM.
        Results are cached and used if utterance completes without more speech.
        """
        try:
            # Speculative STT
            logger.info(f"[SPECULATIVE] Starting STT on {len(audio)} samples")
            await send_message(state, "speculative_stt_start", {})
            
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(
                None,
                self.triton_client.stt.transcribe,
                audio,
            )
            
            if state.speculative_cancelled:
                logger.info(f"[SPECULATIVE] STT completed but cancelled, discarding")
                return
            
            state.speculative_stt_result = transcript
            logger.info(f"[SPECULATIVE] STT result: {transcript}")
            await send_message(state, "speculative_stt_complete", {"text": transcript})
            
            if not transcript.strip():
                return
            
            # Speculative LLM - start generating but don't use TTS yet
            logger.info(f"[SPECULATIVE] Starting LLM generation")
            await send_message(state, "speculative_llm_start", {})
            
            # Build prompt using existing history (don't add user message to history yet)
            prompt = self.triton_client.llm.build_prompt(
                transcript,
                state.conversation.get_history(),  # Current history without this message
            )
            
            # Run LLM in executor (it's synchronous)
            loop = asyncio.get_event_loop()
            llm_response = ""
            
            def run_llm():
                nonlocal llm_response
                for token in self.triton_client.llm.generate_stream(prompt):
                    if state.speculative_cancelled:
                        return
                    llm_response += token
                    # We can't easily send websocket messages from here
                    # Just collect the full response
            
            await loop.run_in_executor(None, run_llm)
            
            if state.speculative_cancelled:
                logger.info(f"[SPECULATIVE] LLM cancelled, discarding")
                return
            
            state.speculative_llm_result = llm_response
            logger.info(f"[SPECULATIVE] LLM complete: {len(llm_response)} chars")
            await send_message(state, "speculative_llm_complete", {"text": llm_response})
            
        except asyncio.CancelledError:
            logger.info(f"[SPECULATIVE] Task cancelled")
        except Exception as exc:
            logger.error(f"[SPECULATIVE] Error: {exc}")
            import traceback
            logger.error(traceback.format_exc())

    async def _process_voice_to_voice(
        self,
        state: ConnectionState,
        audio: np.ndarray,
    ) -> None:
        """Process complete utterance through the pipeline."""
        state.is_generating = True
        t_start = time.time()
        
        # Check if we have valid speculative results
        use_speculative = (
            state.speculative_task is not None and
            not state.speculative_cancelled and
            state.speculative_stt_result is not None
        )
        
        logger.info(f"[VOICE2VOICE] use_speculative={use_speculative}, "
                    f"task={state.speculative_task is not None}, "
                    f"cancelled={state.speculative_cancelled}, "
                    f"stt_result={state.speculative_stt_result is not None}, "
                    f"llm_result={state.speculative_llm_result is not None}")

        try:
            if use_speculative:
                # Wait for speculative task to complete if still running
                if not state.speculative_task.done():
                    logger.info(f"[VOICE2VOICE] Waiting for speculative task to complete")
                    try:
                        await asyncio.wait_for(state.speculative_task, timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"[VOICE2VOICE] Speculative task timed out, falling back to normal")
                        use_speculative = False
                    except asyncio.CancelledError:
                        use_speculative = False
            
            if use_speculative and state.speculative_stt_result:
                # Use speculative STT result
                transcript = state.speculative_stt_result
                t_stt = (time.time() - t_start) * 1000
                logger.info(f"[LATENCY] STT (speculative): {t_stt:.0f}ms - '{transcript}'")
                await send_message(state, "stt_start", {})
                await send_message(state, "stt_complete", {"text": transcript, "speculative": True})
            else:
                # Normal STT processing
                await send_message(state, "stt_start", {})
                loop = asyncio.get_event_loop()
                transcript = await loop.run_in_executor(
                    None,
                    self.triton_client.stt.transcribe,
                    audio,
                )
                t_stt = (time.time() - t_start) * 1000
                logger.info(f"[LATENCY] STT (normal): {t_stt:.0f}ms - '{transcript}'")
                await send_message(state, "stt_complete", {"text": transcript})

            if not transcript.strip():
                return

            state.conversation.add_user_message(transcript)

            # Check if we have speculative LLM result
            t_llm_start = time.time()
            if use_speculative and state.speculative_llm_result:
                t_llm = (time.time() - t_start) * 1000
                logger.info(f"[LATENCY] LLM (speculative): {t_llm:.0f}ms - {len(state.speculative_llm_result)} chars")
                # Use cached LLM result, just need to run TTS
                await self._process_tts_with_cached_llm(state, transcript, state.speculative_llm_result)
            else:
                # Normal LLM + TTS + MuseTalk
                await self._process_llm_and_tts(state, transcript)

        except Exception as exc:
            logger.error(f"Voice to voice error: {exc}")
            import traceback
            logger.error(traceback.format_exc())
            await send_message(state, "error", {"message": str(exc)})
        finally:
            state.is_generating = False
            # Clean up speculative state
            state.speculative_task = None
            state.speculative_stt_result = None
            state.speculative_llm_result = None
            state.speculative_cancelled = False

    async def _process_tts_with_cached_llm(
        self,
        state: ConnectionState,
        user_input: str,
        cached_llm_response: str,
    ) -> None:
        """Process TTS with pre-computed LLM result from speculative processing."""
        await send_message(state, "llm_start", {})
        
        # Send LLM tokens immediately (they're already computed)
        await send_message(state, "llm_token", {"token": cached_llm_response, "full_text": cached_llm_response})
        await send_message(state, "llm_complete", {"text": cached_llm_response})

        # Close any existing TTS session and init new one
        if state.tts_session_id is not None:
            await self._close_tts_session(state)

        await self._init_tts_session(state)

        try:
            await asyncio.wait_for(
                state.tts_session_ready.wait(),
                timeout=self.config.streaming.tts_init_timeout_s,
            )
        except asyncio.TimeoutError:
            await send_message(state, "error", {"message": "TTS session not ready"})
            return

        if state.tts_session_id is None:
            await send_message(state, "error", {"message": "TTS session not initialized"})
            return

        # Check video availability
        video_enabled = await self._check_musetalk_available()

        await send_message(state, "tts_start", {
            "text": "",
            "video_enabled": video_enabled,
        })

        # Create a simple generator that yields the entire cached response
        async def cached_llm_generator():
            # Yield the cached response as tokens (simulate streaming)
            # Split into words/chunks for more natural TTS processing
            words = cached_llm_response.split()
            for i, word in enumerate(words):
                if not state.is_generating:
                    break
                # Add space before word except for first
                token = (" " + word) if i > 0 else word
                yield token

        # Process through pipeline with cached LLM
        llm_response = await self.pipeline.process_llm_and_tts(
            llm_generator=cached_llm_generator(),
            tts_session_id=state.tts_session_id,
            video_enabled=video_enabled,
            base_frame_index=state.musetalk_frame_index,
            is_generating=lambda: state.is_generating,
            on_llm_token=lambda token, full: None,  # Already sent above
            on_av_frame=lambda frame: self._handle_av_frame(state, frame),
            on_error=lambda msg: asyncio.create_task(
                send_message(state, "error", {"message": msg})
            ),
            on_llm_complete=lambda text: None,  # Already sent above
            on_tts_complete=lambda: asyncio.create_task(self._on_tts_complete(state)),
        )

        # Update conversation
        state.conversation.add_assistant_message(llm_response)

        # Close TTS session
        if state.tts_session_id is not None:
            await self._close_tts_session(state)

    async def _process_llm_and_tts(
        self,
        state: ConnectionState,
        user_input: str,
    ) -> None:
        """Process user input through LLM, TTS, and MuseTalk."""
        # Build LLM prompt
        prompt = self.triton_client.llm.build_prompt(
            user_input,
            state.conversation.get_history()[:-1],  # Exclude current message
        )

        await send_message(state, "llm_start", {})

        # Close any existing TTS session and init new one
        if state.tts_session_id is not None:
            await self._close_tts_session(state)

        await self._init_tts_session(state)

        try:
            await asyncio.wait_for(
                state.tts_session_ready.wait(),
                timeout=self.config.streaming.tts_init_timeout_s,
            )
        except asyncio.TimeoutError:
            await send_message(state, "error", {"message": "TTS session not ready"})
            return

        if state.tts_session_id is None:
            await send_message(state, "error", {"message": "TTS session not initialized"})
            return

        # Check video availability
        video_enabled = await self._check_musetalk_available()

        await send_message(state, "tts_start", {
            "text": "",
            "video_enabled": video_enabled,
        })

        # Create async LLM generator
        async def llm_generator():
            loop = asyncio.get_event_loop()
            token_queue: asyncio.Queue = asyncio.Queue()
            done = False

            def llm_worker():
                nonlocal done
                try:
                    for token in self.triton_client.llm.generate_stream(prompt):
                        if not state.is_generating:
                            break
                        loop.call_soon_threadsafe(token_queue.put_nowait, ("token", token))
                finally:
                    done = True
                    loop.call_soon_threadsafe(token_queue.put_nowait, ("done", None))

            # Start LLM in thread
            loop.run_in_executor(None, llm_worker)

            while not done or not token_queue.empty():
                try:
                    msg_type, data = await asyncio.wait_for(
                        token_queue.get(),
                        timeout=1.0,
                    )
                    if msg_type == "token":
                        yield data
                    elif msg_type == "done":
                        break
                except asyncio.TimeoutError:
                    continue

        # Process through pipeline
        llm_response = await self.pipeline.process_llm_and_tts(
            llm_generator=llm_generator(),
            tts_session_id=state.tts_session_id,
            video_enabled=video_enabled,
            base_frame_index=state.musetalk_frame_index,
            is_generating=lambda: state.is_generating,
            on_llm_token=lambda token, full: asyncio.create_task(
                send_message(state, "llm_token", {"token": token, "full_text": full})
            ),
            on_av_frame=lambda frame: self._handle_av_frame(state, frame),
            on_error=lambda msg: asyncio.create_task(
                send_message(state, "error", {"message": msg})
            ),
            on_llm_complete=lambda text: asyncio.create_task(
                send_message(state, "llm_complete", {"text": text})
            ),
            on_tts_complete=lambda: asyncio.create_task(self._on_tts_complete(state)),
        )

        # Update conversation
        state.conversation.add_assistant_message(llm_response)

        # Close TTS session
        if state.tts_session_id is not None:
            await self._close_tts_session(state)

    def _handle_av_frame(self, state: ConnectionState, frame) -> asyncio.Task:
        """Handle AV frame - track last video frame and send to client."""
        # Log first frame latency - THE KEY METRIC
        if not state.first_av_frame_sent and state.utterance_end_time is not None:
            latency_ms = (time.time() - state.utterance_end_time) * 1000
            state.first_av_frame_sent = True
            logger.info(f"[LATENCY] ⚡ FIRST AV FRAME: {latency_ms:.0f}ms after utterance end")
        
        # Track last video frame for idle display (both per-connection and global)
        if frame.video_jpeg is not None:
            state.last_video_frame = frame.video_jpeg
            self._global_last_video_frame = frame.video_jpeg  # Persist across connections
        
        # Update frame index to continue animation from this point
        state.musetalk_frame_index = frame.frame_index + 1
        self._global_frame_index = frame.frame_index + 1  # Persist across connections
        
        return asyncio.create_task(
            send_message(state, "synced_av_frame", frame.to_websocket_payload())
        )

    async def _on_tts_complete(self, state: ConnectionState) -> None:
        """Handle TTS/video completion."""
        await send_message(state, "tts_complete", {})
        await send_message(state, "video_complete", {})

    async def _init_tts_session(self, state: ConnectionState) -> None:
        """Initialize TTS session for the connection."""
        state.tts_session_ready.clear()

        if not self.triton_client.is_healthy():
            await send_message(state, "tts_cache_ready", {
                "success": False,
                "session_id": None,
                "reason": "server_unavailable",
            })
            return

        if state.tts_session_id is not None:
            await self._close_tts_session(state)

        new_session_id = self.tts_service._get_next_session_id()
        state.tts_session_id = new_session_id

        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            self.tts_service.init_session,
            new_session_id,
        )

        if not state.is_connected:
            # Connection closed during init
            try:
                await loop.run_in_executor(
                    None,
                    self.tts_service.close_session,
                    new_session_id,
                )
            except Exception:
                pass
            state.tts_session_id = None
            return

        if success:
            state.tts_session_ready.set()
            await send_message(state, "tts_cache_ready", {
                "success": True,
                "session_id": new_session_id,
            })
            logger.info(f"TTS cache initialized for session {new_session_id}")
        else:
            state.tts_session_id = None
            await send_message(state, "tts_cache_ready", {
                "success": False,
                "session_id": new_session_id,
                "reason": "init_failed",
            })

    async def _close_tts_session(self, state: ConnectionState) -> None:
        """Close TTS session for the connection."""
        if state.tts_session_id is None:
            return

        old_session_id = state.tts_session_id
        state.tts_session_ready.clear()

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.tts_service.close_session,
                old_session_id,
            )
            logger.info(f"TTS session {old_session_id} closed")
        except Exception as exc:
            logger.warning(f"Error closing TTS session {old_session_id}: {exc}")

        state.tts_session_id = None

    async def _cleanup_connection(self, state: ConnectionState) -> None:
        """Clean up connection resources."""
        state.is_connected = False
        state.is_generating = False
        state.is_recording = False

        if state.audio_processor_task is not None:
            state.audio_processor_task.cancel()
            try:
                await state.audio_processor_task
            except asyncio.CancelledError:
                pass

        if state.tts_session_id is not None:
            try:
                await self._close_tts_session(state)
            except Exception:
                pass

        self.active_connections.pop(state.connection_id, None)
        logger.info(f"WebSocket disconnected: {state.connection_id}")
