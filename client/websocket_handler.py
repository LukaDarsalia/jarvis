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
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

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

    # Speculative processing state (full pipeline: STT → LLM → TTS → MuseTalk)
    # Runs in background during early silence, buffers AV frames on frontend
    speculative_task: Optional[asyncio.Task] = None
    speculative_stt_result: Optional[str] = None
    speculative_llm_result: Optional[str] = None
    speculative_cancelled: bool = False
    # Keep last completed speculative results to compare with final STT
    # Even if cancelled, if final STT matches, we can reuse buffered frames
    last_speculative_stt: Optional[str] = None
    last_speculative_llm: Optional[str] = None
    # Track if speculative pipeline completed TTS+MuseTalk
    speculative_pipeline_complete: bool = False
    # Track whether speculative pipeline actually started TTS (not LLM-only fallback)
    speculative_tts_started: bool = False
    # Track speculative AV frame count for sanity checks
    speculative_av_frame_count: int = 0
    # Flag to signal that llm_start has been sent to frontend - start streaming tokens
    speculative_llm_ui_started: bool = False

    # Serialize websocket sends to preserve message order
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    # Lock to prevent multiple TTS init calls
    tts_init_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tts_init_in_progress: bool = False
    # Lock to avoid resetting/closing TTS while a pipeline is using it
    tts_pipeline_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    # Key latency metric: time from utterance end to first AV frame
    utterance_end_time: Optional[float] = None  # When VAD detected utterance complete
    first_av_frame_sent: bool = False  # Track if we've sent first frame this generation

    # Voice prompt (voice cloning)
    voice_prompt_mode: bool = False
    voice_prompt_buffer: List[bytes] = field(default_factory=list)
    voice_prompt_audio: Optional[np.ndarray] = None
    voice_prompt_sample_rate: Optional[int] = None
    voice_id: Optional[str] = None
    custom_voice_prompts: Dict[str, Tuple[np.ndarray, int]] = field(default_factory=dict)

    # Avatar selection
    avatar_id: Optional[str] = None

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
        # Global custom voice prompts - survive across connections while server is running
        self._custom_voice_prompts: Dict[str, Tuple[np.ndarray, int]] = {}

    def _normalize_custom_voice_name(self, raw: str) -> Optional[str]:
        name = re.sub(r"[^a-zA-Z0-9 _-]", "", raw).strip()
        if not name:
            return None
        return name[:32]

    def _resolve_custom_voice_prompt(
        self,
        state: ConnectionState,
        voice_id: Optional[str],
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        if not voice_id or not voice_id.startswith("custom:"):
            return state.voice_prompt_audio, state.voice_prompt_sample_rate
        prompt = state.custom_voice_prompts.get(voice_id)
        if prompt is None:
            prompt = self._custom_voice_prompts.get(voice_id)
        if prompt is None:
            return None, None
        return prompt

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Handle a new WebSocket connection."""
        await websocket.accept()

        connection_id = f"conn_{int(time.time() * 1000)}"
        state = ConnectionState(
            websocket=websocket,
            connection_id=connection_id,
            musetalk_frame_index=self._global_frame_index,  # Continue from global index
        )
        state.voice_id = self.config.tts.voice_id
        state.avatar_id = self.config.musetalk.avatar_id
        if state.voice_id and state.voice_id.startswith("custom:"):
            prompt = self._custom_voice_prompts.get(state.voice_id)
            if prompt is not None:
                state.custom_voice_prompts[state.voice_id] = prompt
                state.voice_prompt_audio, state.voice_prompt_sample_rate = prompt
            else:
                state.voice_id = None
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
                    state.avatar_id,
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
            if state.voice_prompt_mode:
                state.voice_prompt_buffer.append(audio_bytes)
                return
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

        elif msg_type == "voice_prompt_start":
            await self._handle_voice_prompt_start(state)

        elif msg_type == "voice_prompt_stop":
            await self._handle_voice_prompt_stop(state, data)

        elif msg_type == "set_voice_id":
            await self._handle_set_voice_id(state, data)

        elif msg_type == "set_avatar_id":
            await self._handle_set_avatar_id(state, data)

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
        state.voice_prompt_mode = False
        state.voice_prompt_buffer.clear()

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

    async def _handle_voice_prompt_start(self, state: ConnectionState) -> None:
        """Start capturing a voice prompt for cloning."""
        state.voice_prompt_mode = True
        state.voice_prompt_buffer.clear()

    async def _handle_voice_prompt_stop(
        self,
        state: ConnectionState,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Stop capturing voice prompt and store it."""
        state.voice_prompt_mode = False
        data = data or {}
        raw_name = str(data.get("voice_name") or data.get("voice_id") or "").strip()
        voice_name = self._normalize_custom_voice_name(raw_name)

        if not state.voice_prompt_buffer:
            await send_message(state, "voice_prompt_error", {"message": "no_audio"})
            return

        try:
            chunks = [np.frombuffer(b, dtype=np.float32) for b in state.voice_prompt_buffer]
            audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
            state.voice_prompt_buffer.clear()

            if audio.size == 0:
                await send_message(state, "voice_prompt_error", {"message": "empty_audio"})
                return

            state.voice_prompt_audio = audio
            state.voice_prompt_sample_rate = 16000
            if voice_name:
                custom_voice_id = f"custom:{voice_name}"
                state.custom_voice_prompts[custom_voice_id] = (audio, 16000)
                self._custom_voice_prompts[custom_voice_id] = (audio, 16000)
                state.voice_id = custom_voice_id

            duration_ms = int(audio.size / 16.0)
            await send_message(
                state,
                "voice_prompt_ready",
                {
                    "duration_ms": duration_ms,
                    "voice_id": state.voice_id,
                    "voice_name": voice_name,
                },
            )

            # Re-init TTS session to apply new voice prompt
            await self._close_tts_session(state)
            await self._init_tts_session(state)
        except Exception as exc:
            logger.error(f"Voice prompt processing error: {exc}")
            await send_message(state, "voice_prompt_error", {"message": "processing_failed"})

    async def _handle_set_voice_id(self, state: ConnectionState, data: Dict[str, Any]) -> None:
        """Update the predefined voice ID."""
        voice_id = data.get("voice_id")
        if not voice_id:
            return
        state.voice_id = str(voice_id)
        if state.voice_id.startswith("custom:"):
            prompt = state.custom_voice_prompts.get(state.voice_id)
            if prompt is None:
                prompt = self._custom_voice_prompts.get(state.voice_id)
            if prompt is not None:
                state.voice_prompt_audio, state.voice_prompt_sample_rate = prompt
            else:
                state.voice_prompt_audio = None
                state.voice_prompt_sample_rate = None
                state.voice_id = None
                await send_message(state, "voice_prompt_error", {"message": "unknown_voice"})
        else:
            # Selecting a preset clears any recorded voice prompt
            state.voice_prompt_audio = None
            state.voice_prompt_sample_rate = None
        # Re-init TTS session to apply new voice preset
        await self._close_tts_session(state)
        await self._init_tts_session(state)

    async def _handle_set_avatar_id(self, state: ConnectionState, data: Dict[str, Any]) -> None:
        """Update the avatar ID for MuseTalk."""
        avatar_id = data.get("avatar_id")
        if not avatar_id:
            return
        state.avatar_id = str(avatar_id)
        state.musetalk_frame_index = 0
        try:
            loop = asyncio.get_event_loop()
            idle_frame = await loop.run_in_executor(
                None,
                self.triton_client.musetalk.get_idle_frame,
                state.avatar_id,
            )
            if idle_frame:
                state.last_video_frame = idle_frame
                self._global_last_video_frame = idle_frame
                await send_message(state, "avatar_ready", {
                    "avatar_id": state.avatar_id,
                    "idle_frame": base64.b64encode(idle_frame).decode("utf-8"),
                })
        except Exception as exc:
            logger.warning(f"Failed to load avatar idle frame: {exc}")

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

                # Always process VAD even during generation (for barge-in detection)
                # But only act on new utterances after handling potential barge-in
                if not state.is_recording:
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
                    logger.debug(f"[VAD_WS] Status: {status.value}, speculative_task={'running' if state.speculative_task else 'none'}, is_generating={state.is_generating}")

                # ================================================================
                # BARGE-IN HANDLING: User starts speaking during generation
                # ================================================================
                if state.is_generating and status == VADStatus.SPEECH_START:
                    logger.info(f"[BARGE-IN] 🛑 User started speaking during generation - stopping current output")
                    state.is_generating = False
                    
                    # Cancel any speculative task from previous response
                    if state.speculative_task is not None:
                        state.speculative_cancelled = True
                        state.speculative_task.cancel()
                        state.speculative_task = None
                    
                    # Clear speculative results
                    state.speculative_stt_result = None
                    state.speculative_llm_result = None
                    state.speculative_pipeline_complete = False
                    state.speculative_tts_started = False
                    state.speculative_av_frame_count = 0
                    state.speculative_llm_ui_started = False
                    
                    # Tell frontend to stop playback and clear buffer
                    await send_message(state, "stop_playback", {})
                    await send_message(state, "clear_speculative_buffer", {})
                    
                    # Reset TTS session for new utterance
                    if state.tts_session_id is not None:
                        asyncio.create_task(self._reset_tts_session_after_cancel(state))
                    
                    # Reset VAD to start fresh
                    self.triton_client.vad.reset_state()
                    state.vad_time_ms = 0.0
                    
                    # Process this audio chunk as start of new speech
                    status, audio_data = await loop.run_in_executor(
                        None,
                        self.triton_client.vad.process_with_state,
                        combined_audio,
                        state.vad_time_ms,
                    )
                    state.vad_time_ms += len(combined_audio) / 16000.0 * 1000.0

                # Handle speech resumption - cancel speculative processing so we can re-run with more audio
                # Only cancel on actual SPEECH_RESUMED, not on SPEAKING (which is just "in utterance")
                if status == VADStatus.SPEECH_RESUMED and state.speculative_task is not None:
                    logger.info(f"[SPECULATIVE] Speech resumed, cancelling speculative and clearing frontend buffer")
                    state.speculative_cancelled = True
                    state.speculative_task.cancel()
                    state.speculative_task = None
                    state.speculative_stt_result = None
                    state.speculative_llm_result = None
                    state.speculative_pipeline_complete = False
                    state.speculative_tts_started = False
                    state.speculative_av_frame_count = 0
                    state.speculative_llm_ui_started = False
                    # Tell frontend to clear its speculative buffer
                    await send_message(state, "clear_speculative_buffer", {})
                    
                    # TTS session may be in bad state after cancellation
                    # Close and re-init for next speculative run
                    if state.tts_session_id is not None:
                        asyncio.create_task(self._reset_tts_session_after_cancel(state))

                if status in {VADStatus.SPEAKING, VADStatus.SPEECH_START, VADStatus.SPEECH_CONTINUE, VADStatus.SPEECH_RESUMED, VADStatus.UTTERANCE_COMPLETE, VADStatus.EARLY_SILENCE}:
                    await send_message(state, "vad_status", {"status": status.value})

                # Handle early silence - start/restart speculative STT/LLM with latest audio
                if status == VADStatus.EARLY_SILENCE and audio_data is not None:
                    # Cancel any existing speculative task and start fresh with new audio
                    if state.speculative_task is not None:
                        state.speculative_task.cancel()
                        state.speculative_task = None
                    
                    logger.info(f"[SPECULATIVE] Early silence detected, starting speculative STT+LLM ({len(audio_data)} samples)")
                    state.speculative_cancelled = False
                    state.speculative_stt_result = None
                    state.speculative_llm_result = None
                    state.speculative_llm_ui_started = False  # Reset UI streaming flag
                    state.speculative_pipeline_complete = False
                    state.speculative_tts_started = False
                    state.speculative_av_frame_count = 0
                    state.speculative_task = asyncio.create_task(
                        self._process_speculative_stt_llm(state, audio_data)
                    )
                    
                    # Also ensure TTS session is being initialized if not ready yet
                    # This handles the case where user speaks before previous TTS session pre-init completes
                    if state.tts_session_id is None and not state.tts_session_ready.is_set():
                        logger.info(f"[SPECULATIVE] TTS session not ready, ensuring init in progress")
                        # The _init_tts_session handles the case where init is already in progress
                        asyncio.create_task(self._init_tts_session(state))

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
        Process audio speculatively through FULL pipeline: STT → LLM → TTS → MuseTalk.
        AV frames are sent to frontend as 'speculative_av_frame' and buffered there.
        On utterance_complete, frontend plays from buffer immediately (near-zero latency).
        """
        try:
            loop = asyncio.get_event_loop()
            state.speculative_pipeline_complete = False
            state.speculative_tts_started = False
            state.speculative_av_frame_count = 0
            
            # ================================================================
            # Phase 1: Speculative STT
            # ================================================================
            logger.info(f"[SPECULATIVE] Starting STT on {len(audio)} samples")
            await send_message(state, "speculative_stt_start", {})
            
            transcript = await loop.run_in_executor(
                None,
                self.triton_client.stt.transcribe,
                audio,
            )
            
            if state.speculative_cancelled:
                if transcript.strip():
                    state.last_speculative_stt = transcript
                logger.info(f"[SPECULATIVE] STT completed but cancelled, saving for potential reuse: '{transcript}'")
                return
            
            state.speculative_stt_result = transcript
            state.last_speculative_stt = transcript
            logger.info(f"[SPECULATIVE] STT result: {transcript}")
            await send_message(state, "speculative_stt_complete", {"text": transcript})
            
            if not transcript.strip():
                return
            
            # ================================================================
            # Phase 2: Speculative LLM (streaming tokens to TTS)
            # ================================================================
            logger.info(f"[SPECULATIVE] Starting LLM + TTS + MuseTalk pipeline")
            await send_message(state, "speculative_llm_start", {})
            
            # Wait for TTS session to be ready
            try:
                if state.tts_session_id is None and not state.tts_init_in_progress:
                    asyncio.create_task(self._init_tts_session(state))
                await asyncio.wait_for(
                    state.tts_session_ready.wait(),
                    timeout=self.config.streaming.tts_init_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[SPECULATIVE] TTS session not ready, falling back to STT+LLM only")
                # Fall back to LLM-only speculative
                state.speculative_tts_started = False
                state.speculative_av_frame_count = 0
                await self._speculative_llm_only(state, transcript)
                return
            
            if state.tts_session_id is None:
                logger.warning(f"[SPECULATIVE] TTS session not initialized, falling back to STT+LLM only")
                state.speculative_tts_started = False
                state.speculative_av_frame_count = 0
                await self._speculative_llm_only(state, transcript)
                return
            
            if state.speculative_cancelled:
                return
            
            state.speculative_tts_started = True
            
            # Check video availability
            video_enabled = await self._check_musetalk_available()
            
            # Build prompt
            prompt = self.triton_client.llm.build_prompt(
                transcript,
                state.conversation.get_history(),
            )
            
            # Create async LLM generator
            async def llm_generator():
                token_queue: asyncio.Queue = asyncio.Queue()
                done = False

                def llm_worker():
                    nonlocal done
                    try:
                        for token in self.triton_client.llm.generate_stream(prompt):
                            if state.speculative_cancelled:
                                break
                            loop.call_soon_threadsafe(token_queue.put_nowait, ("token", token))
                    finally:
                        done = True
                        loop.call_soon_threadsafe(token_queue.put_nowait, ("done", None))

                loop.run_in_executor(None, llm_worker)

                while not done or not token_queue.empty():
                    try:
                        msg_type, data = await asyncio.wait_for(
                            token_queue.get(), timeout=0.1
                        )
                        if msg_type == "done":
                            break
                        if state.speculative_cancelled:
                            break
                        yield data
                    except asyncio.TimeoutError:
                        if state.speculative_cancelled:
                            break
                        continue

            llm_response = ""
            
            def on_llm_token(token, full_text):
                nonlocal llm_response
                llm_response = full_text
                # Update state so partial result is available even before pipeline completes
                state.speculative_llm_result = full_text
                
                # Stream LLM tokens to frontend for live text display
                # Only do this if utterance has completed (llm_start sent)
                if state.speculative_llm_ui_started:
                    asyncio.create_task(
                        send_message(state, "llm_token", {
                            "token": token,
                            "full_text": full_text
                        })
                    )
            
            def on_speculative_av_frame(frame):
                """Send AV frame as speculative (buffered on frontend)."""
                # Track last video frame for idle display
                if frame.video_jpeg is not None:
                    state.last_video_frame = frame.video_jpeg
                    self._global_last_video_frame = frame.video_jpeg
                
                state.musetalk_frame_index = frame.frame_index + 1
                self._global_frame_index = frame.frame_index + 1
                state.speculative_av_frame_count += 1
                
                payload = frame.to_websocket_payload()
                payload["speculative"] = True  # Mark as speculative
                return asyncio.create_task(
                    send_message(state, "speculative_av_frame", payload)
                )
            
            # Run full pipeline with speculative AV frame handler
            try:
                async with state.tts_pipeline_lock:
                    final_response = await self.pipeline.process_llm_and_tts(
                        llm_generator=llm_generator(),
                        tts_session_id=state.tts_session_id,
                        video_enabled=video_enabled,
                        base_frame_index=state.musetalk_frame_index,
                        avatar_id=state.avatar_id,
                        is_generating=lambda: not state.speculative_cancelled,
                        on_llm_token=on_llm_token,
                        on_av_frame=on_speculative_av_frame,
                        on_error=lambda msg: logger.error(f"[SPECULATIVE] Pipeline error: {msg}"),
                        on_llm_complete=lambda text: None,
                        on_tts_complete=lambda: None,
                    )
                
                if state.speculative_cancelled:
                    if llm_response.strip():
                        state.last_speculative_llm = llm_response
                    logger.info(f"[SPECULATIVE] Pipeline cancelled after partial completion")
                    return
                
                state.speculative_llm_result = final_response or llm_response
                state.last_speculative_llm = state.speculative_llm_result
                state.speculative_pipeline_complete = True
                
                logger.info(f"[SPECULATIVE] ✅ Full pipeline complete: {len(state.speculative_llm_result)} chars")
                await send_message(state, "speculative_pipeline_complete", {
                    "text": state.speculative_llm_result
                })
                
            except Exception as exc:
                logger.error(f"[SPECULATIVE] Pipeline error: {exc}")
                # Fall back to LLM-only result if we have it
                if llm_response.strip():
                    state.speculative_llm_result = llm_response
                    state.last_speculative_llm = llm_response
            
        except asyncio.CancelledError:
            logger.info(f"[SPECULATIVE] Task cancelled")
        except Exception as exc:
            logger.error(f"[SPECULATIVE] Error: {exc}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def _speculative_llm_only(
        self,
        state: ConnectionState,
        transcript: str,
    ) -> None:
        """Run LLM-only speculative processing (fallback when TTS not ready)."""
        loop = asyncio.get_event_loop()
        state.speculative_tts_started = False
        state.speculative_av_frame_count = 0
        
        prompt = self.triton_client.llm.build_prompt(
            transcript,
            state.conversation.get_history(),
        )
        
        llm_response = ""
        
        def run_llm():
            nonlocal llm_response
            for token in self.triton_client.llm.generate_stream(prompt):
                if state.speculative_cancelled:
                    return
                llm_response += token
        
        await loop.run_in_executor(None, run_llm)
        
        if state.speculative_cancelled:
            if llm_response.strip():
                state.last_speculative_llm = llm_response
            logger.info(f"[SPECULATIVE] LLM cancelled, saving: {len(llm_response)} chars")
            return
        
        state.speculative_llm_result = llm_response
        state.last_speculative_llm = llm_response
        logger.info(f"[SPECULATIVE] LLM complete (no TTS): {len(llm_response)} chars")
        await send_message(state, "speculative_llm_complete", {"text": llm_response})

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
            state.speculative_stt_result is not None and
            state.speculative_tts_started
        )
        
        logger.info(f"[VOICE2VOICE] use_speculative={use_speculative}, "
                    f"task={state.speculative_task is not None}, "
                    f"cancelled={state.speculative_cancelled}, "
                    f"stt_result={state.speculative_stt_result is not None}, "
                    f"llm_result={state.speculative_llm_result is not None}, "
                    f"pipeline_complete={state.speculative_pipeline_complete}")

        try:
            if use_speculative and state.speculative_stt_result:
                # Use speculative STT result - DON'T wait for full pipeline!
                transcript = state.speculative_stt_result
                t_stt = (time.time() - t_start) * 1000
                logger.info(f"[LATENCY] STT (speculative): {t_stt:.0f}ms - '{transcript}'")
                await send_message(state, "stt_start", {})
                await send_message(state, "stt_complete", {"text": transcript, "speculative": True})
                
                # Update conversation history
                state.conversation.add_user_message(transcript)
                
                # Send UI updates for LLM (even if still running, show what we have)
                await send_message(state, "llm_start", {})
                
                # Enable LLM token streaming to frontend from speculative pipeline
                state.speculative_llm_ui_started = True
                
                if state.speculative_llm_result:
                    # LLM already has some/all result, send it immediately
                    await send_message(state, "llm_token", {
                        "token": state.speculative_llm_result,
                        "full_text": state.speculative_llm_result
                    })
                    # Only send llm_complete if pipeline is done
                    if state.speculative_pipeline_complete:
                        await send_message(state, "llm_complete", {"text": state.speculative_llm_result})
                        state.conversation.add_assistant_message(state.speculative_llm_result)
                
                await send_message(state, "tts_start", {"text": "", "video_enabled": True})
                
                # Signal frontend to start playing from speculative buffer IMMEDIATELY
                # Frontend will play whatever frames are buffered and continue receiving more
                logger.info(f"[VOICE2VOICE] 🚀 Signaling frontend to start playing from buffer (pipeline still running)")
                await send_message(state, "start_speculative_playback", {})
                
                # Now wait for the speculative task to complete (in background, frames still streaming)
                if state.speculative_task and not state.speculative_task.done():
                    logger.info(f"[VOICE2VOICE] Waiting for speculative pipeline to finish...")
                    try:
                        await state.speculative_task
                    except asyncio.CancelledError:
                        pass
                
                # If we didn't send the LLM result yet (was still generating when playback started),
                # send it now that pipeline is complete
                if state.speculative_llm_result:
                    # Send full LLM text to frontend (final update)
                    await send_message(state, "llm_token", {
                        "token": "",
                        "full_text": state.speculative_llm_result
                    })
                    await send_message(state, "llm_complete", {"text": state.speculative_llm_result})
                    # Add to conversation if not already added
                    last_assistant = state.conversation.get_last_assistant_message()
                    if not last_assistant or last_assistant != state.speculative_llm_result:
                        state.conversation.add_assistant_message(state.speculative_llm_result)
                
                # Reset UI streaming flag
                state.speculative_llm_ui_started = False
                
                # Signal completion
                await self._on_tts_complete(state)
                
                # Re-init TTS session for next utterance
                if state.tts_session_id is not None:
                    await self._close_tts_session(state)
                asyncio.create_task(self._init_tts_session(state))
                return
            
            # If speculative ran LLM-only (no TTS), reuse STT/LLM and run TTS now
            if (
                state.speculative_task is not None and
                not state.speculative_cancelled and
                state.speculative_stt_result is not None and
                not state.speculative_tts_started
            ):
                transcript = state.speculative_stt_result
                t_stt = (time.time() - t_start) * 1000
                logger.info(f"[LATENCY] STT (speculative): {t_stt:.0f}ms - '{transcript}'")
                await send_message(state, "stt_start", {})
                await send_message(state, "stt_complete", {"text": transcript, "speculative": True})

                if not transcript.strip():
                    return

                state.conversation.add_user_message(transcript)

                if state.speculative_task and not state.speculative_task.done():
                    logger.info("[VOICE2VOICE] Waiting for speculative LLM-only result...")
                    try:
                        await state.speculative_task
                    except asyncio.CancelledError:
                        pass

                if state.speculative_llm_result:
                    await self._process_tts_with_cached_llm(
                        state,
                        transcript,
                        state.speculative_llm_result,
                    )
                else:
                    await self._process_llm_and_tts(state, transcript)
                return

            # === Non-speculative path: normal STT processing ===
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
            
            # Check if final STT matches last speculative STT - can reuse results!
            reuse_llm_result: Optional[str] = None
            if (state.last_speculative_stt and
                transcript.strip() == state.last_speculative_stt.strip()):
                # Check if full pipeline was completed - use the buffered frames!
                if state.speculative_pipeline_complete and state.last_speculative_llm:
                    logger.info(f"[VOICE2VOICE] ✅ Final STT matches speculative, using buffered frames!")
                    
                    # Update conversation
                    state.conversation.add_user_message(transcript)
                    state.conversation.add_assistant_message(state.last_speculative_llm)
                    
                    # Send UI updates
                    await send_message(state, "llm_start", {})
                    await send_message(state, "llm_token", {
                        "token": state.last_speculative_llm,
                        "full_text": state.last_speculative_llm
                    })
                    await send_message(state, "llm_complete", {"text": state.last_speculative_llm})
                    await send_message(state, "tts_start", {"text": "", "video_enabled": True})
                    
                    # Signal frontend to play from buffer
                    await send_message(state, "start_speculative_playback", {})
                    await self._on_tts_complete(state)
                    
                    if state.tts_session_id is not None:
                        await self._close_tts_session(state)
                    asyncio.create_task(self._init_tts_session(state))
                    return
                
                # Only LLM cached, need to run TTS
                if state.last_speculative_llm:
                    logger.info(f"[VOICE2VOICE] ✅ Final STT matches, reusing LLM result")
                    reuse_llm_result = state.last_speculative_llm
            elif state.last_speculative_stt:
                logger.info(f"[VOICE2VOICE] ❌ STT mismatch: final='{transcript}' vs speculative='{state.last_speculative_stt}'")
                # Clear speculative buffer on frontend since STT changed
                await send_message(state, "clear_speculative_buffer", {})

            if not transcript.strip():
                return

            state.conversation.add_user_message(transcript)

            # Use cached LLM result if available, otherwise run normal pipeline
            if reuse_llm_result:
                t_llm = (time.time() - t_start) * 1000
                logger.info(f"[LATENCY] LLM (reused): {t_llm:.0f}ms - {len(reuse_llm_result)} chars")
                await self._process_tts_with_cached_llm(state, transcript, reuse_llm_result)
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
            state.speculative_pipeline_complete = False
            state.speculative_tts_started = False
            state.speculative_av_frame_count = 0
            # Clear last speculative results after use to avoid stale data
            state.last_speculative_stt = None
            state.last_speculative_llm = None

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

        # Use the pre-initialized TTS session (initialized after previous response)
        # Wait for it to be ready
        try:
            if state.tts_session_id is None and not state.tts_init_in_progress:
                asyncio.create_task(self._init_tts_session(state))
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
        async with state.tts_pipeline_lock:
            llm_response = await self.pipeline.process_llm_and_tts(
                llm_generator=cached_llm_generator(),
                tts_session_id=state.tts_session_id,
                video_enabled=video_enabled,
                base_frame_index=state.musetalk_frame_index,
                avatar_id=state.avatar_id,
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

        # Close TTS session and pre-initialize next one for faster subsequent responses
        if state.tts_session_id is not None:
            await self._close_tts_session(state)
        
        # Pre-initialize next TTS session in background (don't await)
        # This ensures the session is ready for the next utterance
        asyncio.create_task(self._init_tts_session(state))

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

        # Use the pre-initialized TTS session (initialized after previous response or on connect)
        # Wait for it to be ready
        try:
            if state.tts_session_id is None and not state.tts_init_in_progress:
                asyncio.create_task(self._init_tts_session(state))
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
        async with state.tts_pipeline_lock:
            llm_response = await self.pipeline.process_llm_and_tts(
                llm_generator=llm_generator(),
                tts_session_id=state.tts_session_id,
                video_enabled=video_enabled,
                base_frame_index=state.musetalk_frame_index,
                avatar_id=state.avatar_id,
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

        # Close TTS session and pre-initialize next one for faster subsequent responses
        if state.tts_session_id is not None:
            await self._close_tts_session(state)
        
        # Pre-initialize next TTS session in background (don't await)
        # This ensures the session is ready for the next utterance
        asyncio.create_task(self._init_tts_session(state))

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

        idle_frame = state.last_video_frame or self._global_last_video_frame
        payload: Dict[str, Any] = {}
        if idle_frame:
            payload["idle_frame"] = base64.b64encode(idle_frame).decode("utf-8")

        await send_message(state, "video_complete", payload)

    async def _init_tts_session(self, state: ConnectionState) -> None:
        """Initialize TTS session for the connection."""
        # Prevent multiple concurrent TTS inits
        if state.tts_init_in_progress:
            logger.info(f"[TTS_INIT] Already in progress, skipping duplicate init")
            return
            
        async with state.tts_init_lock:
            # Double check after acquiring lock
            if state.tts_session_ready.is_set() and state.tts_session_id is not None:
                logger.info(f"[TTS_INIT] Session {state.tts_session_id} already ready, skipping")
                return
                
            state.tts_init_in_progress = True
            state.tts_session_ready.clear()

            try:
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
                voice_prompt_audio, voice_prompt_sample_rate = self._resolve_custom_voice_prompt(
                    state,
                    state.voice_id,
                )
                voice_id = state.voice_id
                if not voice_id and self.config.tts.voice_id.startswith("custom:"):
                    voice_id = "alba"
                success = await loop.run_in_executor(
                    None,
                    self.tts_service.init_session,
                    new_session_id,
                    voice_prompt_audio,
                    voice_prompt_sample_rate,
                    voice_id,
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
            finally:
                state.tts_init_in_progress = False

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

    async def _reset_tts_session_after_cancel(self, state: ConnectionState) -> None:
        """Reset TTS session after speculative pipeline was cancelled."""
        # Wait a tiny bit for the cancellation to propagate
        await asyncio.sleep(0.1)
        # Avoid closing while a pipeline is still using the session
        async with state.tts_pipeline_lock:
            # Close the potentially corrupted session
            await self._close_tts_session(state)
            
            # Re-initialize for the next speculative run
            await self._init_tts_session(state)
            logger.info(f"[TTS_RESET] TTS session reset after speculative cancellation")

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
