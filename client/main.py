"""
FastAPI Backend for Voice Assistant
WebSocket-based streaming for VAD, STT, LLM, and TTS
"""

import asyncio
import json
import logging
import time
import base64
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from typing import List
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from queue import Queue

from triton_client import (
    TritonVoiceClient, 
    ConversationManager,
    VADParams, 
    LLMParams, 
    TTSParams,
    TTSMetrics,
    TTSSession,
    MuseTalkParams,
    MuseTalkMetrics,
    MuseTalkSession,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
triton_client: Optional[TritonVoiceClient] = None
active_connections: dict = {}


def _split_text_for_streaming( text: str, append_empty_stings:bool = True) -> List[str]:
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
    
    result = [' '.join(words[:3])]
    for w in words[3:]:
        result.append(' ' + w)
    if append_empty_stings:
        result.extend(["", ""])
    return result

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global triton_client
    
    # Startup
    logger.info("Initializing Triton client...")
    triton_client = TritonVoiceClient(
        triton_url="localhost:8001",
        vad_params=VADParams(),
        llm_params=LLMParams(),
        tts_params=TTSParams()
    )
    
    # Check connectivity
    if triton_client.check_health():
        logger.info("Triton server is healthy")
        models_status = triton_client.check_models_ready()
        logger.info(f"Models status: {models_status}")
    else:
        logger.warning("Triton server is not available - client will retry on requests")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Voice Assistant API",
    description="WebSocket-based voice assistant with VAD, STT, LLM, and TTS",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# ============== REST Endpoints ==============

@app.get("/")
async def root():
    """Serve the main UI"""
    index_path = static_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Voice Assistant API - WebSocket endpoint at /ws"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    if triton_client is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Client not initialized"}
        )
    
    server_healthy = triton_client.check_health()
    models_status = triton_client.check_models_ready()
    
    return {
        "status": "healthy" if server_healthy else "degraded",
        "triton_server": server_healthy,
        "models": models_status
    }


@app.get("/config")
async def get_config():
    """Get current configuration"""
    if triton_client is None:
        raise HTTPException(status_code=503, detail="Client not initialized")
    
    buffer_config = triton_client.get_buffer_config()
    
    return {
        "vad": {
            "speech_threshold_ms": triton_client.vad_params.speech_threshold_ms,
            "silence_threshold_ms": triton_client.vad_params.silence_threshold_ms,
            "prob_threshold": triton_client.vad_params.prob_threshold,
        },
        "llm": {
            "max_new_tokens": triton_client.llm_params.max_new_tokens,
            "temperature": triton_client.llm_params.temperature,
            "top_p": triton_client.llm_params.top_p,
            "system_prompt": triton_client.llm_params.system_prompt,
        },
        "tts": {
            "backbone_temperature": triton_client.tts_params.backbone_temperature,
            "backbone_top_p": triton_client.tts_params.backbone_top_p,
            "depth_temperature": triton_client.tts_params.depth_temperature,
            "depth_top_p": triton_client.tts_params.depth_top_p,
            "target_sample_rate": triton_client.tts_params.target_sample_rate,
        },
        "buffer": {
            "manual_buffer_ms": buffer_config.get("manual_buffer_ms"),
            "current_buffer_ms": buffer_config.get("buffer_ms"),
            "buffer_source": buffer_config.get("buffer_source"),
            "frame_buffer": buffer_config.get("frame_buffer"),
        }
    }


@app.post("/config")
async def update_config(config: dict):
    """Update configuration"""
    if triton_client is None:
        raise HTTPException(status_code=503, detail="Client not initialized")
    
    # Update VAD params
    if "vad" in config:
        vad = config["vad"]
        if "speech_threshold_ms" in vad:
            triton_client.vad_params.speech_threshold_ms = float(vad["speech_threshold_ms"])
        if "silence_threshold_ms" in vad:
            triton_client.vad_params.silence_threshold_ms = float(vad["silence_threshold_ms"])
        if "prob_threshold" in vad:
            triton_client.vad_params.prob_threshold = float(vad["prob_threshold"])
    
    # Update LLM params
    if "llm" in config:
        llm = config["llm"]
        if "max_new_tokens" in llm:
            triton_client.llm_params.max_new_tokens = int(llm["max_new_tokens"])
        if "temperature" in llm:
            triton_client.llm_params.temperature = float(llm["temperature"])
        if "top_p" in llm:
            triton_client.llm_params.top_p = float(llm["top_p"])
        if "system_prompt" in llm:
            triton_client.llm_params.system_prompt = str(llm["system_prompt"])
    
    # Update TTS params
    if "tts" in config:
        tts = config["tts"]
        if "backbone_temperature" in tts:
            triton_client.tts_params.backbone_temperature = float(tts["backbone_temperature"])
        if "backbone_top_p" in tts:
            triton_client.tts_params.backbone_top_p = float(tts["backbone_top_p"])
        if "depth_temperature" in tts:
            triton_client.tts_params.depth_temperature = float(tts["depth_temperature"])
        if "depth_top_p" in tts:
            triton_client.tts_params.depth_top_p = float(tts["depth_top_p"])
        if "target_sample_rate" in tts:
            triton_client.tts_params.target_sample_rate = int(tts["target_sample_rate"])
    
    # Update buffer override
    if "buffer" in config:
        buffer_cfg = config["buffer"]
        manual_buffer = buffer_cfg.get("manual_buffer_ms", None)
        # Accept null/None to return to adaptive mode
        if manual_buffer in ["", None]:
            triton_client.set_manual_buffer_ms(None)
        else:
            try:
                triton_client.set_manual_buffer_ms(float(manual_buffer))
            except (TypeError, ValueError):
                logger.warning(f"Ignoring invalid manual buffer value: {manual_buffer}")
    
    return await get_config()


# ============== WebSocket Connection Handler ==============

class ConnectionState:
    """State for a WebSocket connection"""
    def __init__(self, websocket: WebSocket, connection_id: str):
        self.websocket = websocket
        self.connection_id = connection_id
        self.conversation = ConversationManager()
        self.mode = "voice_to_voice"  # voice_to_voice | text_to_voice | tts_only
        self.tts_session_id: Optional[int] = None
        self.tts_session_ready = asyncio.Event()  # Set when TTS session is ready
        self.is_generating = False
        self.is_connected = True  # Track connection state
        self.vad_time_ms = 0
        
        # MuseTalk state
        self.musetalk_session_id: Optional[int] = None
        self.musetalk_session_ready = asyncio.Event()
        self.musetalk_enabled = True  # Can be disabled if model not available
        self.musetalk_audio_buffer: List[np.ndarray] = []  # Buffer TTS audio for MuseTalk


async def send_message(state_or_ws, msg_type: str, data: dict):
    """Send a JSON message through WebSocket"""
    try:
        # Handle both ConnectionState and raw WebSocket
        if isinstance(state_or_ws, ConnectionState):
            if not state_or_ws.is_connected:
                logger.debug(f"Skipping message {msg_type} - connection closed")
                return
            ws = state_or_ws.websocket
        else:
            ws = state_or_ws
        await ws.send_json({"type": msg_type, **data})
    except Exception as e:
        logger.error(f"Failed to send message: {e}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for voice assistant"""
    await websocket.accept()
    connection_id = f"conn_{int(time.time() * 1000)}"
    state = ConnectionState(websocket, connection_id)
    active_connections[connection_id] = state
    
    logger.info(f"WebSocket connected: {connection_id}")
    
    try:
        # Send connection confirmation
        await send_message(websocket, "connected", {
            "connection_id": connection_id,
            "message": "Connected to Voice Assistant"
        })
        
        # Initialize TTS and MuseTalk sessions on connection
        if triton_client:
            # Run cache init in background
            asyncio.create_task(init_tts_cache_async(state))
            asyncio.create_task(init_musetalk_session_async(state))
        
        # Main message loop
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                await handle_text_message(state, json.loads(message["text"]))
            elif "bytes" in message:
                await handle_audio_message(state, message["bytes"])
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Mark connection as closed first (so async tasks know to stop)
        state.is_connected = False
        
        # Clean up TTS session on disconnect
        if state.tts_session_id is not None and triton_client is not None:
            try:
                triton_client.end_tts_session(state.tts_session_id)
                logger.info(f"TTS session {state.tts_session_id} cleaned up on disconnect")
            except Exception as e:
                logger.warning(f"Error cleaning up TTS session: {e}")
        
        # Clean up MuseTalk session on disconnect
        if state.musetalk_session_id is not None and triton_client is not None:
            try:
                triton_client.end_musetalk_session(state.musetalk_session_id)
                logger.info(f"MuseTalk session {state.musetalk_session_id} cleaned up on disconnect")
            except Exception as e:
                logger.warning(f"Error cleaning up MuseTalk session: {e}")
        
        if connection_id in active_connections:
            del active_connections[connection_id]


async def close_tts_session_async(state: ConnectionState):
    """Close existing TTS session asynchronously"""
    if state.tts_session_id is not None and triton_client is not None:
        old_session_id = state.tts_session_id
        state.tts_session_ready.clear()  # Mark as not ready
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                triton_client.end_tts_session,
                old_session_id
            )
            logger.info(f"TTS session {old_session_id} closed")
        except Exception as e:
            logger.warning(f"Error closing TTS session {old_session_id}: {e}")
        state.tts_session_id = None


async def init_tts_cache_async(state: ConnectionState):
    """Initialize TTS cache asynchronously"""
    # Clear ready state at the start
    state.tts_session_ready.clear()
    
    try:
        # Check if connection is still active
        if not state.is_connected:
            logger.info("Connection closed, skipping TTS cache init")
            return
        
        # Check if server is available before attempting cache init
        if not triton_client.check_health():
            logger.warning("Triton server not healthy, skipping TTS cache init")
            await send_message(state, "tts_cache_ready", {
                "success": False,
                "session_id": None,
                "reason": "server_unavailable"
            })
            return
        
        # Close existing session first if any
        if state.tts_session_id is not None:
            logger.info(f"Closing existing TTS session {state.tts_session_id} before creating new one")
            await close_tts_session_async(state)
        
        # Get a new session ID BEFORE init
        new_session_id = triton_client._get_next_session_id()
        state.tts_session_id = new_session_id  # Set immediately so it's available
        
        # Check connection again before blocking call
        if not state.is_connected:
            logger.info("Connection closed during TTS init, aborting")
            state.tts_session_id = None
            return
        
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, 
            triton_client.init_tts_session, 
            new_session_id
        )
        
        # Check connection after blocking call
        if not state.is_connected:
            logger.info(f"Connection closed after TTS init, cleaning up session {new_session_id}")
            # Clean up the session we just created
            try:
                await loop.run_in_executor(None, triton_client.end_tts_session, new_session_id)
            except Exception:
                pass
            state.tts_session_id = None
            return
        
        if success:
            state.tts_session_ready.set()  # Signal that session is ready
            await send_message(state, "tts_cache_ready", {
                "success": True,
                "session_id": new_session_id
            })
            logger.info(f"TTS cache initialized for session {new_session_id}: success=True")
        else:
            state.tts_session_id = None  # Clear on failure
            await send_message(state, "tts_cache_ready", {
                "success": False,
                "session_id": new_session_id,
                "reason": "init_failed"
            })
            logger.warning(f"TTS cache initialization failed for session {new_session_id}")
            
    except Exception as e:
        logger.error(f"TTS cache init error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        state.tts_session_id = None


async def close_musetalk_session_async(state: ConnectionState):
    """Close existing MuseTalk session asynchronously"""
    if state.musetalk_session_id is not None and triton_client is not None:
        old_session_id = state.musetalk_session_id
        state.musetalk_session_ready.clear()
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                triton_client.end_musetalk_session,
                old_session_id
            )
            logger.info(f"MuseTalk session {old_session_id} closed")
        except Exception as e:
            logger.warning(f"Error closing MuseTalk session {old_session_id}: {e}")
        state.musetalk_session_id = None


async def init_musetalk_session_async(state: ConnectionState, avatar_id: str = "default"):
    """Initialize MuseTalk session asynchronously"""
    state.musetalk_session_ready.clear()
    
    try:
        if not state.is_connected:
            logger.info("Connection closed, skipping MuseTalk init")
            return
        
        # Check if musetalk model is available
        models_status = triton_client.check_models_ready()
        if not models_status.get("musetalk", False):
            logger.warning("MuseTalk model not available, disabling avatar")
            state.musetalk_enabled = False
            await send_message(state, "musetalk_ready", {
                "success": False,
                "session_id": None,
                "reason": "model_unavailable"
            })
            return
        
        # Close existing session if any
        if state.musetalk_session_id is not None:
            logger.info(f"Closing existing MuseTalk session {state.musetalk_session_id}")
            await close_musetalk_session_async(state)
        
        # Get new session ID
        new_session_id = triton_client._get_next_musetalk_session_id()
        state.musetalk_session_id = new_session_id
        
        if not state.is_connected:
            state.musetalk_session_id = None
            return
        
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            triton_client.init_musetalk_session,
            new_session_id,
            avatar_id
        )
        
        if not state.is_connected:
            try:
                await loop.run_in_executor(None, triton_client.end_musetalk_session, new_session_id)
            except Exception:
                pass
            state.musetalk_session_id = None
            return
        
        if success:
            state.musetalk_session_ready.set()
            state.musetalk_enabled = True
            
            # Get idle frame and buffer config to send to client
            idle_frame = triton_client.get_musetalk_idle_frame(new_session_id)
            buffer_config = triton_client.get_buffer_config()

            await send_message(state, "musetalk_ready", {
                "success": True,
                "session_id": new_session_id,
                "idle_frame": base64.b64encode(idle_frame).decode("utf-8") if idle_frame else None,
                "buffer_config": buffer_config,
            })
            logger.info(
                f"MuseTalk session {new_session_id} initialized successfully, "
                f"buffer: {buffer_config.get('buffer_ms', 0)}ms "
                f"(source={buffer_config.get('buffer_source')}, manual={buffer_config.get('manual_buffer_ms')}) "
                f"details: {buffer_config}"
            )
        else:
            state.musetalk_session_id = None
            state.musetalk_enabled = False
            await send_message(state, "musetalk_ready", {
                "success": False,
                "session_id": new_session_id,
                "reason": "init_failed"
            })
            logger.warning(f"MuseTalk initialization failed for session {new_session_id}")
            
    except Exception as e:
        logger.error(f"MuseTalk init error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        state.musetalk_session_id = None
        state.musetalk_enabled = False


async def handle_text_message(state: ConnectionState, data: dict):
    """Handle text messages from WebSocket"""
    msg_type = data.get("type", "")
    
    if msg_type == "set_mode":
        state.mode = data.get("mode", "voice_to_voice")
        await send_message(state, "mode_changed", {"mode": state.mode})
        logger.info(f"Mode changed to: {state.mode}")
    
    elif msg_type == "text_input":
        # Text input for text_to_voice or tts_only mode
        text = data.get("text", "").strip()
        if not text:
            return
        
        if state.mode == "tts_only":
            # Direct TTS
            await process_tts_only(state, text)
        else:
            # Text to voice (with LLM)
            await process_text_to_voice(state, text)
    
    elif msg_type == "clear_conversation":
        state.conversation.clear()
        await send_message(state, "conversation_cleared", {})
    
    elif msg_type == "stop_generation":
        state.is_generating = False

    elif msg_type == "toggle_musetalk":
        # Toggle MuseTalk on/off for testing
        enabled = data.get("enabled", True)
        state.musetalk_enabled = enabled
        logger.info(f"MuseTalk toggled: {'enabled' if enabled else 'disabled'}")
        await send_message(state, "musetalk_toggled", {"enabled": enabled})

    elif msg_type == "get_buffer_config":
        # Get current adaptive buffer configuration
        if triton_client:
            buffer_config = triton_client.get_buffer_config()
            await send_message(state, "buffer_config", buffer_config)
            logger.info(f"Buffer config sent to client: {buffer_config}")


async def handle_audio_message(state: ConnectionState, audio_bytes: bytes):
    """Handle audio data from WebSocket"""
    if state.mode not in ["voice_to_voice"]:
        return
    
    if triton_client is None:
        return
    
    # Decode audio (expecting float32 PCM at 16kHz)
    try:
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
    except Exception as e:
        logger.error(f"Failed to decode audio: {e}")
        return
    
    # Process through VAD
    state.vad_time_ms += len(audio) / 16000 * 1000
    
    try:
        loop = asyncio.get_event_loop()
        status, complete_audio = await loop.run_in_executor(
            None,
            triton_client.process_vad_with_state,
            audio,
            state.vad_time_ms
        )
        
        # Send VAD status
        await send_message(state, "vad_status", {"status": status})
        
        if status == "utterance_complete" and complete_audio is not None:
            # Process complete utterance
            await process_voice_to_voice(state, complete_audio)
            state.vad_time_ms = 0
            triton_client.reset_vad_state()
    
    except Exception as e:
        logger.error(f"VAD processing error: {e}")


async def process_voice_to_voice(state: ConnectionState, audio: np.ndarray):
    """Process voice input through STT -> LLM -> TTS pipeline"""
    if triton_client is None:
        return
    
    state.is_generating = True
    
    try:
        # STT
        await send_message(state, "stt_start", {})
        loop = asyncio.get_event_loop()
        transcript = await loop.run_in_executor(None, triton_client.transcribe, audio)
        
        await send_message(state, "stt_complete", {"text": transcript})
        logger.info(f"STT result: {transcript}")
        
        if not transcript.strip():
            return
        
        # Add to conversation
        state.conversation.add_user_message(transcript)
        
        # LLM
        await process_llm_and_tts(state, transcript)
    
    except Exception as e:
        logger.error(f"Voice to voice error: {e}")
        await send_message(state, "error", {"message": str(e)})
    finally:
        state.is_generating = False


async def process_text_to_voice(state: ConnectionState, text: str):
    """Process text input through LLM -> TTS pipeline"""
    if triton_client is None:
        return
    
    state.is_generating = True
    
    try:
        # Add user message to conversation
        state.conversation.add_user_message(text)
        await send_message(state, "user_message", {"text": text})
        
        # LLM and TTS
        await process_llm_and_tts(state, text)
    
    except Exception as e:
        logger.error(f"Text to voice error: {e}")
        await send_message(state, "error", {"message": str(e)})
    finally:
        state.is_generating = False


async def process_llm_and_tts(state: "ConnectionState", user_input: str):
    """Process LLM generation and stream TTS with incremental word streaming + MuseTalk video.
    
    When MuseTalk is enabled, audio and video are synchronized:
    - Audio is buffered until corresponding video frames are generated
    - Combined synced_av_frame messages are sent with both audio and video
    """
    loop = asyncio.get_event_loop()

    # Build prompt
    prompt = triton_client.build_prompt(
        user_input,
        state.conversation.get_history()[:-1]
    )

    await send_message(state, "llm_start", {})

    llm_response = ""

    # Queues
    token_queue: asyncio.Queue = asyncio.Queue()   # LLM tokens -> main
    tts_input_queue: Queue = Queue()               # main -> TTS (blocking, thread-safe)
    
    # Sync queue for combined audio+video (used when MuseTalk is enabled)
    sync_queue: asyncio.Queue = asyncio.Queue()    # Synced audio+video -> main

    # Wait for TTS session to be ready
    try:
        await asyncio.wait_for(state.tts_session_ready.wait(), timeout=35.0)
    except asyncio.TimeoutError:
        logger.error("TTS session not ready after timeout")
        await send_message(state, "error", {"message": "TTS session not ready"})
        return

    if state.tts_session_id is None:
        logger.error("TTS session ID is None")
        await send_message(state, "error", {"message": "TTS session not initialized"})
        return

    # Check if MuseTalk is available
    musetalk_available = (
        state.musetalk_enabled and 
        state.musetalk_session_id is not None and 
        state.musetalk_session_ready.is_set()
    )
    
    if musetalk_available:
        logger.info(f"MuseTalk enabled for session {state.musetalk_session_id}")
    else:
        logger.info("MuseTalk not available, audio-only mode")

    # Get current buffer config for adaptive buffering
    buffer_config = triton_client.get_buffer_config() if triton_client else {}
    
    await send_message(state, "tts_start", {
        "text": "",
        "video_enabled": musetalk_available,
        "buffer_config": buffer_config,
    })

    # Constants for syncing
    # At 24kHz, 40ms (one video frame at 25 FPS) = 960 samples
    SAMPLES_PER_FRAME = 960  # 40ms at 24kHz
    MUSETALK_CHUNK_SIZE = 1920  # 80ms at 24kHz (produces 2 frames)

    # ---------- Blocking workers (run in executor) ----------

    def run_llm_blocking():
        """Run LLM in a thread and push tokens into token_queue."""
        try:
            for token in triton_client.generate_llm_stream(prompt):
                if not state.is_generating:
                    break
                loop.call_soon_threadsafe(
                    token_queue.put_nowait,
                    ("token", token)
                )
            loop.call_soon_threadsafe(token_queue.put_nowait, ("done", None))
        except Exception as e:
            loop.call_soon_threadsafe(token_queue.put_nowait, ("error", str(e)))

    def run_tts_and_musetalk_synced():
        """
        Run TTS and MuseTalk in sync.
        Buffers audio, generates video frames, and sends synced messages.
        """
        try:
            audio_buffer = np.array([], dtype=np.float32)
            frame_index = 0
            # Track latest TTS metrics and word to include with synced frames
            latest_metrics = None
            latest_word = ""
            
            while True:
                item = tts_input_queue.get()
                if item is None:
                    break

                text_chunks = item

                for audio, word, metrics in triton_client.generate_tts_stream(
                    text_chunks,
                    session_id=state.tts_session_id,
                ):
                    if not state.is_generating:
                        break
                    
                    # Store latest metrics and word
                    latest_metrics = metrics
                    if word:
                        latest_word = word
                    
                    # Add audio to buffer
                    audio_buffer = np.concatenate([audio_buffer, audio])
                    
                    if musetalk_available:
                        # Process complete 80ms chunks through MuseTalk
                        while len(audio_buffer) >= MUSETALK_CHUNK_SIZE:
                            chunk = audio_buffer[:MUSETALK_CHUNK_SIZE]
                            audio_buffer = audio_buffer[MUSETALK_CHUNK_SIZE:]
                            
                            # Get video frames for this audio chunk
                            frames_for_chunk = []
                            for frame_bytes, frame_idx, timestamp_ms, mt_metrics in triton_client.send_musetalk_audio(
                                chunk,
                                session_id=state.musetalk_session_id
                            ):
                                if not state.is_generating:
                                    break
                                frames_for_chunk.append((frame_bytes, frame_idx, timestamp_ms))
                            
                            # Send synced audio+video
                            # 80ms audio = 2 frames, each frame gets 40ms of audio
                            if frames_for_chunk:
                                # Split audio chunk into per-frame segments
                                for i, (frame_bytes, f_idx, ts_ms) in enumerate(frames_for_chunk):
                                    audio_start = i * SAMPLES_PER_FRAME
                                    audio_end = (i + 1) * SAMPLES_PER_FRAME
                                    frame_audio = chunk[audio_start:audio_end]
                                    
                                    audio_b64 = base64.b64encode(frame_audio.tobytes()).decode("utf-8")
                                    frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
                                    
                                    # Build message with metrics
                                    msg_data = {
                                        "audio": audio_b64,
                                        "frame": frame_b64,
                                        "frame_index": frame_index,
                                        "timestamp_ms": frame_index * 40.0,  # 40ms per frame
                                        "word": latest_word if i == 0 else "",
                                    }
                                    # Include TTS metrics if available
                                    if latest_metrics and i == 0:
                                        msg_data["rtf"] = round(latest_metrics.rtf, 3)
                                        msg_data["generation_time_ms"] = round(latest_metrics.generation_time_ms, 1)
                                        msg_data["audio_duration_ms"] = round(latest_metrics.audio_duration_ms, 1)
                                    
                                    loop.call_soon_threadsafe(
                                        sync_queue.put_nowait,
                                        ("synced", msg_data)
                                    )
                                    frame_index += 1
                                    # Clear word after first frame of chunk
                                    if i == 0:
                                        latest_word = ""
                            else:
                                # No frames generated, send audio only
                                audio_b64 = base64.b64encode(chunk.tobytes()).decode("utf-8")
                                msg_data = {
                                    "audio": audio_b64,
                                    "word": latest_word,
                                }
                                if latest_metrics:
                                    msg_data["rtf"] = round(latest_metrics.rtf, 3)
                                    msg_data["generation_time_ms"] = round(latest_metrics.generation_time_ms, 1)
                                    msg_data["audio_duration_ms"] = round(latest_metrics.audio_duration_ms, 1)
                                loop.call_soon_threadsafe(
                                    sync_queue.put_nowait,
                                    ("audio_only", msg_data)
                                )
                                latest_word = ""
                    else:
                        # No MuseTalk - send audio directly
                        audio_b64 = base64.b64encode(audio.tobytes()).decode("utf-8")
                        loop.call_soon_threadsafe(
                            sync_queue.put_nowait,
                            ("audio_only", {
                                "audio": audio_b64,
                                "word": word,
                                "rtf": round(metrics.rtf, 3),
                                "generation_time_ms": round(metrics.generation_time_ms, 1),
                                "audio_duration_ms": round(metrics.audio_duration_ms, 1),
                            })
                        )

            # Process any remaining audio in buffer
            if len(audio_buffer) > 0 and state.is_generating:
                if musetalk_available:
                    # Pad to chunk size if needed
                    if len(audio_buffer) < MUSETALK_CHUNK_SIZE:
                        padded_audio = np.pad(audio_buffer, (0, MUSETALK_CHUNK_SIZE - len(audio_buffer)))
                    else:
                        padded_audio = audio_buffer
                    
                    frames_for_chunk = []
                    for frame_bytes, f_idx, ts_ms, mt_metrics in triton_client.send_musetalk_audio(
                        padded_audio[:MUSETALK_CHUNK_SIZE],
                        session_id=state.musetalk_session_id
                    ):
                        if not state.is_generating:
                            break
                        frames_for_chunk.append((frame_bytes, f_idx, ts_ms))
                    
                    # Send remaining synced frames
                    remaining_audio = audio_buffer
                    for i, (frame_bytes, f_idx, ts_ms) in enumerate(frames_for_chunk):
                        audio_start = i * SAMPLES_PER_FRAME
                        audio_end = min((i + 1) * SAMPLES_PER_FRAME, len(remaining_audio))
                        frame_audio = remaining_audio[audio_start:audio_end] if audio_start < len(remaining_audio) else np.array([], dtype=np.float32)
                        
                        if len(frame_audio) > 0:
                            audio_b64 = base64.b64encode(frame_audio.tobytes()).decode("utf-8")
                        else:
                            audio_b64 = ""
                        frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
                        
                        loop.call_soon_threadsafe(
                            sync_queue.put_nowait,
                            ("synced", {
                                "audio": audio_b64,
                                "frame": frame_b64,
                                "frame_index": frame_index,
                                "timestamp_ms": frame_index * 40.0,
                                "word": "",
                            })
                        )
                        frame_index += 1
                else:
                    # Send remaining audio without video
                    audio_b64 = base64.b64encode(audio_buffer.tobytes()).decode("utf-8")
                    loop.call_soon_threadsafe(
                        sync_queue.put_nowait,
                        ("audio_only", {
                            "audio": audio_b64,
                            "word": "",
                        })
                    )
            
            loop.call_soon_threadsafe(sync_queue.put_nowait, ("done", None))
            
        except Exception as e:
            logger.error(f"TTS/MuseTalk sync thread error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            loop.call_soon_threadsafe(sync_queue.put_nowait, ("error", str(e)))

    # ---------- Async consumer ----------

    async def sync_consumer():
        """Consume synced audio+video and forward to client."""
        try:
            while state.is_generating:
                try:
                    msg_type, data = await asyncio.wait_for(sync_queue.get(), timeout=120.0)
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for synced data")
                    break

                if msg_type == "synced":
                    # Send combined audio+video frame
                    await send_message(state, "synced_av_frame", data)
                elif msg_type == "audio_only":
                    # Send audio only (when MuseTalk disabled or no frames)
                    await send_message(state, "tts_audio", data)
                elif msg_type == "done":
                    break
                elif msg_type == "error":
                    logger.error(f"Sync error: {data}")
                    break
        finally:
            await send_message(state, "tts_complete", {})
            if musetalk_available:
                await send_message(state, "video_complete", {})
            logger.info("TTS/Video complete")

    try:
        # Start workers in background threads
        llm_future = loop.run_in_executor(None, run_llm_blocking)
        sync_future = loop.run_in_executor(None, run_tts_and_musetalk_synced)

        # Start consumer
        sync_task = asyncio.create_task(sync_consumer())

        # State for word-based TTS streaming
        tts_started = False
        words_sent_to_tts = 0

        # ---------- Main LLM token loop ----------
        while state.is_generating:
            try:
                msg_type, data = await asyncio.wait_for(token_queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for LLM token")
                break

            if msg_type == "token":
                token = data
                llm_response += token

                await send_message(state, "llm_token", {
                    "token": token,
                    "full_text": llm_response,
                })

                words = llm_response.split()
                num_words = len(words)

                if (not tts_started) and num_words >= 4:
                    first_chunk = " ".join(words[:3])
                    tts_input_queue.put([first_chunk])
                    tts_started = True
                    words_sent_to_tts = 3

                if tts_started and num_words > words_sent_to_tts:
                    new_words = words[words_sent_to_tts:]
                    for w in new_words:
                        tts_input_queue.put([" " + w])
                    words_sent_to_tts = num_words

            elif msg_type == "done":
                break
            elif msg_type == "error":
                logger.error(f"LLM error: {data}")
                break

        await llm_future

        await send_message(state, "llm_complete", {"text": llm_response})
        logger.info(f"LLM complete: {llm_response[:50]}...")

        state.conversation.add_assistant_message(llm_response)

        if tts_started:
            tts_input_queue.put([""])
            tts_input_queue.put([""])

        tts_input_queue.put(None)

        await sync_future
        await sync_task

        # Reinitialize sessions for next request
        asyncio.create_task(init_tts_cache_async(state))
        if musetalk_available:
            asyncio.create_task(init_musetalk_session_async(state))

    except Exception as e:
        logger.error(f"LLM/TTS/MuseTalk error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await send_message(state, "error", {"message": str(e)})

async def process_tts_only(state: ConnectionState, text: str):
    """Process direct TTS synthesis with synchronized MuseTalk video.
    
    Audio and video are synced - each synced_av_frame contains 40ms of audio
    paired with its corresponding video frame.
    """
    if triton_client is None:
        return
    
    state.is_generating = True
    loop = asyncio.get_event_loop()

    text_chunks = _split_text_for_streaming(text)
    try:
        # Wait for TTS session to be ready (with timeout)
        try:
            await asyncio.wait_for(state.tts_session_ready.wait(), timeout=35.0)
        except asyncio.TimeoutError:
            logger.error("TTS session not ready after timeout")
            await send_message(state, "error", {"message": "TTS session not ready"})
            return
        
        # Double check session is valid
        if state.tts_session_id is None:
            logger.error("TTS session ID is None after ready signal")
            await send_message(state, "error", {"message": "TTS session not initialized"})
            return
        
        # Check if MuseTalk is available
        musetalk_available = False
        if state.musetalk_enabled and state.musetalk_session_id is not None:
            # Wait briefly for MuseTalk session to be ready; if not, fall back to audio-only
            try:
                await asyncio.wait_for(state.musetalk_session_ready.wait(), timeout=5.0)
                musetalk_available = state.musetalk_session_ready.is_set()
            except asyncio.TimeoutError:
                logger.warning("MuseTalk session not ready in time; continuing without video for this request")
                musetalk_available = False
    
        # Get current buffer config for adaptive buffering
        buffer_config = triton_client.get_buffer_config() if triton_client else {}
        
        await send_message(state, "tts_start", {
            "text": text,
            "video_enabled": musetalk_available,
            "buffer_config": buffer_config,
        })
        
        # Sync queue for combined audio+video
        sync_queue: asyncio.Queue = asyncio.Queue()
        
        # Constants for syncing
        SAMPLES_PER_FRAME = 960  # 40ms at 24kHz
        MUSETALK_CHUNK_SIZE = 1920  # 80ms at 24kHz (produces 2 frames)
        
        def run_tts_and_musetalk_synced():
            """Run TTS and MuseTalk in sync."""
            try:
                audio_buffer = np.array([], dtype=np.float32)
                frame_index = 0
                latest_metrics = None
                latest_word = ""
                
                for audio, word, metrics in triton_client.generate_tts_stream(text_chunks, session_id=state.tts_session_id):
                    if not state.is_generating:
                        break
                    
                    # Store latest metrics and word
                    latest_metrics = metrics
                    if word:
                        latest_word = word
                    
                    # Add audio to buffer
                    audio_buffer = np.concatenate([audio_buffer, audio])
                    
                    if musetalk_available:
                        # Process complete 80ms chunks through MuseTalk
                        while len(audio_buffer) >= MUSETALK_CHUNK_SIZE:
                            chunk = audio_buffer[:MUSETALK_CHUNK_SIZE]
                            audio_buffer = audio_buffer[MUSETALK_CHUNK_SIZE:]
                            
                            # Get video frames for this audio chunk
                            frames_for_chunk = []
                            for frame_bytes, frame_idx, timestamp_ms, mt_metrics in triton_client.send_musetalk_audio(
                                chunk,
                                session_id=state.musetalk_session_id
                            ):
                                if not state.is_generating:
                                    break
                                frames_for_chunk.append((frame_bytes, frame_idx, timestamp_ms))
                            
                            # Send synced audio+video
                            if frames_for_chunk:
                                for i, (frame_bytes, f_idx, ts_ms) in enumerate(frames_for_chunk):
                                    audio_start = i * SAMPLES_PER_FRAME
                                    audio_end = (i + 1) * SAMPLES_PER_FRAME
                                    frame_audio = chunk[audio_start:audio_end]
                                    
                                    audio_b64 = base64.b64encode(frame_audio.tobytes()).decode("utf-8")
                                    frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
                                    
                                    # Build message with metrics
                                    msg_data = {
                                        "audio": audio_b64,
                                        "frame": frame_b64,
                                        "frame_index": frame_index,
                                        "timestamp_ms": frame_index * 40.0,
                                        "word": latest_word if i == 0 else "",
                                    }
                                    # Include TTS metrics if available
                                    if latest_metrics and i == 0:
                                        msg_data["rtf"] = round(latest_metrics.rtf, 3)
                                        msg_data["generation_time_ms"] = round(latest_metrics.generation_time_ms, 1)
                                        msg_data["audio_duration_ms"] = round(latest_metrics.audio_duration_ms, 1)
                                    
                                    loop.call_soon_threadsafe(
                                        sync_queue.put_nowait,
                                        ("synced", msg_data)
                                    )
                                    frame_index += 1
                                    if i == 0:
                                        latest_word = ""
                            else:
                                # No frames generated, send audio only
                                audio_b64 = base64.b64encode(chunk.tobytes()).decode("utf-8")
                                msg_data = {
                                    "audio": audio_b64,
                                    "word": latest_word,
                                }
                                if latest_metrics:
                                    msg_data["rtf"] = round(latest_metrics.rtf, 3)
                                    msg_data["generation_time_ms"] = round(latest_metrics.generation_time_ms, 1)
                                    msg_data["audio_duration_ms"] = round(latest_metrics.audio_duration_ms, 1)
                                loop.call_soon_threadsafe(
                                    sync_queue.put_nowait,
                                    ("audio_only", msg_data)
                                )
                                latest_word = ""
                    else:
                        # No MuseTalk - send audio directly
                        audio_b64 = base64.b64encode(audio.tobytes()).decode("utf-8")
                        loop.call_soon_threadsafe(
                            sync_queue.put_nowait,
                            ("audio_only", {
                                "audio": audio_b64,
                                "word": word,
                                "rtf": round(metrics.rtf, 3),
                                "generation_time_ms": round(metrics.generation_time_ms, 1),
                                "audio_duration_ms": round(metrics.audio_duration_ms, 1),
                            })
                        )
                
                # Process remaining audio
                if len(audio_buffer) > 0 and state.is_generating:
                    if musetalk_available:
                        if len(audio_buffer) < MUSETALK_CHUNK_SIZE:
                            padded_audio = np.pad(audio_buffer, (0, MUSETALK_CHUNK_SIZE - len(audio_buffer)))
                        else:
                            padded_audio = audio_buffer
                        
                        frames_for_chunk = []
                        for frame_bytes, f_idx, ts_ms, mt_metrics in triton_client.send_musetalk_audio(
                            padded_audio[:MUSETALK_CHUNK_SIZE],
                            session_id=state.musetalk_session_id
                        ):
                            if not state.is_generating:
                                break
                            frames_for_chunk.append((frame_bytes, f_idx, ts_ms))
                        
                        remaining_audio = audio_buffer
                        for i, (frame_bytes, f_idx, ts_ms) in enumerate(frames_for_chunk):
                            audio_start = i * SAMPLES_PER_FRAME
                            audio_end = min((i + 1) * SAMPLES_PER_FRAME, len(remaining_audio))
                            frame_audio = remaining_audio[audio_start:audio_end] if audio_start < len(remaining_audio) else np.array([], dtype=np.float32)
                            
                            if len(frame_audio) > 0:
                                audio_b64 = base64.b64encode(frame_audio.tobytes()).decode("utf-8")
                            else:
                                audio_b64 = ""
                            frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
                            
                            loop.call_soon_threadsafe(
                                sync_queue.put_nowait,
                                ("synced", {
                                    "audio": audio_b64,
                                    "frame": frame_b64,
                                    "frame_index": frame_index,
                                    "timestamp_ms": frame_index * 40.0,
                                    "word": "",
                                })
                            )
                            frame_index += 1
                    else:
                        audio_b64 = base64.b64encode(audio_buffer.tobytes()).decode("utf-8")
                        loop.call_soon_threadsafe(
                            sync_queue.put_nowait,
                            ("audio_only", {
                                "audio": audio_b64,
                                "word": "",
                            })
                        )
                
                loop.call_soon_threadsafe(sync_queue.put_nowait, ("done", None))
                
            except Exception as e:
                logger.error(f"TTS/MuseTalk sync thread error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                loop.call_soon_threadsafe(sync_queue.put_nowait, ("error", str(e)))

        async def sync_consumer():
            """Consume synced audio+video and forward to client."""
            try:
                while state.is_generating:
                    try:
                        msg_type, data = await asyncio.wait_for(sync_queue.get(), timeout=120.0)
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for synced data")
                        break

                    if msg_type == "synced":
                        await send_message(state, "synced_av_frame", data)
                    elif msg_type == "audio_only":
                        await send_message(state, "tts_audio", data)
                    elif msg_type == "done":
                        break
                    elif msg_type == "error":
                        logger.error(f"Sync error: {data}")
                        break
            finally:
                await send_message(state, "tts_complete", {})
                if musetalk_available:
                    await send_message(state, "video_complete", {})
                logger.info("TTS only complete")

        # Start workers
        sync_future = loop.run_in_executor(None, run_tts_and_musetalk_synced)
        sync_task = asyncio.create_task(sync_consumer())
        
        # Wait for completion
        await sync_future
        await sync_task
        
        # Reinitialize TTS session for next generation; keep MuseTalk session if already ready
        asyncio.create_task(init_tts_cache_async(state))
        if not state.musetalk_session_ready.is_set() or state.musetalk_session_id is None:
            asyncio.create_task(init_musetalk_session_async(state))
    
    except Exception as e:
        logger.error(f"TTS only error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await send_message(state, "error", {"message": str(e)})
    finally:
        state.is_generating = False




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
