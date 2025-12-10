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
    TTSSession
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
        
        # Initialize TTS cache on connection
        if triton_client:
            # Run cache init in background
            asyncio.create_task(init_tts_cache_async(state))
        
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
    """Process LLM generation and stream TTS with incremental word streaming."""
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
    audio_queue: asyncio.Queue = asyncio.Queue()   # TTS audio -> main
    tts_input_queue: Queue = Queue()               # main -> TTS (blocking, thread-safe)

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

    await send_message(state, "tts_start", {"text": ""})

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

    def run_tts_blocking():
        """
        Run TTS in a thread.

        It waits for text-chunk lists from tts_input_queue.
        For each list, it calls generate_tts_stream and pushes audio chunks to audio_queue.
        """
        try:
            while True:
                item = tts_input_queue.get()
                if item is None:
                    # Sentinel -> finish this TTS session
                    break

                text_chunks = item  # List[str]

                for audio, word, metrics in triton_client.generate_tts_stream(
                    text_chunks,
                    session_id=state.tts_session_id,
                ):
                    if not state.is_generating:
                        break
                    loop.call_soon_threadsafe(
                        audio_queue.put_nowait,
                        ("audio", (audio.copy(), word, metrics))
                    )

            loop.call_soon_threadsafe(audio_queue.put_nowait, ("done", None))
        except Exception as e:
            logger.error(f"TTS thread error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            loop.call_soon_threadsafe(audio_queue.put_nowait, ("error", str(e)))

    # ---------- Async consumer for TTS audio ----------

    async def audio_consumer():
        """Consume TTS audio_queue and forward to the client in real time."""
        try:
            while state.is_generating:
                try:
                    msg_type, data = await asyncio.wait_for(audio_queue.get(), timeout=120.0)
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for TTS audio")
                    break

                if msg_type == "audio":
                    audio, word, metrics = data
                    audio_b64 = base64.b64encode(audio.tobytes()).decode("utf-8")

                    await send_message(state, "tts_audio", {
                        "audio": audio_b64,
                        "word": word,
                        "rtf": round(metrics.rtf, 3),
                        "generation_time_ms": round(metrics.generation_time_ms, 1),
                        "audio_duration_ms": round(metrics.audio_duration_ms, 1),
                    })
                elif msg_type == "done":
                    break
                elif msg_type == "error":
                    logger.error(f"TTS error: {data}")
                    break
        finally:
            await send_message(state, "tts_complete", {})
            logger.info("TTS complete")

    try:
        # Start workers in background threads
        llm_future = loop.run_in_executor(None, run_llm_blocking)
        tts_future = loop.run_in_executor(None, run_tts_blocking)

        # Start audio consumer
        audio_task = asyncio.create_task(audio_consumer())

        # State for word-based TTS streaming
        tts_started = False
        words_sent_to_tts = 0  # number of words already sent to TTS

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

                # Stream token to client
                await send_message(state, "llm_token", {
                    "token": token,
                    "full_text": llm_response,
                })

                # Word-based logic for TTS
                words = llm_response.split()
                num_words = len(words)

                # Heuristic: start TTS once we have at least 3 complete words.
                # We trigger at >= 4 to be safer with subwords.
                if (not tts_started) and num_words >= 4:
                    # First chunk: first 3 words
                    first_chunk = " ".join(words[:3])
                    tts_input_queue.put([first_chunk])
                    tts_started = True
                    words_sent_to_tts = 3

                # After starting, send each newly completed word separately
                if tts_started and num_words > words_sent_to_tts:
                    new_words = words[words_sent_to_tts:]
                    for w in new_words:
                        # Send last word only, with a leading space
                        tts_input_queue.put([" " + w])
                    words_sent_to_tts = num_words

            elif msg_type == "done":
                break
            elif msg_type == "error":
                logger.error(f"LLM error: {data}")
                break

        # Ensure LLM thread is finished
        await llm_future

        # Final LLM response to client
        await send_message(state, "llm_complete", {"text": llm_response})
        logger.info(f"LLM complete: {llm_response[:50]}...")

        # Add to conversation history
        state.conversation.add_assistant_message(llm_response)

        # Flush remaining TTS words:
        # With 2-word lookahead, we need two empty chunks to emit audio for the last two words.
        if tts_started:
            tts_input_queue.put([""])
            tts_input_queue.put([""])

        # Sentinel to stop the TTS thread
        tts_input_queue.put(None)

        # Wait for TTS worker and audio consumer
        await tts_future
        await audio_task

        # Reinitialize TTS cache for next request
        asyncio.create_task(init_tts_cache_async(state))

    except Exception as e:
        logger.error(f"LLM/TTS error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await send_message(state, "error", {"message": str(e)})

async def process_tts_only(state: ConnectionState, text: str):
    """Process direct TTS synthesis"""
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
        
        await send_message(state, "tts_start", {"text": text})
        
        audio_queue = asyncio.Queue()
        
        def run_tts_blocking():
            """Run TTS in thread and put audio chunks in queue"""
            try:
                for audio, word, metrics in triton_client.generate_tts_stream(text_chunks, session_id=state.tts_session_id):
                    if not state.is_generating:
                        break
                    loop.call_soon_threadsafe(
                        audio_queue.put_nowait, 
                        ("audio", (audio.copy(), word, metrics))
                    )
                loop.call_soon_threadsafe(audio_queue.put_nowait, ("done", None))
            except Exception as e:
                logger.error(f"TTS thread error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                loop.call_soon_threadsafe(audio_queue.put_nowait, ("error", str(e)))
        
        # Start TTS generation in background thread
        tts_future = loop.run_in_executor(None, run_tts_blocking)
        
        # Stream audio as it arrives
        while state.is_generating:
            try:
                msg_type, data = await asyncio.wait_for(audio_queue.get(), timeout=120.0)
                
                if msg_type == "audio":
                    audio, word, metrics = data
                    audio_b64 = base64.b64encode(audio.tobytes()).decode("utf-8")
                    
                    await send_message(state, "tts_audio", {
                        "audio": audio_b64,
                        "word": word,
                        "rtf": round(metrics.rtf, 3),
                        "generation_time_ms": round(metrics.generation_time_ms, 1),
                        "audio_duration_ms": round(metrics.audio_duration_ms, 1)
                    })
                elif msg_type == "done":
                    break
                elif msg_type == "error":
                    logger.error(f"TTS error: {data}")
                    break
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for TTS audio")
                break
        
        # Wait for TTS thread to finish
        await tts_future
        
        await send_message(state, "tts_complete", {})
        logger.info("TTS only complete")
        
        # Reinitialize TTS cache for next generation
        asyncio.create_task(init_tts_cache_async(state))
    
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

