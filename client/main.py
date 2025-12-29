"""
FastAPI Backend for Voice Assistant
WebSocket-based streaming for VAD, STT, LLM, and TTS
"""

import asyncio
import json
import logging
import os
import time
import base64
import sys
import signal
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
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global exception handler to catch uncaught exceptions
def global_exception_handler(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Uncaught exception!", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = global_exception_handler

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


async def run_tts_with_musetalk(
    state: "ConnectionState",
    tts_input_queue: Queue,
    musetalk_available: bool,
):
    """
    Shared TTS + MuseTalk synchronization logic.
    
    Consumes text chunks from tts_input_queue, generates audio via TTS,
    optionally generates video frames via MuseTalk, and sends synced 
    audio+video frames to the client.
    MuseTalk runs in a stateless, chunked streaming mode with configurable
    start buffering and lookahead to avoid future padding artifacts.
    
    Args:
        state: The connection state
        tts_input_queue: Queue of text chunks (list of strings). Send None to terminate.
        musetalk_available: Whether MuseTalk video generation is enabled
    """
    loop = asyncio.get_event_loop()
    
    # Constants for syncing
    SAMPLES_PER_FRAME = 960  # 40ms at 24kHz
    
    # Sync queue for combined audio+video
    sync_queue: asyncio.Queue = asyncio.Queue()
    
    def run_tts_and_musetalk_synced():
        """Run TTS and MuseTalk in sync in a background thread."""
        try:
            all_audio = []  # TTS audio chunks (80ms each)
            all_words = []  # Word per audio chunk
            all_metrics = []  # Metrics per audio chunk
            word_sample_boundaries = []  # Cumulative samples per chunk
            chunks_received = 0
            audio_chunks_generated = 0
            total_samples = 0

            # MuseTalk streaming controls (chunk-based)
            mt_start_after = 0
            mt_lookahead = 0
            if triton_client is not None:
                mt_start_after = max(0, int(getattr(triton_client.musetalk_params, "start_after_chunks", 0)))
                mt_lookahead = max(0, int(getattr(triton_client.musetalk_params, "lookahead_chunks", 0)))

            base_frame_index = state.musetalk_frame_index
            frames_sent = 0  # Frames sent in this session
            processed_chunk_index = 0  # Chunks already sent to MuseTalk
            word_idx = 0

            logger.info(
                "[TTS_WORKER] MuseTalk buffering: start_after_chunks=%s, lookahead_chunks=%s",
                mt_start_after,
                mt_lookahead,
            )
            
            logger.info(f"[TTS_WORKER] Started, session_id={state.tts_session_id}, musetalk={musetalk_available}")

            def samples_before_chunk(chunk_idx: int) -> int:
                if chunk_idx <= 0:
                    return 0
                return word_sample_boundaries[chunk_idx - 1]

            def enqueue_synced_frame(
                frame_bytes: bytes,
                frame_idx: int,
                timestamp_ms: float,
                segment_audio: np.ndarray,
                segment_start_samples: int,
                frame_audio_start: int,
                allow_partial_last: bool,
                segment_start_ms: float,
            ) -> None:
                nonlocal frames_sent, word_idx

                expected_idx = base_frame_index + frames_sent
                if frame_idx < expected_idx:
                    return
                if frame_idx > expected_idx:
                    logger.warning(
                        "[TTS_WORKER] Frame index gap: expected=%s got=%s",
                        expected_idx,
                        frame_idx,
                    )
                    frames_sent = frame_idx - base_frame_index

                audio_start = int(frame_audio_start)
                audio_end = audio_start + SAMPLES_PER_FRAME
                if allow_partial_last:
                    audio_end = min(audio_end, len(segment_audio))
                else:
                    audio_end = min(audio_end, len(segment_audio), audio_start + SAMPLES_PER_FRAME)

                if audio_start < len(segment_audio):
                    frame_audio = segment_audio[audio_start:audio_end]
                    audio_b64 = base64.b64encode(frame_audio.tobytes()).decode("utf-8")
                else:
                    audio_b64 = ""

                frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
                global_audio_start = segment_start_samples + audio_start

                word = ""
                metrics_data = {}
                while word_idx < len(word_sample_boundaries) and global_audio_start >= word_sample_boundaries[word_idx]:
                    word_idx += 1
                if word_idx < len(all_words) and all_words[word_idx]:
                    word = all_words[word_idx]
                    all_words[word_idx] = ""  # Clear so we don't repeat
                    if word_idx < len(all_metrics):
                        metrics_data = {
                            "rtf": round(all_metrics[word_idx].rtf, 3),
                            "generation_time_ms": round(all_metrics[word_idx].generation_time_ms, 1),
                            "audio_duration_ms": round(all_metrics[word_idx].audio_duration_ms, 1),
                        }

                msg_data = {
                    "audio": audio_b64,
                    "frame": frame_b64,
                    "frame_index": frame_idx,
                    "timestamp_ms": segment_start_ms + timestamp_ms,
                    "word": word,
                    **metrics_data,
                }

                loop.call_soon_threadsafe(
                    sync_queue.put_nowait,
                    ("synced", msg_data)
                )
                frames_sent = frame_idx - base_frame_index + 1

            def process_musetalk_segment(flush: bool = False) -> None:
                nonlocal processed_chunk_index, frames_sent, word_idx

                if not musetalk_available or not state.is_generating:
                    return

                total_chunks = len(all_audio)
                if total_chunks == 0:
                    return

                if not flush and total_chunks < mt_start_after:
                    return

                output_end_chunk = total_chunks if flush else max(0, total_chunks - mt_lookahead)
                if output_end_chunk <= processed_chunk_index:
                    return

                overlap_chunks = 1 if processed_chunk_index > 0 else 0
                segment_start_chunk = max(0, processed_chunk_index - overlap_chunks)
                segment_end_chunk = total_chunks

                if segment_end_chunk <= segment_start_chunk:
                    processed_chunk_index = output_end_chunk
                    return

                segment_audio = np.concatenate(all_audio[segment_start_chunk:segment_end_chunk])
                if segment_audio.size == 0:
                    processed_chunk_index = output_end_chunk
                    return

                segment_start_samples = samples_before_chunk(segment_start_chunk)
                overlap_samples = 0
                if processed_chunk_index > segment_start_chunk:
                    overlap_samples = samples_before_chunk(processed_chunk_index) - segment_start_samples
                output_end_samples = samples_before_chunk(output_end_chunk) - segment_start_samples

                segment_start_frame_index = base_frame_index + int(segment_start_samples // SAMPLES_PER_FRAME)
                segment_start_ms = segment_start_samples / 24.0

                for frame_bytes, frame_idx, timestamp_ms, _ in triton_client.generate_musetalk_frames(
                    segment_audio,
                    frame_index=segment_start_frame_index,
                ):
                    if not state.is_generating:
                        break

                    local_frame_idx = frame_idx - segment_start_frame_index
                    frame_audio_start = local_frame_idx * SAMPLES_PER_FRAME
                    if frame_audio_start < overlap_samples:
                        continue
                    if not flush and (frame_audio_start + SAMPLES_PER_FRAME) > output_end_samples:
                        continue
                    if flush and frame_audio_start >= output_end_samples:
                        continue

                    enqueue_synced_frame(
                        frame_bytes,
                        frame_idx,
                        timestamp_ms,
                        segment_audio,
                        segment_start_samples,
                        frame_audio_start,
                        allow_partial_last=flush,
                        segment_start_ms=segment_start_ms,
                    )

                processed_chunk_index = output_end_chunk
                state.musetalk_frame_index = base_frame_index + frames_sent

            # Phase 1: Generate TTS audio and stream MuseTalk as chunks arrive
            while True:
                item = tts_input_queue.get()
                if item is None:
                    logger.info(f"[TTS_WORKER] Received None, TTS generation complete. chunks_received={chunks_received}, audio_generated={audio_chunks_generated}")
                    break
                
                text_chunks = item
                chunks_received += 1
                
                if chunks_received <= 5 or chunks_received % 10 == 0:
                    logger.info(f"[TTS_WORKER] Processing chunk #{chunks_received}: {text_chunks}")
                
                for audio, word, metrics in triton_client.generate_tts_stream(
                    text_chunks,
                    session_id=state.tts_session_id,
                ):
                    audio_chunks_generated += 1
                    if audio_chunks_generated <= 3:
                        logger.info(f"[TTS_WORKER] Audio chunk #{audio_chunks_generated}, samples={len(audio)}, word={word}")
                    
                    if not state.is_generating:
                        logger.info(f"[TTS_WORKER] state.is_generating=False, breaking")
                        break
                    
                    all_audio.append(audio)
                    all_words.append(word)
                    all_metrics.append(metrics)
                    total_samples += len(audio)
                    word_sample_boundaries.append(total_samples)
                    
                    # If MuseTalk not available, send audio immediately
                    if not musetalk_available:
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
                    else:
                        process_musetalk_segment(flush=False)
            
            # Phase 2: Flush remaining MuseTalk frames without lookahead
            if musetalk_available and len(all_audio) > 0 and state.is_generating:
                process_musetalk_segment(flush=True)
            
            logger.info(f"[TTS_WORKER] Finished. Total chunks_received={chunks_received}, audio_chunks_generated={audio_chunks_generated}")
            loop.call_soon_threadsafe(sync_queue.put_nowait, ("done", None))
            
        except Exception as e:
            logger.error(f"[TTS_WORKER] Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            loop.call_soon_threadsafe(sync_queue.put_nowait, ("error", str(e)))
    
    async def sync_consumer():
        """Consume synced audio+video and forward to client."""
        messages_sent = 0
        synced_frames = 0
        audio_only_msgs = 0
        try:
            logger.info(f"[SYNC_CONSUMER] Started, musetalk={musetalk_available}")
            while state.is_generating:
                try:
                    msg_type, data = await asyncio.wait_for(sync_queue.get(), timeout=120.0)
                except asyncio.TimeoutError:
                    logger.warning("[SYNC_CONSUMER] Timeout waiting for synced data")
                    break

                if msg_type == "synced":
                    synced_frames += 1
                    await send_message(state, "synced_av_frame", data)
                    messages_sent += 1
                elif msg_type == "audio_only":
                    audio_only_msgs += 1
                    await send_message(state, "tts_audio", data)
                    messages_sent += 1
                elif msg_type == "done":
                    logger.info(f"[SYNC_CONSUMER] Got 'done', stopping")
                    break
                elif msg_type == "error":
                    logger.error(f"[SYNC_CONSUMER] Error: {data}")
                    break
        finally:
            logger.info(f"[SYNC_CONSUMER] Finished. synced_frames={synced_frames}, audio_only_msgs={audio_only_msgs}, total_sent={messages_sent}")
            # Only send completion messages if connection is still active
            if state.is_connected:
                await send_message(state, "tts_complete", {})
                if musetalk_available:
                    await send_message(state, "video_complete", {})
                logger.info("[SYNC_CONSUMER] TTS/Video complete sent")
            else:
                logger.info("[SYNC_CONSUMER] Connection closed, skipping completion messages")
    
    # Start worker in background thread
    sync_future = loop.run_in_executor(None, run_tts_and_musetalk_synced)
    
    # Start consumer task
    sync_task = asyncio.create_task(sync_consumer())
    
    # Return futures so caller can await them
    return sync_future, sync_task


def handle_asyncio_exception(loop, context):
    """Global asyncio exception handler"""
    msg = context.get("exception", context["message"])
    logger.error(f"Asyncio exception: {msg}")
    if "exception" in context:
        import traceback
        logger.error("".join(traceback.format_exception(type(context["exception"]), context["exception"], context["exception"].__traceback__)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global triton_client
    
    # Set asyncio exception handler
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(handle_asyncio_exception)
    
    # Startup
    triton_url = os.environ.get("TRITON_URL", "localhost:8001")
    logger.info(f"Initializing Triton client with URL: {triton_url}")
    triton_client = TritonVoiceClient(
        triton_url=triton_url,
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
        },
        "musetalk": {
            "start_after_chunks": triton_client.musetalk_params.start_after_chunks,
            "lookahead_chunks": triton_client.musetalk_params.lookahead_chunks,
        },
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

    # Update MuseTalk params
    if "musetalk" in config:
        mt_cfg = config["musetalk"]
        if "start_after_chunks" in mt_cfg:
            try:
                triton_client.musetalk_params.start_after_chunks = max(0, int(mt_cfg["start_after_chunks"]))
            except (TypeError, ValueError):
                logger.warning(f"Ignoring invalid start_after_chunks: {mt_cfg['start_after_chunks']}")
        if "lookahead_chunks" in mt_cfg:
            try:
                triton_client.musetalk_params.lookahead_chunks = max(0, int(mt_cfg["lookahead_chunks"]))
            except (TypeError, ValueError):
                logger.warning(f"Ignoring invalid lookahead_chunks: {mt_cfg['lookahead_chunks']}")
    
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
        
        # Audio processing queue (decoupled from WS receive loop)
        self.audio_queue: asyncio.Queue = asyncio.Queue(maxsize=100)  # ~3.2s buffer at 32ms chunks
        self.audio_processor_task: Optional[asyncio.Task] = None
        
        # MuseTalk state (stateless - only track frame index for video continuity)
        self.musetalk_enabled = True  # Can be disabled if model not available
        self.musetalk_frame_index: int = 0  # Track frame index for avatar animation continuity
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
        error_msg = str(e) if str(e) else type(e).__name__
        logger.error(f"Failed to send message '{msg_type}': {error_msg}")


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
        
        # Check if MuseTalk is available and send ready message with idle frame
        musetalk_available = await check_musetalk_available()
        if musetalk_available:
            # Get idle frame in background to not block connection
            loop = asyncio.get_event_loop()
            idle_frame = await loop.run_in_executor(None, triton_client.get_musetalk_idle_frame)
            buffer_config = triton_client.get_buffer_config()
            
            await send_message(websocket, "musetalk_ready", {
                "success": True,
                "session_id": None,  # Stateless - no session
                "idle_frame": base64.b64encode(idle_frame).decode("utf-8") if idle_frame else None,
                "buffer_config": buffer_config,
            })
            logger.info(f"MuseTalk ready (stateless), idle_frame={'yes' if idle_frame else 'no'}")
        else:
            state.musetalk_enabled = False
            await send_message(websocket, "musetalk_ready", {
                "success": False,
                "session_id": None,
                "reason": "model_unavailable"
            })
            logger.info("MuseTalk not available")
        
        logger.info("WebSocket ready - TTS will be initialized on-demand")
        
        # Start audio processor task (decoupled from receive loop)
        state.audio_processor_task = asyncio.create_task(audio_processor_loop(state))
        
        # Main message loop - fast receive, no blocking
        while True:
            message = await websocket.receive()
            
            # Check for disconnect message type
            if message.get("type") == "websocket.disconnect":
                logger.info(f"WebSocket disconnect message received: {connection_id}")
                break
            
            if "text" in message:
                await handle_text_message(state, json.loads(message["text"]))
            elif "bytes" in message:
                # Non-blocking enqueue - drop if queue is full (backpressure)
                try:
                    state.audio_queue.put_nowait(message["bytes"])
                except asyncio.QueueFull:
                    # Queue full - drop oldest and add new (sliding window)
                    try:
                        state.audio_queue.get_nowait()
                        state.audio_queue.put_nowait(message["bytes"])
                    except asyncio.QueueEmpty:
                        pass
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Mark connection as closed first (so async tasks know to stop)
        state.is_connected = False
        state.is_generating = False  # Stop any ongoing generation
        
        # Cancel audio processor task
        if state.audio_processor_task is not None:
            state.audio_processor_task.cancel()
            try:
                await state.audio_processor_task
            except asyncio.CancelledError:
                pass
        
        # Clean up TTS session on disconnect
        if state.tts_session_id is not None and triton_client is not None:
            try:
                triton_client.end_tts_session(state.tts_session_id)
                logger.info(f"TTS session {state.tts_session_id} cleaned up on disconnect")
            except Exception as e:
                logger.warning(f"Error cleaning up TTS session: {e}")
        
        # MuseTalk is now stateless - no session cleanup needed
        
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


async def check_musetalk_available() -> bool:
    """Check if MuseTalk model is available on the server (stateless check)"""
    try:
        models_status = triton_client.check_models_ready()
        return models_status.get("musetalk", False)
    except Exception as e:
        logger.warning(f"Error checking MuseTalk availability: {e}")
        return False


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
        # End TTS session on stop in voice_to_voice mode
        if state.mode == "voice_to_voice" and state.tts_session_id is not None:
            await close_tts_session_async(state)
            logger.info("TTS session closed on stop_generation")

    elif msg_type == "recording_start":
        # Voice-to-voice: Initialize TTS when recording starts
        if state.mode == "voice_to_voice":
            logger.info("Recording started - initializing TTS session")
            await init_tts_cache_async(state)

    elif msg_type == "recording_stop":
        # Voice-to-voice: Clean up TTS session when recording stops (if not generating)
        if state.mode == "voice_to_voice" and not state.is_generating:
            if state.tts_session_id is not None:
                await close_tts_session_async(state)
                logger.info("TTS session closed on recording_stop")

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


async def audio_processor_loop(state: ConnectionState):
    """
    Dedicated audio processor that consumes from audio queue.
    
    This decouples the WebSocket receive loop from audio processing,
    preventing message backlog and disconnection issues.
    
    Audio is batched and processed together to reduce per-chunk overhead.
    """
    audio_buffer: List[np.ndarray] = []
    BATCH_SIZE = 5  # Process 5 chunks at once (~160ms at 32ms/chunk)
    BATCH_TIMEOUT = 0.05  # 50ms max wait for batch to fill
    
    logger.info(f"[AUDIO_PROC] Started for {state.connection_id}")
    
    try:
        while state.is_connected:
            # Collect a batch of audio chunks
            batch_start = time.time()
            
            while len(audio_buffer) < BATCH_SIZE:
                remaining_time = BATCH_TIMEOUT - (time.time() - batch_start)
                if remaining_time <= 0:
                    break
                    
                try:
                    audio_bytes = await asyncio.wait_for(
                        state.audio_queue.get(),
                        timeout=remaining_time
                    )
                    
                    # Decode audio (expecting float32 PCM at 16kHz)
                    try:
                        audio = np.frombuffer(audio_bytes, dtype=np.float32)
                        audio_buffer.append(audio)
                    except Exception as e:
                        logger.error(f"Failed to decode audio: {e}")
                        
                except asyncio.TimeoutError:
                    break
            
            # Process batch if we have any audio
            if audio_buffer and state.mode == "voice_to_voice":
                # Skip processing if currently generating (TTS/LLM running)
                if state.is_generating:
                    audio_buffer.clear()
                    continue
                
                # Concatenate all buffered audio
                combined_audio = np.concatenate(audio_buffer)
                audio_buffer.clear()
                
                # Update VAD time
                state.vad_time_ms += len(combined_audio) / 16000 * 1000
                
                if triton_client is not None:
                    try:
                        loop = asyncio.get_event_loop()
                        status, complete_audio = await loop.run_in_executor(
                            None,
                            triton_client.process_vad_with_state,
                            combined_audio,
                            state.vad_time_ms
                        )
                        
                        # Send VAD status (only for speech-related changes)
                        if status in ["speaking", "utterance_complete"]:
                            await send_message(state, "vad_status", {"status": status})
                        
                        if status == "utterance_complete" and complete_audio is not None:
                            # Process complete utterance
                            await process_voice_to_voice(state, complete_audio)
                            state.vad_time_ms = 0
                            triton_client.reset_vad_state()
                    
                    except Exception as e:
                        logger.error(f"VAD processing error: {e}")
            elif audio_buffer:
                # Clear buffer if not in voice_to_voice mode or generating
                audio_buffer.clear()
                
    except asyncio.CancelledError:
        logger.info(f"[AUDIO_PROC] Cancelled for {state.connection_id}")
    except Exception as e:
        logger.error(f"[AUDIO_PROC] Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"[AUDIO_PROC] Stopped for {state.connection_id}")


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
    
    TTS and MuseTalk sessions are initialized on-demand to avoid Triton reaping idle sessions.
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

    # Initialize TTS session on-demand (just-in-time)
    # Close any existing session first to ensure fresh state
    if state.tts_session_id is not None:
        logger.info(f"Closing existing TTS session {state.tts_session_id} before new generation")
        await close_tts_session_async(state)
    
    # Initialize fresh TTS session
    logger.info("Initializing TTS session on-demand for generation...")
    await init_tts_cache_async(state)
    
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
    
    logger.info(f"TTS session {state.tts_session_id} ready for generation")

    # MuseTalk is now stateless - just check if model is available
    musetalk_available = False
    if state.musetalk_enabled:
        musetalk_available = await check_musetalk_available()
        if musetalk_available:
            logger.info("MuseTalk model available (stateless mode)")
        else:
            logger.info("MuseTalk model not available, audio-only mode")
            state.musetalk_enabled = False

    # Get current buffer config for adaptive buffering
    buffer_config = triton_client.get_buffer_config() if triton_client else {}
    
    await send_message(state, "tts_start", {
        "text": "",
        "video_enabled": musetalk_available,
        "buffer_config": buffer_config,
    })

    # ---------- Blocking LLM worker (run in executor) ----------

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

    try:
        # Start LLM worker in background thread
        llm_future = loop.run_in_executor(None, run_llm_blocking)
        
        # Start shared TTS + MuseTalk worker
        sync_future, sync_task = await run_tts_with_musetalk(
            state, tts_input_queue, musetalk_available
        )

        # State for word-based TTS streaming
        tts_started = False
        words_sent_to_tts = 0
        logger.info(f"[LLM_TTS] Starting token processing. is_generating={state.is_generating}")

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
                    logger.info(f"[LLM_TTS] Starting TTS with first chunk: '{first_chunk}'")
                    tts_input_queue.put([first_chunk])
                    tts_started = True
                    words_sent_to_tts = 3

                if tts_started and num_words > words_sent_to_tts:
                    new_words = words[words_sent_to_tts:]
                    for w in new_words:
                        tts_input_queue.put([" " + w])
                    words_sent_to_tts = num_words

            elif msg_type == "done":
                logger.info(f"[LLM_TTS] LLM done. tts_started={tts_started}, words_sent={words_sent_to_tts}")
                break
            elif msg_type == "error":
                logger.error(f"LLM error: {data}")
                break

        await llm_future

        await send_message(state, "llm_complete", {"text": llm_response})
        logger.info(f"LLM complete: {llm_response[:50]}...")

        state.conversation.add_assistant_message(llm_response)

        # Send final empty strings to flush TTS lookahead buffer
        logger.info(f"[LLM_TTS] Flushing TTS and signaling end. tts_started={tts_started}")
        if tts_started:
            tts_input_queue.put([""])
            tts_input_queue.put([""])

        tts_input_queue.put(None)
        logger.info(f"[LLM_TTS] Sent None to TTS queue. Waiting for sync_future and sync_task...")

        await sync_future
        logger.info(f"[LLM_TTS] sync_future completed")
        await sync_task
        logger.info(f"[LLM_TTS] sync_task completed")

        # Voice-to-voice mode: End current session and reinitialize for next request
        # This keeps sessions fresh and avoids stale state
        if state.mode == "voice_to_voice":
            if state.tts_session_id is not None:
                await close_tts_session_async(state)
                logger.info("[LLM_TTS] TTS session closed after voice_to_voice completion")
            # Only reinitialize if connection is still active
            if state.is_connected:
                logger.info("[LLM_TTS] Reinitializing TTS session for next voice_to_voice request")
                await init_tts_cache_async(state)
            else:
                logger.info("[LLM_TTS] Connection closed, skipping TTS session reinitialization")
        else:
            # Chat-to-voice: Sessions initialized on-demand
            logger.info("[LLM_TTS] Generation complete. Sessions will be initialized on-demand for next request.")

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

    try:
        # Initialize TTS session on-demand (just-in-time)
        if state.tts_session_id is not None:
            logger.info(f"Closing existing TTS session {state.tts_session_id} before TTS-only generation")
            await close_tts_session_async(state)
        
        logger.info("Initializing TTS session on-demand for TTS-only generation...")
        await init_tts_cache_async(state)
        
        try:
            await asyncio.wait_for(state.tts_session_ready.wait(), timeout=35.0)
        except asyncio.TimeoutError:
            logger.error("TTS session not ready after timeout")
            await send_message(state, "error", {"message": "TTS session not ready"})
            return
        
        if state.tts_session_id is None:
            logger.error("TTS session ID is None after initialization")
            await send_message(state, "error", {"message": "TTS session not initialized"})
            return
        
        logger.info(f"TTS session {state.tts_session_id} ready for TTS-only generation")
        
        # MuseTalk is now stateless - just check if model is available
        musetalk_available = False
        if state.musetalk_enabled:
            musetalk_available = await check_musetalk_available()
            if musetalk_available:
                logger.info("MuseTalk model available (stateless mode) for TTS-only")
            else:
                logger.info("MuseTalk model not available, audio-only mode for TTS-only")
                state.musetalk_enabled = False
    
        # Get current buffer config for adaptive buffering
        buffer_config = triton_client.get_buffer_config() if triton_client else {}
        
        await send_message(state, "tts_start", {
            "text": text,
            "video_enabled": musetalk_available,
            "buffer_config": buffer_config,
        })
        
        # Create TTS input queue and pre-fill with text chunks
        tts_input_queue: Queue = Queue()
        text_chunks = _split_text_for_streaming(text)
        for chunk in text_chunks:
            tts_input_queue.put([chunk])
        tts_input_queue.put(None)  # Signal end
        
        # Run shared TTS + MuseTalk logic
        sync_future, sync_task = await run_tts_with_musetalk(
            state, tts_input_queue, musetalk_available
        )
        
        # Wait for completion
        await sync_future
        await sync_task
        
        # Don't pre-initialize sessions for next request
        # They will be initialized on-demand when next generation starts
        logger.info("[TTS_ONLY] Generation complete. Sessions will be initialized on-demand for next request.")
    
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
