"""
FastAPI Backend for Voice Assistant
WebSocket-based streaming for VAD, STT, LLM, TTS, and MuseTalk
"""

import asyncio
import json
import logging
import os
import time
import threading
import base64
import sys
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


async def run_tts_with_musetalk(
    state: "ConnectionState",
    tts_input_queue: Queue,
):
    """
    Shared TTS + MuseTalk synchronization logic.
    
    Consumes text chunks from tts_input_queue, generates audio via TTS,
    generates video frames via MuseTalk, and sends synced 
    audio+video frames to the client.
    MuseTalk runs in a stateless, chunked streaming mode with configurable
    start buffering and lookahead to avoid future padding artifacts.
    
    Args:
        state: The connection state
        tts_input_queue: Queue of text chunks (list of strings). Send None to terminate.
    """
    loop = asyncio.get_event_loop()
    
    # Constants for syncing
    SAMPLES_PER_FRAME = 960  # 40ms at 24kHz
    
    # Sync queue for combined audio+video
    sync_queue: asyncio.Queue = asyncio.Queue()
    
    def run_tts_and_musetalk_synced():
        """Run TTS and MuseTalk in parallel in a background thread."""
        all_audio = []  # TTS audio chunks (80ms each)
        all_words = []  # Word per audio chunk
        all_metrics = []  # Metrics per audio chunk
        word_sample_boundaries = []  # Cumulative samples per chunk
        chunks_received = 0
        audio_chunks_generated = 0
        total_samples = 0
        audio_lock = threading.Lock()
        audio_ready = threading.Condition(audio_lock)
        tts_done = threading.Event()
        error_event = threading.Event()

        # MuseTalk streaming controls (chunk-based)
        mt_start_after = 0
        mt_lookahead = 0
        if triton_client is not None:
            mt_start_after = max(0, int(getattr(triton_client.musetalk_params, "start_after_chunks", 0)))
            mt_lookahead = max(0, int(getattr(triton_client.musetalk_params, "lookahead_chunks", 0)))

        base_frame_index = state.musetalk_frame_index

        def report_error(msg: str) -> None:
            if error_event.is_set():
                return
            error_event.set()
            loop.call_soon_threadsafe(sync_queue.put_nowait, ("error", msg))

        def samples_before_chunk(boundaries: list[int], chunk_idx: int) -> int:
            if chunk_idx <= 0:
                return 0
            if chunk_idx - 1 >= len(boundaries):
                return boundaries[-1] if boundaries else 0
            return boundaries[chunk_idx - 1]

        def enqueue_synced_frame(
            frame_bytes: bytes,
            frame_idx: int,
            timestamp_ms: float,
            segment_audio: np.ndarray,
            segment_start_samples: int,
            frame_audio_start: int,
            allow_partial_last: bool,
            segment_start_ms: float,
            word_cursor: dict,
            last_emitted_word_idx: dict,
            boundaries_snapshot: list[int],
            words_snapshot: list[str],
            metrics_snapshot: list[TTSMetrics],
            frames_sent_ref: dict,
        ) -> None:
            expected_idx = base_frame_index + frames_sent_ref["value"]
            if frame_idx < expected_idx:
                return
            if frame_idx > expected_idx:
                logger.warning(
                    "[MUSETALK_WORKER] Frame index gap: expected=%s got=%s",
                    expected_idx,
                    frame_idx,
                )
                frames_sent_ref["value"] = frame_idx - base_frame_index

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

            while word_cursor["value"] < len(boundaries_snapshot) and global_audio_start >= boundaries_snapshot[word_cursor["value"]]:
                word_cursor["value"] += 1

            word = ""
            metrics_data = {}
            current_word_idx = word_cursor["value"]
            if current_word_idx < len(words_snapshot) and current_word_idx != last_emitted_word_idx["value"]:
                word = words_snapshot[current_word_idx]
                last_emitted_word_idx["value"] = current_word_idx
                if current_word_idx < len(metrics_snapshot):
                    metrics_data = {
                        "rtf": round(metrics_snapshot[current_word_idx].rtf, 3),
                        "generation_time_ms": round(metrics_snapshot[current_word_idx].generation_time_ms, 1),
                        "audio_duration_ms": round(metrics_snapshot[current_word_idx].audio_duration_ms, 1),
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
            frames_sent_ref["value"] = frame_idx - base_frame_index + 1

        def tts_worker() -> None:
            nonlocal chunks_received, audio_chunks_generated, total_samples
            try:
                logger.info(f"[TTS_WORKER] Started, session_id={state.tts_session_id}")
                while True:
                    item = tts_input_queue.get()
                    if item is None:
                        logger.info(
                            "[TTS_WORKER] Received None, TTS generation complete. chunks_received=%s, audio_generated=%s",
                            chunks_received,
                            audio_chunks_generated,
                        )
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
                            logger.info(
                                "[TTS_WORKER] Audio chunk #%s, samples=%s, word=%s",
                                audio_chunks_generated,
                                len(audio),
                                word,
                            )

                        if not state.is_generating:
                            logger.info("[TTS_WORKER] state.is_generating=False, breaking")
                            break

                        with audio_ready:
                            all_audio.append(audio)
                            all_words.append(word)
                            all_metrics.append(metrics)
                            total_samples += len(audio)
                            word_sample_boundaries.append(total_samples)
                            audio_ready.notify_all()
            except Exception as e:
                logger.error(f"[TTS_WORKER] Error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                report_error(str(e))
            finally:
                tts_done.set()
                with audio_ready:
                    audio_ready.notify_all()

        def musetalk_worker() -> None:
            frames_sent_ref = {"value": 0}
            processed_chunk_index = 0
            word_cursor = {"value": 0}
            last_emitted_word_idx = {"value": -1}

            logger.info(
                "[MUSETALK_WORKER] Started. start_after_chunks=%s, lookahead_chunks=%s",
                mt_start_after,
                mt_lookahead,
            )
            try:
                while state.is_generating and not error_event.is_set():
                    done = False
                    segment_info = None
                    with audio_ready:
                        while True:
                            total_chunks = len(all_audio)
                            flush = tts_done.is_set()
                            if total_chunks == 0 and not flush:
                                audio_ready.wait(timeout=0.1)
                                if not state.is_generating or error_event.is_set():
                                    break
                                continue
                            if not flush and total_chunks < mt_start_after:
                                audio_ready.wait(timeout=0.1)
                                if not state.is_generating or error_event.is_set():
                                    break
                                continue

                            output_end_chunk = total_chunks if flush else max(0, total_chunks - mt_lookahead)
                            if output_end_chunk <= processed_chunk_index:
                                if flush:
                                    done = True
                                    break
                                audio_ready.wait(timeout=0.1)
                                if not state.is_generating or error_event.is_set():
                                    break
                                continue

                            segment_start_chunk = max(0, processed_chunk_index - (1 if processed_chunk_index > 0 else 0))
                            segment_end_chunk = total_chunks
                            segment_info = (
                                list(all_audio[segment_start_chunk:segment_end_chunk]),
                                list(word_sample_boundaries),
                                list(all_words),
                                list(all_metrics),
                                output_end_chunk,
                                segment_start_chunk,
                                flush,
                            )
                            break

                    if not state.is_generating or error_event.is_set():
                        break

                    if done:
                        break

                    if segment_info is None:
                        continue

                    (
                        audio_slice,
                        boundaries_snapshot,
                        words_snapshot,
                        metrics_snapshot,
                        output_end_chunk,
                        segment_start_chunk,
                        flush,
                    ) = segment_info

                    if not audio_slice:
                        if flush:
                            break
                        continue

                    output_end_chunk = min(output_end_chunk, len(boundaries_snapshot))
                    if output_end_chunk <= processed_chunk_index:
                        if flush:
                            break
                        continue

                    segment_audio = np.concatenate(audio_slice)
                    if segment_audio.size == 0:
                        processed_chunk_index = output_end_chunk
                        if flush:
                            break
                        continue

                    segment_start_samples = samples_before_chunk(boundaries_snapshot, segment_start_chunk)
                    overlap_samples = 0
                    if processed_chunk_index > segment_start_chunk:
                        overlap_samples = samples_before_chunk(boundaries_snapshot, processed_chunk_index) - segment_start_samples
                    output_end_samples = samples_before_chunk(boundaries_snapshot, output_end_chunk) - segment_start_samples

                    segment_start_frame_index = base_frame_index + int(segment_start_samples // SAMPLES_PER_FRAME)
                    segment_start_ms = segment_start_samples / 24.0
                    allow_partial_last = flush

                    for frame_bytes, frame_idx, timestamp_ms, _ in triton_client.generate_musetalk_frames(
                        segment_audio,
                        frame_index=segment_start_frame_index,
                    ):
                        if not state.is_generating or error_event.is_set():
                            break

                        local_frame_idx = frame_idx - segment_start_frame_index
                        frame_audio_start = local_frame_idx * SAMPLES_PER_FRAME
                        if frame_audio_start < overlap_samples:
                            continue
                        if not allow_partial_last and (frame_audio_start + SAMPLES_PER_FRAME) > output_end_samples:
                            continue
                        if allow_partial_last and frame_audio_start >= output_end_samples:
                            continue

                        enqueue_synced_frame(
                            frame_bytes,
                            frame_idx,
                            timestamp_ms,
                            segment_audio,
                            segment_start_samples,
                            frame_audio_start,
                            allow_partial_last=allow_partial_last,
                            segment_start_ms=segment_start_ms,
                            word_cursor=word_cursor,
                            last_emitted_word_idx=last_emitted_word_idx,
                            boundaries_snapshot=boundaries_snapshot,
                            words_snapshot=words_snapshot,
                            metrics_snapshot=metrics_snapshot,
                            frames_sent_ref=frames_sent_ref,
                        )

                    processed_chunk_index = output_end_chunk
                    state.musetalk_frame_index = base_frame_index + frames_sent_ref["value"]

                    if allow_partial_last and output_end_chunk <= processed_chunk_index:
                        break
            except Exception as e:
                logger.error(f"[MUSETALK_WORKER] Error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                report_error(str(e))
            finally:
                logger.info("[MUSETALK_WORKER] Finished")

        tts_thread = threading.Thread(target=tts_worker, name="tts-worker", daemon=True)
        tts_thread.start()

        musetalk_thread = threading.Thread(target=musetalk_worker, name="musetalk-worker", daemon=True)
        musetalk_thread.start()

        tts_thread.join()
        musetalk_thread.join()

        logger.info(
            "[TTS_WORKER] Finished. Total chunks_received=%s, audio_chunks_generated=%s",
            chunks_received,
            audio_chunks_generated,
        )
        if not error_event.is_set():
            loop.call_soon_threadsafe(sync_queue.put_nowait, ("done", None))
    
    async def sync_consumer():
        """Consume synced audio+video and forward to client."""
        messages_sent = 0
        synced_frames = 0
        try:
            logger.info("[SYNC_CONSUMER] Started")
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
                elif msg_type == "done":
                    logger.info(f"[SYNC_CONSUMER] Got 'done', stopping")
                    break
                elif msg_type == "error":
                    logger.error(f"[SYNC_CONSUMER] Error: {data}")
                    break
        finally:
            logger.info(f"[SYNC_CONSUMER] Finished. synced_frames={synced_frames}, total_sent={messages_sent}")
            # Only send completion messages if connection is still active
            if state.is_connected:
                await send_message(state, "tts_complete", {})
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
        self.tts_session_id: Optional[int] = None
        self.tts_session_ready = asyncio.Event()  # Set when TTS session is ready
        self.is_generating = False
        self.is_connected = True  # Track connection state
        self.vad_time_ms = 0
        
        # Audio processing queue (decoupled from WS receive loop)
        self.audio_queue: asyncio.Queue = asyncio.Queue(maxsize=100)  # ~3.2s buffer at 32ms chunks
        self.audio_processor_task: Optional[asyncio.Task] = None
        
        # MuseTalk state (stateless - only track frame index for video continuity)
        self.musetalk_frame_index: int = 0  # Track frame index for avatar animation continuity


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
    
    if msg_type == "stop_generation":
        state.is_generating = False
        if state.tts_session_id is not None:
            await close_tts_session_async(state)
            logger.info("TTS session closed on stop_generation")

    elif msg_type == "recording_start":
        logger.info("Recording started - initializing TTS session")
        await init_tts_cache_async(state)

    elif msg_type == "recording_stop":
        # Clean up TTS session when recording stops (if not generating)
        if not state.is_generating:
            if state.tts_session_id is not None:
                await close_tts_session_async(state)
                logger.info("TTS session closed on recording_stop")


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
            if audio_buffer:
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

    musetalk_available = await check_musetalk_available()
    if musetalk_available:
        logger.info("MuseTalk model available (stateless mode)")
    else:
        logger.error("MuseTalk model not available; aborting generation")
        await send_message(state, "error", {"message": "MuseTalk model not available"})
        if state.tts_session_id is not None:
            await close_tts_session_async(state)
        return

    # Get current buffer config for adaptive buffering
    buffer_config = triton_client.get_buffer_config() if triton_client else {}
    
    await send_message(state, "tts_start", {
        "text": "",
        "video_enabled": True,
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
            state, tts_input_queue
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

        if state.tts_session_id is not None:
            await close_tts_session_async(state)
            logger.info("[LLM_TTS] TTS session closed after completion")
        # Only reinitialize if connection is still active
        if state.is_connected:
            logger.info("[LLM_TTS] Reinitializing TTS session for next request")
            await init_tts_cache_async(state)
        else:
            logger.info("[LLM_TTS] Connection closed, skipping TTS session reinitialization")

    except Exception as e:
        logger.error(f"LLM/TTS/MuseTalk error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await send_message(state, "error", {"message": str(e)})




if __name__ == "__main__":
    import uvicorn
    import os
    os.environ['TRITON_URL'] = '185.151.171.35:51954'
    uvicorn.run(app, host="0.0.0.0", port=8080)
