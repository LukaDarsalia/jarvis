"""
Voice Assistant FastAPI Application.

A WebSocket-based voice assistant with:
- Voice Activity Detection (VAD)
- Speech-to-Text (STT)
- Large Language Model (LLM) streaming
- Text-to-Speech (TTS) with 2-word lookahead
- MuseTalk avatar video generation
- Synchronized audio/video streaming

Architecture:
    User (WebSocket) <-> FastAPI <-> Triton (gRPC)
                                       |
                                       v
                            VAD, STT, LLM, TTS, MuseTalk
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import load_config, AppConfig
from triton_services import TritonClient
from tts_service import TTSService
from pipeline import VoiceToVoicePipeline, StreamingMetricsManager, PipelineConfig
from websocket_handler import WebSocketHandler

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Uncaught exception!", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = global_exception_handler


# ============================================================================
# Global State
# ============================================================================

# Loaded during lifespan
config: Optional[AppConfig] = None
triton_client: Optional[TritonClient] = None
tts_service: Optional[TTSService] = None
pipeline: Optional[VoiceToVoicePipeline] = None
ws_handler: Optional[WebSocketHandler] = None


# ============================================================================
# Application Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global config, triton_client, tts_service, pipeline, ws_handler

    # Load configuration
    config = load_config()
    logger.info(f"Initializing with Triton URL: {config.triton_url}")

    # Create Triton client
    triton_client = TritonClient(
        triton_url=config.triton_url,
        vad_config=config.vad,
        llm_config=config.llm,
        tts_config=config.tts,
        musetalk_config=config.musetalk,
    )

    # Create TTS service
    tts_service = TTSService(
        triton_url=config.triton_url,
        config=config.tts,
    )

    # Create metrics manager
    metrics_manager = StreamingMetricsManager(config.streaming)

    # Create pipeline
    pipeline_config = PipelineConfig(
        tts_config=config.tts,
        musetalk_config=config.musetalk,
        streaming_config=config.streaming,
    )
    pipeline = VoiceToVoicePipeline(
        tts_service=tts_service,
        musetalk_service=triton_client.musetalk,
        metrics_manager=metrics_manager,
        config=pipeline_config,
    )

    # Create WebSocket handler
    ws_handler = WebSocketHandler(
        triton_client=triton_client,
        tts_service=tts_service,
        pipeline=pipeline,
        config=config,
    )

    # Health check
    if triton_client.is_healthy():
        logger.info("Triton server is healthy")
        logger.info(f"Models status: {triton_client.check_models_ready()}")
    else:
        logger.warning("Triton server is not available - client will retry on requests")

    yield

    logger.info("Shutting down...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Voice Assistant API",
    description="WebSocket-based voice assistant with VAD, STT, LLM, TTS, and avatar generation",
    version="2.0.0",
    lifespan=lifespan,
)

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


# ============================================================================
# Routes
# ============================================================================

@app.get("/")
async def root():
    """Serve the main page."""
    index_path = static_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Voice Assistant API - WebSocket endpoint at /ws"}


@app.get("/health")
async def health():
    """Check server and model health."""
    if triton_client is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Client not initialized"},
        )

    server_healthy = triton_client.is_healthy()
    models_status = triton_client.check_models_ready()

    return {
        "status": "healthy" if server_healthy else "degraded",
        "triton_server": server_healthy,
        "models": models_status,
    }


@app.get("/config")
async def get_config():
    """Get current configuration."""
    if config is None or triton_client is None:
        raise HTTPException(status_code=503, detail="Client not initialized")

    buffer_config = pipeline.get_buffer_config() if pipeline else {}

    return {
        "vad": {
            "speech_threshold_ms": config.vad.speech_threshold_ms,
            "silence_threshold_ms": config.vad.silence_threshold_ms,
            "prob_threshold": config.vad.prob_threshold,
        },
        "llm": {
            "max_new_tokens": config.llm.max_new_tokens,
            "temperature": config.llm.temperature,
            "top_p": config.llm.top_p,
            "system_prompt": config.llm.system_prompt,
        },
        "tts": {
            "backbone_temperature": config.tts.backbone_temperature,
            "backbone_top_p": config.tts.backbone_top_p,
            "depth_temperature": config.tts.depth_temperature,
            "depth_top_p": config.tts.depth_top_p,
            "sample_rate": config.tts.sample_rate,
        },
        "buffer": buffer_config.to_dict() if hasattr(buffer_config, 'to_dict') else buffer_config,
        "musetalk": {
            "start_after_chunks": config.musetalk.start_after_chunks,
            "lookahead_chunks": config.musetalk.lookahead_chunks,
        },
    }


@app.post("/config")
async def update_config(new_config: dict):
    """Update configuration."""
    if config is None or triton_client is None:
        raise HTTPException(status_code=503, detail="Client not initialized")

    if "vad" in new_config:
        vad = new_config["vad"]
        if "speech_threshold_ms" in vad:
            config.vad.speech_threshold_ms = float(vad["speech_threshold_ms"])
        if "silence_threshold_ms" in vad:
            config.vad.silence_threshold_ms = float(vad["silence_threshold_ms"])
        if "prob_threshold" in vad:
            config.vad.prob_threshold = float(vad["prob_threshold"])

    if "llm" in new_config:
        llm = new_config["llm"]
        if "max_new_tokens" in llm:
            config.llm.max_new_tokens = int(llm["max_new_tokens"])
        if "temperature" in llm:
            config.llm.temperature = float(llm["temperature"])
        if "top_p" in llm:
            config.llm.top_p = float(llm["top_p"])
        if "system_prompt" in llm:
            config.llm.system_prompt = str(llm["system_prompt"])

    if "tts" in new_config:
        tts = new_config["tts"]
        if "backbone_temperature" in tts:
            config.tts.backbone_temperature = float(tts["backbone_temperature"])
        if "backbone_top_p" in tts:
            config.tts.backbone_top_p = float(tts["backbone_top_p"])
        if "depth_temperature" in tts:
            config.tts.depth_temperature = float(tts["depth_temperature"])
        if "depth_top_p" in tts:
            config.tts.depth_top_p = float(tts["depth_top_p"])
        if "sample_rate" in tts:
            config.tts.sample_rate = int(tts["sample_rate"])

    if "buffer" in new_config:
        buffer_cfg = new_config["buffer"]
        manual_buffer = buffer_cfg.get("manual_buffer_ms", None)
        if manual_buffer in ["", None]:
            pipeline.set_manual_buffer(None)
        else:
            try:
                pipeline.set_manual_buffer(float(manual_buffer))
            except (TypeError, ValueError):
                logger.warning(f"Ignoring invalid manual buffer value: {manual_buffer}")

    if "musetalk" in new_config:
        mt_cfg = new_config["musetalk"]
        if "start_after_chunks" in mt_cfg:
            try:
                config.musetalk.start_after_chunks = max(0, int(mt_cfg["start_after_chunks"]))
            except (TypeError, ValueError):
                logger.warning(f"Ignoring invalid start_after_chunks: {mt_cfg['start_after_chunks']}")
        if "lookahead_chunks" in mt_cfg:
            try:
                config.musetalk.lookahead_chunks = max(0, int(mt_cfg["lookahead_chunks"]))
            except (TypeError, ValueError):
                logger.warning(f"Ignoring invalid lookahead_chunks: {mt_cfg['lookahead_chunks']}")

    return await get_config()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for voice assistant."""
    if ws_handler is None:
        await websocket.close(code=1011, reason="Server not ready")
        return

    try:
        await ws_handler.handle_connection(websocket)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as exc:
        logger.error(f"WebSocket error: {exc}")
        import traceback
        logger.error(traceback.format_exc())


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    os.environ.setdefault("TRITON_URL", "185.151.171.35:44957")
    uvicorn.run(app, host="0.0.0.0", port=8080)
