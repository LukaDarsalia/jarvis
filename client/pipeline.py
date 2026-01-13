"""
AV Pipeline Orchestrator for TTS + MuseTalk streaming.

Simplified architecture:
    LLM Stream -> TTS Worker -> Audio Frames
                                    |
                             MuseTalk Worker -> Video Frames
                                    |
                              AV Sender -> WebSocket (immediate, no timing)

Client-side handles all buffering and playback timing.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Optional, Callable, Dict, List

import numpy as np

from config import TTSConfig, MuseTalkConfig, StreamingConfig
from models import AudioFrame, AVFrame, TTSMetrics, StreamingStats
from streaming_chunker import StreamingTTSChunker
from tts_service import TTSService
from triton_services import MuseTalkService
from text_utils.numbers_to_text import NumberConverter
from text_utils.streaming_text_processor import StreamingTextProcessor

logger = logging.getLogger(__name__)


# ============================================================================
# Streaming Metrics Manager (simple stats tracking)
# ============================================================================

class StreamingMetricsManager:
    """Tracks generation timing statistics."""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self._tts_rtf = StreamingStats(window_size=config.buffer_window_size)
        self._musetalk_fps = StreamingStats(window_size=config.buffer_window_size)
        self._lock = threading.Lock()

    def record_tts_generation(self, audio_duration_ms: float, generation_time_ms: float) -> None:
        """Record TTS generation timing."""
        if generation_time_ms > 0 and audio_duration_ms > 0:
            rtf = generation_time_ms / audio_duration_ms
            with self._lock:
                self._tts_rtf.add_sample(rtf)

    def record_musetalk_generation(self, frames: int, duration_sec: float) -> None:
        """Record MuseTalk frame generation timing."""
        if duration_sec > 0 and frames > 0:
            fps = frames / duration_sec
            with self._lock:
                self._musetalk_fps.add_sample(fps)

    def get_stats(self) -> dict:
        """Get current generation stats."""
        with self._lock:
            return {
                "tts_rtf_mean": self._tts_rtf.mean,
                "tts_rtf_std": self._tts_rtf.std,
                "musetalk_fps_mean": self._musetalk_fps.mean,
                "musetalk_fps_std": self._musetalk_fps.std,
            }


# ============================================================================
# AV Pipeline
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the AV pipeline."""
    tts_config: TTSConfig
    musetalk_config: MuseTalkConfig
    streaming_config: StreamingConfig


@dataclass
class AVPipelineResult:
    """Result from the AV pipeline."""
    success: bool
    error_message: Optional[str] = None
    total_frames: int = 0
    total_audio_ms: float = 0.0


class AVPipeline:
    """
    Orchestrates TTS and MuseTalk streaming for synchronized AV output.

    Simplified design:
    - TTS Worker: generates audio frames
    - MuseTalk Worker: generates video frames (batched for efficiency)
    - AV Sender: pairs audio+video and sends immediately (no timing delays)

    Client handles all buffering and playback timing.
    """

    def __init__(
        self,
        tts_service: TTSService,
        musetalk_service: MuseTalkService,
        metrics_manager: StreamingMetricsManager,
        config: PipelineConfig,
    ):
        self.tts_service = tts_service
        self.musetalk_service = musetalk_service
        self.metrics_manager = metrics_manager
        self.config = config

        # Sample rate and frame calculations
        self._sample_rate = config.tts_config.sample_rate
        self._fps = config.musetalk_config.fps
        self._samples_per_frame = self._sample_rate // self._fps  # 960 samples @ 25fps

    async def run(
        self,
        text_input_queue: Queue,
        session_id: int,
        video_enabled: bool,
        base_frame_index: int,
        is_generating: Callable[[], bool],
        on_frame: Callable[[AVFrame], None],
        on_error: Callable[[str], None],
        on_complete: Callable[[], None],
    ) -> AVPipelineResult:
        """
        Run the AV pipeline.

        Args:
            text_input_queue: Queue of text chunks to synthesize. None signals end.
            session_id: TTS session ID (must be initialized)
            video_enabled: Whether to generate video frames
            base_frame_index: Starting frame index for avatar cycle
            is_generating: Callback to check if generation should continue
            on_frame: Callback for each AV frame (sent immediately)
            on_error: Callback for errors
            on_complete: Callback when pipeline completes

        Returns:
            AVPipelineResult with statistics
        """
        loop = asyncio.get_event_loop()
        out_queue: asyncio.Queue = asyncio.Queue()

        result = AVPipelineResult(success=True)

        def runner():
            """Main pipeline runner (runs in thread)."""
            nonlocal result

            # Internal queues
            audio_frame_queue: Queue = Queue()  # Audio frames for sending
            musetalk_queue: Queue = Queue()      # Audio for MuseTalk processing

            # Video frame cache: frame_index -> jpeg_bytes
            video_cache: Dict[int, bytes] = {}
            video_lock = threading.Lock()

            # State
            frame_index = base_frame_index
            total_audio_samples = 0

            def report_error(msg: str) -> None:
                loop.call_soon_threadsafe(out_queue.put_nowait, ("error", msg))

            def report_frame(av_frame: AVFrame) -> None:
                loop.call_soon_threadsafe(out_queue.put_nowait, ("frame", av_frame))

            def report_done() -> None:
                loop.call_soon_threadsafe(out_queue.put_nowait, ("done", None))

            # ----------------------------------------------------------------
            # TTS Worker - generates audio frames
            # ----------------------------------------------------------------
            def tts_worker() -> None:
                nonlocal frame_index, result, total_audio_samples

                audio_buffer = np.zeros((0,), dtype=np.float32)
                local_frame_index = base_frame_index
                audio_log_count = 0
                expected_chunk = self.config.tts_config.chunk_samples

                try:
                    while is_generating():
                        try:
                            item = text_input_queue.get(timeout=0.1)
                        except Empty:
                            continue

                        if item is None:
                            break

                        # Generate audio from text chunks
                        for audio, word, metrics in self.tts_service.generate_stream(
                            item,
                            session_id=session_id,
                        ):
                            if not is_generating():
                                break

                            # Normalize audio
                            audio_np = np.asarray(audio)
                            audio_log_count += 1
                            if np.issubdtype(audio_np.dtype, np.integer):
                                max_val = float(np.iinfo(audio_np.dtype).max)
                                audio_np = audio_np.astype(np.float32) / max_val
                            else:
                                audio_np = audio_np.astype(np.float32)
                            audio_np = np.clip(audio_np, -1.0, 1.0).reshape(-1)

                            if audio_np.size == 0:
                                continue

                            log_audio = (
                                audio_log_count <= 3
                                or audio_log_count % 50 == 0
                                or (expected_chunk > 0 and audio_np.size != expected_chunk)
                            )
                            if log_audio:
                                min_val = float(np.min(audio_np))
                                max_val = float(np.max(audio_np))
                                rms_val = float(np.sqrt(np.mean(audio_np ** 2))) if audio_np.size > 0 else 0.0
                                nan_count = int(np.isnan(audio_np).sum())
                                logger.info(
                                    f"TTS audio chunk {audio_log_count}: samples={audio_np.size} "
                                    f"expected={expected_chunk} dtype={audio_np.dtype} "
                                    f"min={min_val:.4f} max={max_val:.4f} rms={rms_val:.4f} nan={nan_count}"
                                )

                            audio_buffer = np.concatenate([audio_buffer, audio_np])

                            # Record metrics
                            if metrics and metrics.audio_duration_ms > 0:
                                self.metrics_manager.record_tts_generation(
                                    metrics.audio_duration_ms,
                                    metrics.generation_time_ms,
                                )

                            # Emit complete frames (960 samples = 40ms each)
                            while audio_buffer.size >= self._samples_per_frame:
                                frame_samples = audio_buffer[:self._samples_per_frame]
                                audio_buffer = audio_buffer[self._samples_per_frame:]

                                frame = AudioFrame(
                                    index=local_frame_index,
                                    samples=frame_samples,
                                    word=word if local_frame_index == base_frame_index or word else "",
                                    metrics=metrics.copy() if metrics else None,
                                    timestamp_ms=(local_frame_index - base_frame_index) * 40.0,
                                )
                                word = ""  # Only include word on first frame

                                audio_frame_queue.put(frame)
                                musetalk_queue.put(frame)
                                local_frame_index += 1
                                total_audio_samples += len(frame_samples)

                except Exception as exc:
                    logger.error(f"TTS worker error: {exc}")
                    import traceback
                    logger.error(traceback.format_exc())
                    report_error(f"TTS worker error: {exc}")

                finally:
                    # Flush remaining audio
                    if audio_buffer.size > 0:
                        frame = AudioFrame(
                            index=local_frame_index,
                            samples=audio_buffer,
                            word="",
                            metrics=None,
                            timestamp_ms=(local_frame_index - base_frame_index) * 40.0,
                        )
                        audio_frame_queue.put(frame)
                        musetalk_queue.put(frame)
                        local_frame_index += 1
                        total_audio_samples += len(audio_buffer)

                    # Signal completion
                    audio_frame_queue.put(None)
                    musetalk_queue.put(None)

                    result.total_frames = local_frame_index - base_frame_index
                    frame_index = local_frame_index

            # ----------------------------------------------------------------
            # MuseTalk Worker - generates video frames in batches
            # ----------------------------------------------------------------
            def musetalk_worker() -> None:
                if not video_enabled:
                    # Just drain the queue
                    while True:
                        try:
                            frame = musetalk_queue.get(timeout=0.1)
                            if frame is None:
                                break
                        except Empty:
                            if not is_generating():
                                break
                    return

                # Batch configuration
                # batch_size: number of video frames per Triton call
                # lookahead_chunks: TTS chunks (80ms each) for Whisper future context
                batch_size = self.config.musetalk_config.batch_size
                lookahead_chunks = self.config.musetalk_config.lookahead_chunks
                
                # Constants
                WHISPER_WINDOW_MS = 200  # Whisper needs 200ms (10 chunks × 20ms) for features
                FRAME_MS = 40  # Each video frame = 40ms @ 25fps
                TTS_CHUNK_MS = 80  # Each TTS output chunk = 80ms
                
                # Calculate audio requirements in samples
                whisper_window_samples = int(WHISPER_WINDOW_MS / 1000 * self._sample_rate)
                lookahead_samples = int(lookahead_chunks * TTS_CHUNK_MS / 1000 * self._sample_rate)
                batch_audio_samples = batch_size * self._samples_per_frame  # batch_size × 40ms
                
                # First batch needs: 200ms + (batch_size-1)*40ms + lookahead*80ms
                # This is because first frame needs 200ms whisper window, subsequent frames add 40ms each
                first_batch_min_samples = whisper_window_samples + (batch_size - 1) * self._samples_per_frame + lookahead_samples
                
                # Subsequent batches need: batch_size * 40ms (lookahead from previous batch provides context)
                subsequent_batch_min_samples = batch_audio_samples
                
                # Max batch size to prevent memory issues
                max_audio_samples = int(2.0 * self._sample_rate)  # 2 seconds max
                
                is_first_batch = True
                
                logger.info(
                    f"MuseTalk config: batch_size={batch_size}, lookahead_chunks={lookahead_chunks} "
                    f"(first_batch_min={first_batch_min_samples / self._sample_rate * 1000:.0f}ms, "
                    f"subsequent_batch_min={subsequent_batch_min_samples / self._sample_rate * 1000:.0f}ms)"
                )
                
                # Lookahead buffer: audio from end of previous batch for Whisper context
                lookahead_buffer = np.zeros((0,), dtype=np.float32)

                audio_buffer = np.zeros((0,), dtype=np.float32)
                frame_buffer: List[AudioFrame] = []
                next_frame_index = base_frame_index
                done = False
                
                # Timing metrics for debugging
                last_batch_end_time: Optional[float] = None
                total_wait_time_ms = 0.0
                total_process_time_ms = 0.0
                total_frames_generated = 0
                batch_count = 0

                try:
                    while True:
                        # Check exit conditions first
                        if done and len(audio_buffer) == 0:
                            break
                        if not is_generating() and done:
                            # Generation stopped and we've processed everything
                            if len(audio_buffer) == 0:
                                break
                        
                        # Collect frames from queue
                        try:
                            frame = musetalk_queue.get(timeout=0.1)
                            if frame is None:
                                done = True
                            else:
                                frame_buffer.append(frame)
                                audio_buffer = np.concatenate([audio_buffer, frame.samples])
                        except Empty:
                            if not is_generating() and musetalk_queue.empty():
                                done = True
                            # If done with no audio, exit
                            if done and len(audio_buffer) == 0:
                                break

                        # Decide whether to process
                        should_process = False
                        current_min_samples = first_batch_min_samples if is_first_batch else subsequent_batch_min_samples
                        
                        if done and len(audio_buffer) > 0:
                            # For tail chunks, process whatever remains if it's enough for at least 1 frame
                            # Need at least 200ms (whisper window) for meaningful processing
                            min_tail_samples = whisper_window_samples
                            if len(audio_buffer) < min_tail_samples:
                                # Not enough audio for meaningful processing, discard and exit
                                logger.info(f"MuseTalk: Discarding {len(audio_buffer)} samples (< {min_tail_samples} min)")
                                break
                            should_process = True
                        elif len(audio_buffer) >= max_audio_samples:
                            should_process = True
                        elif len(audio_buffer) >= current_min_samples:
                            should_process = True

                        if not should_process:
                            continue

                        if len(audio_buffer) == 0:
                            continue

                        # Track wait time (time since last batch finished)
                        batch_start_time = time.time()
                        if last_batch_end_time is not None:
                            wait_ms = (batch_start_time - last_batch_end_time) * 1000.0
                            total_wait_time_ms += wait_ms
                        else:
                            wait_ms = 0.0
                        
                        # Prepend lookahead buffer from previous batch for context overlap
                        # This gives Whisper real audio context instead of zero padding at batch boundaries
                        if lookahead_buffer.size > 0:
                            full_audio = np.concatenate([lookahead_buffer, audio_buffer])
                            # Calculate how many frames the lookahead covers (for frame index offset)
                            lookahead_frames = lookahead_buffer.size // self._samples_per_frame
                        else:
                            full_audio = audio_buffer.copy()
                            lookahead_frames = 0
                        
                        process_audio = full_audio
                        # Frame index for Triton starts at the lookahead portion
                        # But we only keep frames AFTER the lookahead (those are new)
                        process_start_index = next_frame_index - lookahead_frames
                        
                        # Save end of current audio as lookahead for next batch
                        if not done and lookahead_samples > 0 and audio_buffer.size >= lookahead_samples:
                            lookahead_buffer = audio_buffer[-lookahead_samples:]
                        else:
                            lookahead_buffer = np.zeros((0,), dtype=np.float32)

                        # Clear buffers
                        audio_buffer = np.zeros((0,), dtype=np.float32)
                        frame_buffer = []

                        # Generate video frames
                        triton_start_time = time.time()
                        frames_generated = 0
                        new_frames_count = 0  # Frames after lookahead
                        audio_duration_ms = len(process_audio) / self._sample_rate * 1000.0

                        logger.info(
                            f"MuseTalk: Processing {len(process_audio)/self._sample_rate:.3f}s audio "
                            f"(lookahead={lookahead_frames} frames) starting at index {process_start_index} "
                            f"(waited {wait_ms:.0f}ms for audio, first_batch={is_first_batch})"
                        )

                        for frame_bytes, frame_idx, _ in self.musetalk_service.generate_frames(
                            process_audio,
                            frame_index=process_start_index,
                        ):
                            frames_generated += 1
                            # Only cache frames that are NEW (after lookahead region)
                            if frame_idx >= next_frame_index:
                                with video_lock:
                                    video_cache[frame_idx] = frame_bytes
                                new_frames_count += 1

                        # Update next frame index (only count new frames, not re-generated lookahead)
                        next_frame_index = next_frame_index + new_frames_count
                        last_batch_end_time = time.time()
                        batch_count += 1
                        is_first_batch = False  # First batch has been processed

                        # Record metrics (use new_frames_count for accurate stats)
                        if new_frames_count > 0:
                            triton_duration_ms = (last_batch_end_time - triton_start_time) * 1000.0
                            total_process_time_ms += triton_duration_ms
                            total_frames_generated += new_frames_count
                            
                            if triton_duration_ms > 0:
                                self.metrics_manager.record_musetalk_generation(
                                    new_frames_count, triton_duration_ms / 1000.0
                                )
                            
                            # Calculate effective FPS for this batch (new frames only)
                            batch_eff_fps = (new_frames_count / triton_duration_ms * 1000.0) if triton_duration_ms > 0 else 0.0
                            # Calculate sustained FPS including wait time
                            cycle_time_ms = wait_ms + triton_duration_ms
                            sustained_fps = (new_frames_count / cycle_time_ms * 1000.0) if cycle_time_ms > 0 else 0.0
                            
                            logger.info(
                                f"MuseTalk: batch={batch_count} | new_frames={new_frames_count} | "
                                f"total_generated={frames_generated} (lookahead={lookahead_frames}) | "
                                f"triton_ms={triton_duration_ms:.0f} | wait_ms={wait_ms:.0f} | "
                                f"batch_fps={batch_eff_fps:.1f} | sustained_fps={sustained_fps:.1f} | "
                                f"audio_ms={audio_duration_ms:.0f}"
                            )

                        if done:
                            break

                except Exception as exc:
                    logger.error(f"MuseTalk worker error: {exc}")
                    import traceback
                    logger.error(traceback.format_exc())
                    report_error(f"MuseTalk worker error: {exc}")
                
                finally:
                    # Log final summary
                    if total_frames_generated > 0 and batch_count > 0:
                        total_time_ms = total_wait_time_ms + total_process_time_ms
                        overall_sustained_fps = (total_frames_generated / total_time_ms * 1000.0) if total_time_ms > 0 else 0.0
                        avg_wait_ms = total_wait_time_ms / batch_count
                        avg_process_ms = total_process_time_ms / batch_count
                        lookahead_ms = lookahead_chunks * TTS_CHUNK_MS
                        logger.info(
                            f"MuseTalk SUMMARY: batch_size={batch_size} | batches={batch_count} | "
                            f"total_frames={total_frames_generated} | "
                            f"overall_sustained_fps={overall_sustained_fps:.1f} | "
                            f"avg_wait_ms={avg_wait_ms:.0f} | avg_process_ms={avg_process_ms:.0f} | "
                            f"lookahead_ms={lookahead_ms:.0f}"
                        )

            # ----------------------------------------------------------------
            # AV Sender - pairs audio+video and sends immediately
            # ----------------------------------------------------------------
            def av_sender() -> None:
                """
                Simple AV sender - pairs audio with video and sends immediately.
                No timing delays - client handles all buffering and playback.
                """
                last_video_frame: Optional[bytes] = None
                av_log_count = 0

                try:
                    while True:
                        try:
                            frame = audio_frame_queue.get(timeout=0.5)
                        except Empty:
                            if not is_generating():
                                break
                            continue

                        if frame is None:
                            break

                        # Get video frame (wait briefly if not ready yet)
                        video_bytes = None
                        # First check if it's already available (no wait)
                        with video_lock:
                            video_bytes = video_cache.pop(frame.index, None)
                        
                        # If not available, wait briefly (but not too long for tail frames)
                        if video_bytes is None and last_video_frame is not None:
                            # We have a fallback, so only wait a short time (200ms max)
                            for _ in range(4):
                                time.sleep(0.05)
                                with video_lock:
                                    video_bytes = video_cache.pop(frame.index, None)
                                if video_bytes is not None:
                                    break
                        elif video_bytes is None:
                            # No fallback yet, wait longer for first frames (500ms max)
                            for _ in range(10):
                                time.sleep(0.05)
                                with video_lock:
                                    video_bytes = video_cache.pop(frame.index, None)
                                if video_bytes is not None:
                                    break

                        # Use last frame as fallback
                        if video_bytes is None:
                            video_bytes = last_video_frame
                        else:
                            last_video_frame = video_bytes

                        # Create and send AV frame immediately
                        av_frame = AVFrame(
                            frame_index=frame.index,
                            audio_samples=frame.samples,
                            video_jpeg=video_bytes,
                            word=frame.word,
                            timestamp_ms=frame.timestamp_ms,
                            metrics=frame.metrics,
                        )

                        av_log_count += 1
                        if av_log_count <= 3 or av_log_count % 50 == 0 or frame.samples.size != self._samples_per_frame:
                            min_val = float(np.min(frame.samples)) if frame.samples.size > 0 else 0.0
                            max_val = float(np.max(frame.samples)) if frame.samples.size > 0 else 0.0
                            rms_val = (
                                float(np.sqrt(np.mean(frame.samples ** 2))) if frame.samples.size > 0 else 0.0
                            )
                            nan_count = int(np.isnan(frame.samples).sum()) if frame.samples.size > 0 else 0
                            logger.info(
                                f"AV frame {av_log_count} idx={frame.index}: samples={frame.samples.size} "
                                f"expected={self._samples_per_frame} min={min_val:.4f} max={max_val:.4f} "
                                f"rms={rms_val:.4f} nan={nan_count} video={'yes' if video_bytes else 'no'}"
                            )

                        report_frame(av_frame)

                except Exception as exc:
                    logger.error(f"AV sender error: {exc}")
                    import traceback
                    logger.error(traceback.format_exc())
                    report_error(f"AV sender error: {exc}")

                finally:
                    result.total_audio_ms = (total_audio_samples / self._sample_rate) * 1000
                    report_done()

            # ----------------------------------------------------------------
            # Run workers
            # ----------------------------------------------------------------
            threads = [
                threading.Thread(target=tts_worker, name="tts-worker", daemon=True),
                threading.Thread(target=musetalk_worker, name="musetalk-worker", daemon=True),
                threading.Thread(target=av_sender, name="av-sender", daemon=True),
            ]

            for t in threads:
                t.start()
            
            # Join threads with timeout to prevent hanging
            for t in threads:
                t.join(timeout=60.0)  # 60 second max wait per thread
                if t.is_alive():
                    logger.warning(f"Thread {t.name} did not complete within timeout")

        # Start the runner in a thread pool
        runner_future = loop.run_in_executor(None, runner)

        # Process output messages
        try:
            while True:
                try:
                    msg_type, payload = await asyncio.wait_for(
                        out_queue.get(),
                        timeout=1.0,
                    )

                    if msg_type == "frame":
                        on_frame(payload)
                    elif msg_type == "error":
                        logger.error(f"Pipeline error: {payload}")
                        on_error(payload)
                        result.success = False
                        result.error_message = payload
                        break
                    elif msg_type == "done":
                        break

                except asyncio.TimeoutError:
                    if runner_future.done():
                        break
                    continue

        finally:
            # Wait for runner with timeout to prevent hanging
            try:
                await asyncio.wait_for(
                    asyncio.shield(runner_future),
                    timeout=60.0,  # 60 second max wait
                )
            except asyncio.TimeoutError:
                logger.error("Pipeline runner timed out after 60 seconds")
                result.success = False
                result.error_message = "Pipeline timeout"
            except Exception as e:
                logger.error(f"Pipeline runner error: {e}")
            
            on_complete()

        return result


# ============================================================================
# Voice-to-Voice Pipeline
# ============================================================================

class VoiceToVoicePipeline:
    """
    Complete voice-to-voice pipeline.
    Orchestrates: Audio -> VAD -> STT -> LLM -> TTS -> MuseTalk -> Output
    """

    def __init__(
        self,
        tts_service: TTSService,
        musetalk_service: MuseTalkService,
        metrics_manager: StreamingMetricsManager,
        config: PipelineConfig,
    ):
        self.tts_service = tts_service
        self.musetalk_service = musetalk_service
        self.metrics_manager = metrics_manager
        self.config = config

        self.av_pipeline = AVPipeline(
            tts_service,
            musetalk_service,
            metrics_manager,
            config,
        )

    def get_metrics_stats(self) -> dict:
        """Get current generation stats."""
        return self.metrics_manager.get_stats()

    async def process_llm_and_tts_buffered(
        self,
        llm_generator,
        tts_session_id: int,
        video_enabled: bool,
        base_frame_index: int,
        check_cancelled: Callable[[], bool] = lambda: False,
    ):
        """
        Process LLM output through TTS and MuseTalk, yielding AV frames.
        
        This is used for speculative processing where we want to buffer frames.
        
        Args:
            llm_generator: Async generator yielding LLM tokens/text
            tts_session_id: TTS session ID (must be initialized)
            video_enabled: Whether to generate video
            base_frame_index: Starting frame index
            check_cancelled: Callback to check if processing should be cancelled
            
        Yields:
            AVFrame objects as they are generated
        """
        frame_queue: asyncio.Queue = asyncio.Queue()
        llm_response = ""
        pipeline_error = None
        
        def is_generating():
            return not check_cancelled()
        
        def on_llm_token(token: str, full_text: str):
            pass  # We don't need to handle tokens individually for speculative
        
        def on_av_frame(av_frame: AVFrame):
            # Put frame in queue for yielding
            try:
                frame_queue.put_nowait(("frame", av_frame))
            except asyncio.QueueFull:
                logger.warning("Frame queue full, dropping frame")
        
        def on_error(msg: str):
            nonlocal pipeline_error
            pipeline_error = msg
            frame_queue.put_nowait(("error", msg))
        
        def on_llm_complete(full_text: str):
            nonlocal llm_response
            llm_response = full_text
        
        def on_tts_complete():
            frame_queue.put_nowait(("done", None))
        
        # Run the existing pipeline method
        pipeline_task = asyncio.create_task(
            self.process_llm_and_tts(
                llm_generator=llm_generator,
                tts_session_id=tts_session_id,
                video_enabled=video_enabled,
                base_frame_index=base_frame_index,
                is_generating=is_generating,
                on_llm_token=on_llm_token,
                on_av_frame=on_av_frame,
                on_error=on_error,
                on_llm_complete=on_llm_complete,
                on_tts_complete=on_tts_complete,
            )
        )
        
        try:
            # Yield frames as they come
            while True:
                try:
                    msg_type, data = await asyncio.wait_for(frame_queue.get(), timeout=0.5)
                    
                    if msg_type == "done":
                        break
                    elif msg_type == "error":
                        logger.error(f"Pipeline error: {data}")
                        break
                    elif msg_type == "frame":
                        yield data
                        
                except asyncio.TimeoutError:
                    if check_cancelled():
                        pipeline_task.cancel()
                        break
                    # Check if pipeline task is done
                    if pipeline_task.done():
                        # Drain remaining frames
                        while not frame_queue.empty():
                            msg_type, data = frame_queue.get_nowait()
                            if msg_type == "frame":
                                yield data
                        break
                    continue
                    
        finally:
            if not pipeline_task.done():
                pipeline_task.cancel()
                try:
                    await pipeline_task
                except asyncio.CancelledError:
                    pass

    async def process_llm_and_tts(
        self,
        llm_generator,
        tts_session_id: int,
        video_enabled: bool,
        base_frame_index: int,
        is_generating: Callable[[], bool],
        on_llm_token: Callable[[str, str], None],
        on_av_frame: Callable[[AVFrame], None],
        on_error: Callable[[str], None],
        on_llm_complete: Callable[[str], None],
        on_tts_complete: Callable[[], None],
    ) -> str:
        """
        Process LLM output through TTS and MuseTalk.

        Args:
            llm_generator: Async generator yielding LLM tokens
            tts_session_id: TTS session ID (must be initialized)
            video_enabled: Whether to generate video
            base_frame_index: Starting frame index
            is_generating: Callback to check if generation should continue
            on_llm_token: Callback(token, full_text) for each LLM token
            on_av_frame: Callback for each AV frame
            on_error: Callback for errors
            on_llm_complete: Callback(full_text) when LLM completes
            on_tts_complete: Callback when TTS/video completes

        Returns:
            Complete LLM response text
        """
        text_input_queue: Queue = Queue()

        llm_response = ""
        chunker = StreamingTTSChunker(
            text_processor=StreamingTextProcessor(num_converter=NumberConverter()),
        )

        # LLM token processing task
        async def process_llm_tokens():
            nonlocal llm_response

            async for token in llm_generator:
                if not is_generating():
                    break

                llm_response += token
                on_llm_token(token, llm_response)

                # Use StreamingTTSChunker to handle word chunking with proper punctuation handling
                chunks = chunker.push_token(token)
                for chunk in chunks:
                    text_input_queue.put([chunk])

            on_llm_complete(llm_response)

            # Finalize - get any remaining chunks and flush signals
            final_chunks = chunker.finalize()
            for chunk in final_chunks:
                text_input_queue.put([chunk])

            text_input_queue.put(None)

        # Run LLM processing and AV pipeline concurrently
        llm_task = asyncio.create_task(process_llm_tokens())

        try:
            await self.av_pipeline.run(
                text_input_queue=text_input_queue,
                session_id=tts_session_id,
                video_enabled=video_enabled,
                base_frame_index=base_frame_index,
                is_generating=is_generating,
                on_frame=on_av_frame,
                on_error=on_error,
                on_complete=on_tts_complete,
            )
        finally:
            await llm_task

        return llm_response
