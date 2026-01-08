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
from models import AudioFrame, AVFrame, TTSMetrics, BufferConfig, StreamingStats
from tts_service import TTSService
from triton_services import MuseTalkService

logger = logging.getLogger(__name__)


# ============================================================================
# Streaming Metrics Manager (simplified - client handles adaptive buffering)
# ============================================================================

class StreamingMetricsManager:
    """Tracks generation timing statistics for client-side adaptive buffering."""

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
        """Get current stats for client-side adaptive buffering."""
        with self._lock:
            return {
                "tts_rtf_mean": self._tts_rtf.mean,
                "tts_rtf_std": self._tts_rtf.std,
                "musetalk_fps_mean": self._musetalk_fps.mean,
                "musetalk_fps_std": self._musetalk_fps.std,
            }

    def calculate_buffer_config(self) -> BufferConfig:
        """Calculate buffer configuration for client."""
        with self._lock:
            tts_mean = self._tts_rtf.mean
            tts_std = self._tts_rtf.std
            musetalk_mean = self._musetalk_fps.mean if self._musetalk_fps.mean > 0 else 25.0
            musetalk_std = self._musetalk_fps.std

            # Suggest buffer based on RTF - client will use this
            k = self.config.buffer_k_std
            worst_case_rtf = max(0.0, tts_mean + k * tts_std) if tts_mean > 0 else 1.0

            # Suggest minimum buffer frames based on generation speed
            suggested_buffer_ms = max(160.0, worst_case_rtf * 100.0)
            suggested_frames = max(4, int(suggested_buffer_ms / 40.0))

            return BufferConfig(
                buffer_ms=suggested_buffer_ms,
                frame_buffer=suggested_frames,
                buffer_source="server_suggested",
                manual_buffer_ms=None,
                is_calibrated=tts_mean > 0,
                tts_rtf_mean=tts_mean,
                tts_rtf_std=tts_std,
                tts_overage_ms=0,
                musetalk_fps_mean=musetalk_mean,
                musetalk_fps_std=musetalk_std,
                musetalk_overage_ms=0,
                network_latency_mean_ms=0,
                network_latency_std_ms=0,
                network_buffer_ms=0,
            )


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

                # Batch audio for efficient MuseTalk processing
                # 480ms = 12 frames minimum for good quality
                min_audio_samples = int(0.48 * self._sample_rate)
                max_audio_samples = int(2.0 * self._sample_rate)  # 2 seconds max

                audio_buffer = np.zeros((0,), dtype=np.float32)
                frame_buffer: List[AudioFrame] = []
                next_frame_index = base_frame_index
                done = False
                
                # Adaptive skip factor tracking
                musetalk_config = self.config.musetalk_config
                current_skip_factor = musetalk_config.skip_factor
                adaptive_skip_enabled = musetalk_config.adaptive_skip
                lag_threshold_ms = musetalk_config.adaptive_skip_lag_threshold_ms
                last_generation_end_time: Optional[float] = None
                cumulative_lag_ms = 0.0

                try:
                    while is_generating() or not done:
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

                        # Decide whether to process
                        should_process = False
                        if done and len(audio_buffer) > 0:
                            # Skip tiny tail chunks; fallback video will reuse last frame.
                            if len(audio_buffer) < min_audio_samples:
                                break
                            should_process = True
                        elif len(audio_buffer) >= max_audio_samples:
                            should_process = True
                        elif len(audio_buffer) >= min_audio_samples:
                            should_process = True

                        if not should_process:
                            continue

                        if len(audio_buffer) == 0:
                            continue

                        # Process the accumulated audio
                        process_audio = audio_buffer.copy()
                        process_start_index = next_frame_index
                        audio_duration_ms = (len(process_audio) / self._sample_rate) * 1000.0
                        expected_frames = int(len(process_audio) / self._samples_per_frame)

                        # Clear buffers
                        audio_buffer = np.zeros((0,), dtype=np.float32)
                        frame_buffer = []

                        # Adaptive skip factor: increase if falling behind
                        if adaptive_skip_enabled and last_generation_end_time is not None:
                            # Calculate how much we're lagging behind realtime
                            time_since_last = (time.time() - last_generation_end_time) * 1000.0
                            expected_time = audio_duration_ms  # Should take this long in realtime
                            lag_ms = time_since_last - expected_time
                            cumulative_lag_ms += lag_ms
                            
                            # Adjust skip factor based on cumulative lag
                            if cumulative_lag_ms > lag_threshold_ms * 3:
                                current_skip_factor = min(4, current_skip_factor + 1)
                            elif cumulative_lag_ms > lag_threshold_ms * 2:
                                current_skip_factor = min(3, max(current_skip_factor, 2))
                            elif cumulative_lag_ms > lag_threshold_ms:
                                current_skip_factor = max(2, current_skip_factor)
                            elif cumulative_lag_ms < 0:
                                # Catching up, reduce skip factor
                                current_skip_factor = max(1, current_skip_factor - 1)
                                cumulative_lag_ms = max(0, cumulative_lag_ms)

                        # Generate video frames
                        start_time = time.time()
                        frames_generated = 0

                        skip_info = f", skip={current_skip_factor}" if current_skip_factor > 1 else ""
                        logger.info(
                            f"MuseTalk: Processing {len(process_audio)/self._sample_rate:.3f}s audio "
                            f"starting at index {process_start_index}{skip_info}"
                        )

                        for frame_bytes, frame_idx, _ in self.musetalk_service.generate_frames(
                            process_audio,
                            frame_index=process_start_index,
                            skip_factor=current_skip_factor,
                        ):
                            frames_generated += 1
                            with video_lock:
                                video_cache[frame_idx] = frame_bytes
                        
                        last_generation_end_time = time.time()

                        # Update next frame index (advance by expected frames, not generated)
                        next_frame_index = process_start_index + expected_frames

                        # Record metrics
                        if frames_generated > 0:
                            duration = time.time() - start_time
                            if duration > 0:
                                self.metrics_manager.record_musetalk_generation(
                                    frames_generated, duration
                                )
                            lag_info = f", lag={cumulative_lag_ms:.0f}ms" if adaptive_skip_enabled else ""
                            logger.info(
                                f"MuseTalk: Generated {frames_generated}/{expected_frames} frames in {duration:.3f}s{lag_info}"
                            )

                        if done:
                            break

                except Exception as exc:
                    logger.error(f"MuseTalk worker error: {exc}")
                    import traceback
                    logger.error(traceback.format_exc())
                    report_error(f"MuseTalk worker error: {exc}")

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
                        for _ in range(10):  # Wait up to 1 second for video
                            with video_lock:
                                video_bytes = video_cache.pop(frame.index, None)
                            if video_bytes is not None:
                                break
                            time.sleep(0.1)

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
            for t in threads:
                t.join()

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
            await runner_future
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

    def get_buffer_config(self) -> BufferConfig:
        """Get suggested buffer configuration for client."""
        return self.metrics_manager.calculate_buffer_config()

    def get_metrics_stats(self) -> dict:
        """Get metrics stats for client-side adaptive buffering."""
        return self.metrics_manager.get_stats()

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
        tts_started = False
        words_sent = 0

        # LLM token processing task
        async def process_llm_tokens():
            nonlocal llm_response, tts_started, words_sent

            async for token in llm_generator:
                if not is_generating():
                    break

                llm_response += token
                on_llm_token(token, llm_response)

                words = llm_response.split()

                # Start TTS after 3 words (2-word lookahead)
                if not tts_started and len(words) >= 3:
                    first_chunk = " ".join(words[:3])
                    text_input_queue.put([first_chunk])
                    tts_started = True
                    words_sent = 3

                # Send new words one at a time
                if tts_started and len(words) > words_sent:
                    new_words = words[words_sent:]
                    for w in new_words:
                        text_input_queue.put([" " + w])
                    words_sent = len(words)

            on_llm_complete(llm_response)

            # Send flush chunks and end signal
            if tts_started:
                text_input_queue.put([""])
                text_input_queue.put([""])

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
