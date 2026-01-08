import asyncio
import os
import sys
import unittest
from queue import Queue

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import load_config
from pipeline import AVPipeline, StreamingMetricsManager, PipelineConfig


class FakeTTSService:
    def __init__(self, audio_chunks):
        self._audio_chunks = list(audio_chunks)

    def generate_stream(self, text_chunks, session_id, on_audio=None):
        if not self._audio_chunks:
            return iter(())
        audio = self._audio_chunks.pop(0)
        if on_audio:
            on_audio(audio, "", None)
        yield audio, "", None


class DummyMuseTalk:
    def generate_frames(self, *args, **kwargs):
        return iter(())


class TestPipelineFraming(unittest.TestCase):
    def test_audio_framing_is_contiguous(self):
        config = load_config()

        audio_chunks = [
            np.arange(0, 1000, dtype=np.float32),
            np.arange(1000, 3500, dtype=np.float32),
            np.arange(3500, 3900, dtype=np.float32),
        ]
        expected = np.concatenate(audio_chunks)

        tts_service = FakeTTSService(audio_chunks)
        metrics_manager = StreamingMetricsManager(config.streaming)
        pipeline_config = PipelineConfig(
            tts_config=config.tts,
            musetalk_config=config.musetalk,
            streaming_config=config.streaming,
        )
        pipeline = AVPipeline(
            tts_service=tts_service,
            musetalk_service=DummyMuseTalk(),
            metrics_manager=metrics_manager,
            config=pipeline_config,
        )

        text_input_queue = Queue()
        for _ in range(3):
            text_input_queue.put(["chunk"])
        text_input_queue.put(None)

        frames = []

        async def run_pipeline():
            await pipeline.run(
                text_input_queue=text_input_queue,
                session_id=123,
                video_enabled=False,
                base_frame_index=0,
                is_generating=lambda: True,
                on_frame=frames.append,
                on_error=lambda msg: (_ for _ in ()).throw(RuntimeError(msg)),
                on_complete=lambda: None,
            )

        asyncio.run(run_pipeline())

        self.assertTrue(frames, "No frames captured from pipeline")
        frames.sort(key=lambda f: f.frame_index)

        output = np.concatenate([frame.audio_samples for frame in frames])
        self.assertEqual(output.size, expected.size)
        self.assertTrue(np.allclose(output, expected))

        indices = [frame.frame_index for frame in frames]
        self.assertEqual(indices, list(range(indices[0], indices[0] + len(indices))))


if __name__ == "__main__":
    unittest.main()
