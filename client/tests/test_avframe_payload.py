import base64
import os
import sys
import unittest
import zlib

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import AVFrame


class TestAVFramePayload(unittest.TestCase):
    def test_audio_roundtrip_crc(self):
        audio = np.linspace(-1.0, 1.0, 960, dtype=np.float32)
        video_bytes = b"jpegbytes"
        frame = AVFrame(
            frame_index=7,
            audio_samples=audio,
            video_jpeg=video_bytes,
            word="hello",
            timestamp_ms=280.0,
            metrics=None,
        )

        payload = frame.to_websocket_payload()
        self.assertEqual(payload["frame_index"], 7)
        self.assertEqual(payload["audio_samples"], audio.size)

        audio_raw = base64.b64decode(payload["audio"])
        crc = zlib.crc32(audio_raw) & 0xFFFFFFFF
        self.assertEqual(payload["audio_crc32"], crc)

        audio_rt = np.frombuffer(audio_raw, dtype=np.float32)
        self.assertTrue(np.allclose(audio_rt, audio))

        video_raw = base64.b64decode(payload["frame"])
        self.assertEqual(video_raw, video_bytes)

    def test_empty_audio_payload(self):
        frame = AVFrame(
            frame_index=1,
            audio_samples=np.zeros((0,), dtype=np.float32),
            video_jpeg=None,
            word="",
            timestamp_ms=40.0,
            metrics=None,
        )
        payload = frame.to_websocket_payload()
        self.assertEqual(payload["audio_samples"], 0)
        self.assertEqual(payload["audio"], "")
        self.assertEqual(payload["audio_crc32"], 0)


if __name__ == "__main__":
    unittest.main()
