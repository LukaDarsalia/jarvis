import numpy as np
import torch
import triton_python_backend_utils as pb_utils
import os
import time

class TritonPythonModel:

    def initialize(self, args):
        self.model = torch.jit.load("/local_models/vad_model/silero_vad.jit", map_location="cpu")
        self.model.eval()

        # State for hysteresis
        self.silence_ms = 0
        self.sample_rate = 16000
        self.chunk_samples = 512  # Silero VAD requires exactly 512 samples at 16kHz
        self.frame_ms = self.chunk_samples / self.sample_rate * 1000  # ~32ms
        self.silence_threshold_ms = 1000  # 0.4 sec silence = end-of-speech
        self.log_every_n = int(os.environ.get("VAD_LOG_EVERY_N", "200"))
        self._req_count = 0

    def execute(self, requests):
        responses = []

        for req in requests:
            start_time = time.perf_counter()
            audio = pb_utils.get_input_tensor_by_name(req, "AUDIO_PCM").as_numpy()

            # Process audio in chunks of 512 samples
            # For now, if the audio is longer, we process only the last chunk
            # In a real streaming scenario, you'd maintain state across calls
            if len(audio) > self.chunk_samples:
                # Process the last chunk for simplicity
                audio = audio[-self.chunk_samples:]
            elif len(audio) < self.chunk_samples:
                # Pad if too short
                audio = np.pad(audio, (0, self.chunk_samples - len(audio)))

            # Silero expects torch.tensor of shape [wav_len]
            wav = torch.from_numpy(audio).float()

            # probability of speech
            prob = self.model(wav, self.sample_rate).item()

            # classification
            is_speech = int(prob > 0.5)

            if is_speech:
                self.silence_ms = 0
            else:
                self.silence_ms += self.frame_ms

            end_of_utt = int(self.silence_ms >= self.silence_threshold_ms)

            outputs = [
                pb_utils.Tensor("IS_SPEECH", np.array([is_speech], dtype=np.int32)),
                pb_utils.Tensor("PROB", np.array([prob], dtype=np.float32)),
                pb_utils.Tensor("END_OF_UTTERANCE", np.array([end_of_utt], dtype=np.int32)),
            ]

            responses.append(pb_utils.InferenceResponse(outputs))
            self._req_count += 1
            if self.log_every_n > 0 and self._req_count % self.log_every_n == 0:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                pb_utils.Logger.log_info(
                    f"VAD: req={self._req_count} audio_samples={len(audio)} "
                    f"elapsed_ms={elapsed_ms:.2f} prob={prob:.3f} speech={is_speech} end={end_of_utt}"
                )

        return responses
