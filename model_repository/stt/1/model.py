import numpy as np
import torch
import triton_python_backend_utils as pb_utils
import nemo.collections.asr as nemo_asr
import soundfile as sf
import time

class TritonPythonModel:

    def initialize(self, args):
        """
        This runs ONCE when Triton loads the model.
        """
        model_path = "/local_models/stt_model/fast_conformer_georgian.nemo"

        # Load NeMo ASR model (FastConformer)
        self.asr_model = nemo_asr.models.ASRModel.restore_from(
            restore_path=model_path, map_location="cuda"
        )
        self.asr_model.eval()
        torch.set_grad_enabled(False)

    def execute(self, requests):
        responses = []
        for req in requests:

            # Get PCM audio input (float32, 1D array)
            audio = pb_utils.get_input_tensor_by_name(req, "AUDIO_PCM").as_numpy()

            # NeMo expects a list of file paths OR arrays
            # but transcribe() supports raw waveform via parameter
            start_time = time.perf_counter()
            result = self.asr_model.transcribe(
                audio=[audio],
                batch_size=1
            )[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - start_time) * 1000.0

            # Handle different return types - NeMo can return Hypothesis object or string
            transcript = result.text
            audio_ms = (len(audio) / 16000.0) * 1000.0
            rtf = (infer_ms / audio_ms) if audio_ms > 0 else 0.0
            pb_utils.Logger.log_info(
                f"STT: audio_ms={audio_ms:.1f} infer_ms={infer_ms:.1f} rtf={rtf:.3f} "
                f"chars={len(transcript)}"
            )

            out_tensor = pb_utils.Tensor(
                "TRANSCRIPT", np.array([transcript.encode("utf-8")], dtype=object)
            )

            responses.append(pb_utils.InferenceResponse([out_tensor]))

        return responses
