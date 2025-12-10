import numpy as np
import torch
import triton_python_backend_utils as pb_utils
import nemo.collections.asr as nemo_asr
import soundfile as sf

class TritonPythonModel:

    def initialize(self, args):
        """
        This runs ONCE when Triton loads the model.
        """
        model_path = "/local_models/stt_model/stt_ka_fastconformer_hybrid_large_pc.nemo"

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
            result = self.asr_model.transcribe(
                audio=[audio],
                batch_size=1
            )[0]

            # Handle different return types - NeMo can return Hypothesis object or string
            transcript = result.text

            out_tensor = pb_utils.Tensor(
                "TRANSCRIPT", np.array([transcript.encode("utf-8")], dtype=object)
            )

            responses.append(pb_utils.InferenceResponse([out_tensor]))

        return responses
