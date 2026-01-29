import os
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
import nemo.collections.asr as nemo_asr
import time


def _find_nemo_file(model_dir: str) -> str:
    """Find a .nemo file inside the model directory."""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"STT model directory not found: {model_dir}")

    candidates = []
    for root, _, files in os.walk(model_dir):
        for filename in files:
            if filename.endswith(".nemo"):
                candidates.append(os.path.join(root, filename))

    if not candidates:
        raise FileNotFoundError(f"No .nemo files found under {model_dir}")

    # Prefer a deterministic order
    candidates.sort()
    return candidates[0]

class TritonPythonModel:

    def initialize(self, args):
        """
        This runs ONCE when Triton loads the model.
        """
        model_dir = os.environ.get("STT_MODEL_DIR", "/local_models/stt_model")
        model_path = _find_nemo_file(model_dir)

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
            start = time.perf_counter()
            result = self.asr_model.transcribe(
                audio=[audio],
                batch_size=1
            )[0]
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            # Handle different return types - NeMo can return Hypothesis object or string
            transcript = result.text

            out_tensor = pb_utils.Tensor(
                "TRANSCRIPT", np.array([transcript.encode("utf-8")], dtype=object)
            )

            responses.append(pb_utils.InferenceResponse([out_tensor]))
            pb_utils.Logger.log_info(f"STT | total_ms={elapsed_ms:.1f} | chars={len(transcript)}")

        return responses
