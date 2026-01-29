import json
import os
import time
import tempfile
import threading
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from pocket_tts import TTSModel
from pocket_tts.models import tts_model as pocket_tts_model
from pocket_tts.utils.utils import load_predefined_voice
from pocket_tts.modules.stateful_module import StatefulModule
from pocket_tts.modules import conv as pocket_conv
import scipy.io.wavfile


@dataclass
class SessionState:
    voice_state: dict
    voice_id: str
    last_activity: float = 0.0


class TritonPythonModel:
    def initialize(self, args):
        try:
            pb_utils.Logger.log_info("=== INITIALIZING POCKET-TTS (DECOUPLED) ===")
            self.model_config = json.loads(args["model_config"])
            params = self.model_config.get("parameters", {})

            os.environ.setdefault("HF_HOME", "/local_models/hf_cache")

            self.default_voice_id = self._get_param(params, "default_voice_id", "alba")

            self.tts_model = TTSModel.load_model()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tts_model = self.tts_model.to(device)
            self.tts_model.eval()
            torch.set_grad_enabled(False)

            # Ensure model state tensors are created on the same device as the module.
            pocket_tts_model.init_states = self._init_states_on_device
            self._patch_streaming_conv_state_init()

            self.sample_rate = int(self.tts_model.sample_rate)
            pb_utils.Logger.log_info(f"Pocket-TTS sample_rate={self.sample_rate}")

            self.sessions: Dict[int, SessionState] = {}
            self.sessions_lock = threading.Lock()
            self.voice_state_cache: Dict[str, dict] = {}

            # Warm default voice state
            _ = self._get_cached_voice_state(self.default_voice_id)

            pb_utils.Logger.log_info("Pocket-TTS Triton model initialized")

        except Exception as e:
            msg = f"Failed to initialize: {e}\n{traceback.format_exc()}"
            pb_utils.Logger.log_error(msg)
            raise

    def _init_states_on_device(self, model, batch_size: int, sequence_length: int):
        result = {}
        for module_name, module in model.named_modules():
            if not isinstance(module, StatefulModule):
                continue
            module._module_absolute_name = module_name
            module_state = module.init_state(batch_size, sequence_length=sequence_length)

            device = None
            try:
                device = next(module.parameters()).device
            except StopIteration:
                for _, buf in module.named_buffers(recurse=False):
                    device = buf.device
                    break
            if device is None:
                device = torch.device("cpu")

            for key, value in module_state.items():
                if torch.is_tensor(value):
                    module_state[key] = value.to(device=device)
            result[module_name] = module_state
        return result

    def _patch_streaming_conv_state_init(self) -> None:
        """Ensure streaming conv state tensors are allocated on module device."""
        if not hasattr(pocket_conv.StreamingConv1d, "_orig_init_state"):
            pocket_conv.StreamingConv1d._orig_init_state = pocket_conv.StreamingConv1d.init_state

            def _init_state_on_device_conv1d(self, batch_size: int, sequence_length: int):
                state = pocket_conv.StreamingConv1d._orig_init_state(
                    self,
                    batch_size,
                    sequence_length,
                )
                device = self.conv.weight.device
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device=device)
                return state

            pocket_conv.StreamingConv1d.init_state = _init_state_on_device_conv1d

        if not hasattr(pocket_conv.StreamingConvTranspose1d, "_orig_init_state"):
            pocket_conv.StreamingConvTranspose1d._orig_init_state = (
                pocket_conv.StreamingConvTranspose1d.init_state
            )

            def _init_state_on_device_convtr(self, batch_size: int, sequence_length: int):
                state = pocket_conv.StreamingConvTranspose1d._orig_init_state(
                    self,
                    batch_size,
                    sequence_length,
                )
                device = self.convtr.weight.device
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device=device)
                return state

            pocket_conv.StreamingConvTranspose1d.init_state = _init_state_on_device_convtr

    def _get_param(self, params: Dict[str, Any], key: str, default: str) -> str:
        if key in params:
            value = params[key]["string_value"]
        else:
            value = default
        pb_utils.Logger.log_info(f"Parameter {key}: {value}")
        return value

    def _get_bool_input(self, request, name: str, default: bool = False) -> bool:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        arr = tensor.as_numpy()
        if arr.size == 0:
            return default
        return bool(arr.squeeze())

    def _get_int_input(self, request, name: str, default: Optional[int] = None) -> Optional[int]:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return default
        arr = tensor.as_numpy()
        if arr.size == 0:
            return default
        return int(arr.squeeze())

    def _get_text_list(self, request) -> List[str]:
        tensor = pb_utils.get_input_tensor_by_name(request, "TEXTS")
        if tensor is None:
            return []
        arr = tensor.as_numpy()
        if arr.size == 0:
            return []
        arr = arr.reshape(-1)
        return [x.decode("utf-8") for x in arr]

    def _get_str_input(self, request, name: str) -> Optional[str]:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return None
        arr = tensor.as_numpy()
        if arr.size == 0:
            return None
        value = arr.reshape(-1)[0]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _get_audio_prompt(self, request) -> Optional[np.ndarray]:
        tensor = pb_utils.get_input_tensor_by_name(request, "VOICE_PROMPT_PCM")
        if tensor is None:
            return None
        arr = tensor.as_numpy()
        if arr.size == 0:
            return None
        return np.asarray(arr, dtype=np.float32).reshape(-1)

    def _get_cached_voice_state(self, voice_id: str) -> dict:
        if voice_id in self.voice_state_cache:
            return self.voice_state_cache[voice_id]
        # Predefined voices return CPU tensors; move to model device before prompting.
        if voice_id in pocket_tts_model.PREDEFINED_VOICES:
            prompt = load_predefined_voice(voice_id).to(self.tts_model.device)
            model_state = self._init_states_on_device(self.tts_model.flow_lm, batch_size=1, sequence_length=1000)
            self.tts_model._run_flow_lm_and_increment_step(
                model_state=model_state,
                audio_conditioning=prompt,
            )
            self.tts_model._slice_kv_cache(model_state, prompt.shape[1])
            state = model_state
        else:
            state = self.tts_model.get_state_for_audio_prompt(voice_id, truncate=True)
        self.voice_state_cache[voice_id] = state
        return state

    def _write_prompt_wav(self, audio_prompt: np.ndarray, sample_rate: Optional[int]) -> str:
        audio = np.asarray(audio_prompt, dtype=np.float32).reshape(-1)
        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767.0).astype(np.int16)
        sr = int(sample_rate) if sample_rate else self.sample_rate
        temp_file = tempfile.NamedTemporaryFile(
            prefix="voice_prompt_",
            suffix=".wav",
            delete=False,
        )
        temp_path = temp_file.name
        temp_file.close()
        scipy.io.wavfile.write(temp_path, sr, pcm16)
        return temp_path

    def _voice_state_from_prompt(self, audio_prompt: np.ndarray, sample_rate: Optional[int]) -> dict:
        prompt_path = self._write_prompt_wav(audio_prompt, sample_rate)
        try:
            return self.tts_model.get_state_for_audio_prompt(prompt_path, truncate=True)
        finally:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass

    def _send_final(self, sender):
        sender.send(self._empty_response(), flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    def execute(self, requests):
        for request in requests:
            sender = request.get_response_sender()
            self._handle_request_decoupled(request, sender)

    def _handle_request_decoupled(self, request, sender):
        texts = self._get_text_list(request)
        is_start = self._get_bool_input(request, "START", False)
        is_end = self._get_bool_input(request, "END", False)
        seq_id = self._get_int_input(request, "CORRID", None)

        voice_prompt = self._get_audio_prompt(request)
        voice_prompt_sr = self._get_int_input(request, "VOICE_PROMPT_SAMPLE_RATE", None)
        voice_id = self._get_str_input(request, "VOICE_ID")

        if seq_id is None:
            self._send_final(sender)
            raise RuntimeError("CORRID input is required")

        if is_start:
            try:
                if voice_prompt is not None and voice_prompt.size > 0:
                    voice_state = self._voice_state_from_prompt(voice_prompt, voice_prompt_sr)
                    effective_voice_id = "prompt"
                else:
                    effective_voice_id = voice_id or self.default_voice_id
                    voice_state = self._get_cached_voice_state(effective_voice_id)

                with self.sessions_lock:
                    self.sessions[seq_id] = SessionState(
                        voice_state=voice_state,
                        voice_id=effective_voice_id,
                        last_activity=time.time(),
                    )
                pb_utils.Logger.log_info(f"[Seq {seq_id}] Voice initialized: {effective_voice_id}")
                self._send_final(sender)
                return
            except Exception as e:
                pb_utils.Logger.log_error(f"[Seq {seq_id}] Voice init failed: {e}")
                self._send_final(sender)
                raise

        with self.sessions_lock:
            state = self.sessions.get(seq_id)

        if state is None:
            self._send_final(sender)
            raise RuntimeError("You must initialize the session before sending text")

        if is_end:
            with self.sessions_lock:
                self.sessions.pop(seq_id, None)
            pb_utils.Logger.log_info(f"[Seq {seq_id}] Session ended")
            self._send_final(sender)
            return

        start = time.perf_counter()
        total_samples = 0

        try:
            for text in texts:
                if not text or not text.strip():
                    continue
                for audio_chunk in self.tts_model.generate_audio_stream(
                    model_state=state.voice_state,
                    text_to_generate=text,
                    copy_state=True,
                ):
                    audio_np = audio_chunk.detach().cpu().float().numpy().reshape(-1)
                    if audio_np.size == 0:
                        continue
                    total_samples += int(audio_np.size)
                    audio_tensor = pb_utils.Tensor("AUDIO_FRAME", audio_np)
                    sender.send(pb_utils.InferenceResponse(output_tensors=[audio_tensor]))

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            audio_ms = (total_samples / float(self.sample_rate)) * 1000.0 if total_samples > 0 else 0.0
            rtf = (elapsed_ms / audio_ms) if audio_ms > 0 else 0.0
            pb_utils.Logger.log_info(
                "TTS | rtf=%.3f | total_ms=%.1f | audio_ms=%.1f"
                % (rtf, elapsed_ms, audio_ms)
            )
        except Exception as e:
            pb_utils.Logger.log_error(f"[Seq {seq_id}] TTS generation error: {e}")
            pb_utils.Logger.log_error(traceback.format_exc())
        finally:
            self._send_final(sender)

    def _empty_response(self):
        audio = np.zeros((0,), dtype=np.float32)
        audio_tensor = pb_utils.Tensor("AUDIO_FRAME", audio)
        return pb_utils.InferenceResponse(output_tensors=[audio_tensor])

    def finalize(self):
        try:
            pb_utils.Logger.log_info("Finalizing TTS model...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            msg = f"Error in finalize: {e}\n{traceback.format_exc()}"
            pb_utils.Logger.log_error(msg)
