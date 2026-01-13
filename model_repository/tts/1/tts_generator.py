import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Generator
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import soundfile as sf
from transformers import CsmForConditionalGeneration, AutoProcessor

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BACKBONE_ONNX_FILENAME = "csm_backbone_step_past4095.onnx"
DEPTH_DECODER_ONNX_FILENAME = "csm_depth_decoder_step_past31.onnx"
BACKBONE_PAST_LEN = 4095
DEPTH_DECODER_PAST_LEN = 31
_SPEECH_TEXT_RE = re.compile(r"[0-9A-Za-z\u10A0-\u10FF]")


def sample_from_logits(
    logits: torch.Tensor,
    method: str = "nucleus",
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    if method == "greedy":
        return torch.argmax(logits, dim=-1, keepdim=True)

    temperature = max(float(temperature), 1e-8)
    probs = F.softmax(logits / temperature, dim=-1)

    if method == "multinomial":
        return torch.multinomial(probs, num_samples=1)

    if method == "topk":
        topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(topk_probs, num_samples=1)
        return topk_idx.gather(-1, sampled)

    if method == "nucleus":
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumprobs = sorted_probs.cumsum(dim=-1)
        mask = cumprobs <= top_p
        mask[..., 0] = True
        filtered_probs = sorted_probs * mask
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(filtered_probs, num_samples=1)
        return sorted_idx.gather(-1, sampled)

    raise ValueError(f"Unknown sampling method: {method}")


def _pick_providers(device_id: int):
    provs = ort.get_available_providers()
    torch_stream = torch.cuda.default_stream(device_id).cuda_stream

    providers = []
    if "TensorrtExecutionProvider" in provs:
        providers.append(("TensorrtExecutionProvider", {"device_id": device_id}))
    providers.append((
        "CUDAExecutionProvider",
        {"device_id": device_id, "user_compute_stream": torch_stream},
    ))
    providers.append("CPUExecutionProvider")
    return providers


def _create_ort_session(onnx_path: str, device_id: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        onnx_path,
        sess_options=so,
        providers=_pick_providers(device_id),
    )


class ORTDepthDecoder:
    def __init__(
        self,
        onnx_path: str,
        past_len: int,
        device_id: int = 0,
        sess: Optional[ort.InferenceSession] = None,
    ):
        self.device_id = int(device_id)
        self.device = torch.device(f"cuda:{self.device_id}")

        self.sess = sess if sess is not None else _create_ort_session(onnx_path, self.device_id)

        ins = self.sess.get_inputs()
        outs = self.sess.get_outputs()

        self.L = (len(ins) - 4) // 2
        B, KVH, P, HD = map(int, ins[4].shape)
        assert P == past_len

        B2, S2, V = map(int, outs[0].shape)
        assert B2 == B and S2 == 1

        B3, Hbb = map(int, ins[1].shape)
        assert B3 == B

        self.B, self.KVH, self.HD, self.V, self.Hbb = B, KVH, HD, V, Hbb
        self.past_len = past_len

        self.token_buf = torch.zeros((B, 1), device=self.device, dtype=torch.long)
        self.bb_buf = torch.zeros((B, Hbb), device=self.device, dtype=torch.float16)
        self.cache_pos = torch.zeros((1,), device=self.device, dtype=torch.long)
        self.mask_buf = torch.zeros((B, 1, 1, past_len + 1), device=self.device, dtype=torch.float16)

        self.past = [
            torch.zeros((B, KVH, past_len, HD), device=self.device, dtype=torch.float16)
            for _ in range(2 * self.L)
        ]
        self.present = [
            torch.empty((B, KVH, past_len + 1, HD), device=self.device, dtype=torch.float16)
            for _ in range(2 * self.L)
        ]
        self.out_logits = torch.empty((B, 1, V), device=self.device, dtype=torch.float16)

        self.io = self.sess.io_binding()
        self.io.bind_input("input_ids", "cuda", self.device_id, np.int64, self.token_buf.shape, self.token_buf.data_ptr())
        self.io.bind_input(
            "backbone_last_hidden_state",
            "cuda",
            self.device_id,
            np.float16,
            self.bb_buf.shape,
            self.bb_buf.data_ptr(),
        )
        self.io.bind_input(
            "attention_mask",
            "cuda",
            self.device_id,
            np.float16,
            self.mask_buf.shape,
            self.mask_buf.data_ptr(),
        )
        self.io.bind_input("cache_position", "cuda", self.device_id, np.int64, self.cache_pos.shape, self.cache_pos.data_ptr())

        for i in range(2 * self.L):
            t = self.past[i]
            self.io.bind_input(f"past_{i}", "cuda", self.device_id, np.float16, t.shape, t.data_ptr())

        self.io.bind_output("logits", "cuda", self.device_id, np.float16, self.out_logits.shape, self.out_logits.data_ptr())
        for i in range(2 * self.L):
            t = self.present[i]
            self.io.bind_output(f"present_{i}", "cuda", self.device_id, np.float16, t.shape, t.data_ptr())

    @torch.no_grad()
    def _set_mask_for_pos(self, pos: int):
        valid = min(int(pos), self.past_len)
        start = self.past_len - valid
        self.mask_buf.zero_()
        if start > 0:
            self.mask_buf[..., :start] = -1e4

    @torch.no_grad()
    def reset(self):
        for t in self.past:
            t.zero_()

    @torch.no_grad()
    def step(self, token_id: torch.Tensor, h_last: torch.Tensor, pos: int) -> torch.Tensor:
        if token_id.dim() == 1:
            token_id = token_id.view(self.B, 1)
        self.token_buf.copy_(token_id)
        self.bb_buf.copy_(h_last)
        self.cache_pos.fill_(int(pos))
        self._set_mask_for_pos(pos)
        torch.cuda.synchronize()
        if hasattr(self.io, "synchronize_inputs"):
            self.io.synchronize_inputs()
        self.sess.run_with_iobinding(self.io)
        if hasattr(self.io, "synchronize_outputs"):
            self.io.synchronize_outputs()
        torch.cuda.synchronize()
        for i in range(2 * self.L):
            self.past[i].copy_(self.present[i][:, :, 1:, :])

        return self.out_logits[:, 0, :]

    @torch.no_grad()
    def generate_frame(
        self,
        c0: torch.Tensor,
        h_last: torch.Tensor,
        method: str = "nucleus",
        temperature: float = 0.1,
        top_p: float = 0.999,
    ):
        if c0.dim() == 2:
            c0 = c0[:, 0]
        c0 = c0.to(device=self.device, dtype=torch.long).view(self.B, 1)

        self.reset()

        dummy = torch.zeros((self.B, 1), device=self.device, dtype=torch.long)
        _ = self.step(dummy, h_last, pos=0)

        logits = self.step(c0, h_last, pos=1)
        c1 = sample_from_logits(logits, method=method, temperature=temperature, top_p=top_p)

        outs = [c1]
        prev = c1
        for pos in range(2, 32):
            logits = self.step(prev, h_last, pos=pos)
            nxt = sample_from_logits(logits, method=method, temperature=temperature, top_p=top_p)
            outs.append(nxt)
            prev = nxt

        rest = torch.cat(outs, dim=1)
        return torch.cat([c0, rest], dim=1)


class ORTBackbonePastN:
    def __init__(
        self,
        onnx_path: str,
        past_len: int,
        device_id: int = 0,
        sess: Optional[ort.InferenceSession] = None,
    ):
        self.device_id = int(device_id)
        self.device = torch.device(f"cuda:{self.device_id}")

        self.sess = sess if sess is not None else _create_ort_session(onnx_path, self.device_id)

        ins = self.sess.get_inputs()
        outs = self.sess.get_outputs()

        self.L = (len(ins) - 3) // 2

        B, KVH, P, HD = map(int, ins[3].shape)
        assert P == past_len

        B2, S2, V = map(int, outs[0].shape)
        assert B2 == B and S2 == 1

        B3, S3, H = map(int, ins[0].shape)
        assert B3 == B and S3 == 1

        B4, H2 = map(int, outs[1].shape)
        assert B4 == B and H2 == H

        B5, KVH2, ONE, HD2 = map(int, outs[2].shape)
        assert B5 == B and KVH2 == KVH and ONE == 1 and HD2 == HD

        self.B, self.KVH, self.HD, self.V, self.H = B, KVH, HD, V, H
        self.past_len = past_len

        self.embed_buf = torch.zeros((B, 1, H), device=self.device, dtype=torch.float16)
        self.mask_buf = torch.zeros((B, 1, 1, past_len + 1), device=self.device, dtype=torch.float16)
        self.cache_pos = torch.zeros((1,), device=self.device, dtype=torch.long)

        self.past = [
            torch.zeros((B, KVH, past_len, HD), device=self.device, dtype=torch.float16)
            for _ in range(2 * self.L)
        ]
        self.out_logits = torch.empty((B, 1, V), device=self.device, dtype=torch.float16)
        self.out_hidden = torch.empty((B, H), device=self.device, dtype=torch.float16)
        self.new_kv = [
            torch.empty((B, KVH, 1, HD), device=self.device, dtype=torch.float16)
            for _ in range(2 * self.L)
        ]

        self.io = self.sess.io_binding()
        self.io.bind_input(
            "inputs_embeds",
            "cuda",
            self.device_id,
            np.float16,
            self.embed_buf.shape,
            self.embed_buf.data_ptr(),
        )
        self.io.bind_input(
            "attention_mask",
            "cuda",
            self.device_id,
            np.float16,
            self.mask_buf.shape,
            self.mask_buf.data_ptr(),
        )
        self.io.bind_input(
            "cache_position",
            "cuda",
            self.device_id,
            np.int64,
            self.cache_pos.shape,
            self.cache_pos.data_ptr(),
        )

        for i in range(2 * self.L):
            t = self.past[i]
            self.io.bind_input(f"past_{i}", "cuda", self.device_id, np.float16, t.shape, t.data_ptr())

        self.io.bind_output("logits", "cuda", self.device_id, np.float16, self.out_logits.shape, self.out_logits.data_ptr())
        self.io.bind_output(
            "last_hidden_state",
            "cuda",
            self.device_id,
            np.float16,
            self.out_hidden.shape,
            self.out_hidden.data_ptr(),
        )

        for i in range(2 * self.L):
            t = self.new_kv[i]
            self.io.bind_output(f"new_{i}", "cuda", self.device_id, np.float16, t.shape, t.data_ptr())

        self._all_valid_mask = False
        self.mask_buf.zero_()

    @torch.no_grad()
    def reset(self):
        for t in self.past:
            t.zero_()
        self.mask_buf.zero_()
        self._all_valid_mask = False

    @torch.no_grad()
    def _set_mask_for_pos(self, pos: int):
        pos = int(pos)
        if pos >= self.past_len:
            if not self._all_valid_mask:
                self.mask_buf.zero_()
                self._all_valid_mask = True
            return

        self._all_valid_mask = False
        valid = pos
        self.mask_buf.zero_()
        if valid < self.past_len:
            self.mask_buf[..., valid:self.past_len] = -1e4

    @torch.no_grad()
    def step(self, inputs_embeds: torch.Tensor, pos: int):
        if inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(1)
        self.embed_buf.copy_(inputs_embeds)
        self.cache_pos.fill_(int(pos))
        self._set_mask_for_pos(pos)
        torch.cuda.synchronize()
        if hasattr(self.io, "synchronize_inputs"):
            self.io.synchronize_inputs()
        self.sess.run_with_iobinding(self.io)
        if hasattr(self.io, "synchronize_outputs"):
            self.io.synchronize_outputs()
        torch.cuda.synchronize()
        idx = int(pos) % self.past_len
        for i in range(2 * self.L):
            self.past[i][:, :, idx:idx + 1, :].copy_(self.new_kv[i])

        return self.out_logits, self.out_hidden


@torch.no_grad()
def prefill_backbone_with_onnx(ort_bb: ORTBackbonePastN, inputs_embeds: torch.Tensor):
    seq_len = inputs_embeds.shape[1]
    logits_next = None
    h_last = None
    for i in range(seq_len):
        token_embed = inputs_embeds[:, i:i + 1, :]
        ort_logits, h_last = ort_bb.step(token_embed, pos=i)
        logits_next = ort_logits[:, 0, :]
    return logits_next, h_last, seq_len


@dataclass
class GenerationState:
    """State for a single generation session."""
    ort_bb: ORTBackbonePastN
    ort_dd: ORTDepthDecoder
    h_last: torch.Tensor
    logits_next: torch.Tensor
    seq_len: int
    cur_step: int
    frames: List[torch.Tensor]
    input_queue: List[Dict[str, Any]]
    counter: int                # index of next text chunk to inject
    eos: torch.Tensor
    has_active_text: bool       # True if currently generating for a chunk
    min_steps: int              # Minimum steps before allowing EOS
    last_activity_time: float = field(default_factory=time.time)  # Track last activity


class TTSGenerator:
    """
    Wrapper class for CSM TTS model with stateful generation
    supporting multiple concurrent streams.
    """

    def __init__(
        self,
        model_id: str = "/local_models/csm-1b-base",
        model_path: str = "/local_models/georgian-csm-1b",
        device: str = "cuda",
        temperature: float = 0.8,
        top_p: float = 0.9,
        decoder_temperature: float = 0.8,
        decoder_top_p: float = 0.9,
        compile_model: bool = True,
        reference_audio_path: Optional[str] = None,
        reference_json_path: Optional[str] = None,
        max_concurrent_sessions: int = 64,
        idle_timeout_seconds: float = 300.0,  # 5 minutes default
        min_steps_default: int = 1,
        min_steps_max: Optional[int] = None,
        min_steps_char_threshold: int = 5,
        min_steps_char_bucket: int = 6,
    ):
        _ = compile_model  # kept for API compatibility
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("ONNX TTSGenerator requires a CUDA device.")
        self.device_id = 0 if self.device.index is None else int(self.device.index)
        self.temperature = temperature
        self.top_p = top_p
        self.decoder_temperature = decoder_temperature
        self.decoder_top_p = decoder_top_p
        self.max_concurrent_sessions = max_concurrent_sessions
        self.idle_timeout_seconds = idle_timeout_seconds
        self.min_steps_default = max(0, int(min_steps_default))
        if min_steps_max is None:
            min_steps_max = self.min_steps_default + 2
        self.min_steps_max = max(self.min_steps_default, int(min_steps_max))
        self.min_steps_char_threshold = max(1, int(min_steps_char_threshold))
        self.min_steps_char_bucket = max(1, int(min_steps_char_bucket))

        # Torch settings
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        # Processor + model
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CsmForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cpu",
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        ).eval()

        self.model.embed_text_tokens = self.model.embed_text_tokens.to(
            device=self.device, dtype=torch.float16
        )
        self.model.backbone_model.embed_tokens = self.model.backbone_model.embed_tokens.to(
            device=self.device, dtype=torch.float16
        )
        self.model.codec_model = self.model.codec_model.to(self.device)

        self.text_embed = self.model.embed_text_tokens
        self.audio_embed = self.model.backbone_model.embed_tokens
        self.codec_model = self.model.codec_model

        self.backbone_onnx_path = os.path.join(model_path, BACKBONE_ONNX_FILENAME)
        self.depth_decoder_onnx_path = os.path.join(model_path, DEPTH_DECODER_ONNX_FILENAME)
        if not os.path.exists(self.backbone_onnx_path):
            raise FileNotFoundError(f"Missing backbone ONNX: {self.backbone_onnx_path}")
        if not os.path.exists(self.depth_decoder_onnx_path):
            raise FileNotFoundError(f"Missing depth decoder ONNX: {self.depth_decoder_onnx_path}")
        self.ort_bb_sess = _create_ort_session(self.backbone_onnx_path, self.device_id)
        self.ort_dd_sess = _create_ort_session(self.depth_decoder_onnx_path, self.device_id)

        # Reference
        self.reference_audio = None
        self.reference_transcript = None
        if reference_audio_path and reference_json_path:
            self._load_reference(reference_audio_path, reference_json_path)
        else:
            raise RuntimeError("Reference must be passed (audio + json).")

        # Sessions
        self.sessions: Dict[int, GenerationState] = {}
        self.session_lock = threading.Lock()
        
        # Start idle session cleanup thread
        self._cleanup_thread_stop = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._idle_session_cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def _idle_session_cleanup_loop(self):
        """Background thread to clean up idle sessions."""
        while not self._cleanup_thread_stop.is_set():
            try:
                self._cleanup_idle_sessions()
            except Exception as e:
                print(f"[TTSGenerator] Error in idle cleanup: {e}")
            # Check every 30 seconds
            self._cleanup_thread_stop.wait(30.0)

    def _cleanup_idle_sessions(self):
        """Remove sessions that have been idle for too long."""
        current_time = time.time()
        sessions_to_remove = []
        
        with self.session_lock:
            for session_id, state in self.sessions.items():
                idle_time = current_time - state.last_activity_time
                if idle_time > self.idle_timeout_seconds:
                    sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            print(f"[TTSGenerator] Cleaning up idle session {session_id} (idle for >{self.idle_timeout_seconds}s)")
            self.end_session(session_id)
    
    def cleanup_all_sessions(self):
        """Remove all active sessions."""
        with self.session_lock:
            session_ids = list(self.sessions.keys())
        
        for session_id in session_ids:
            print(f"[TTSGenerator] Cleaning up session {session_id}")
            self.end_session(session_id)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about all active sessions."""
        current_time = time.time()
        info = {
            "active_sessions": [],
            "total_count": 0,
            "max_sessions": self.max_concurrent_sessions,
        }
        
        with self.session_lock:
            info["total_count"] = len(self.sessions)
            for session_id, state in self.sessions.items():
                idle_time = current_time - state.last_activity_time
                info["active_sessions"].append({
                    "session_id": session_id,
                    "idle_seconds": round(idle_time, 1),
                    "frames_generated": len(state.frames),
                    "queue_length": len(state.input_queue),
                    "has_active_text": state.has_active_text,
                })
        
        return info

    def save_model(self, save_dir: str = "saved_csm_model"):
        os.makedirs(save_dir, exist_ok=True)
        print(f"[TTSGenerator] Saving model and processor to: {save_dir}")
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)

    def _load_reference(self, audio_path: str, json_path: str):
        import json

        audio, _ = sf.read(audio_path, dtype="float32")
        audio = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
        with open(json_path) as f:
            audio_transcript = json.load(f)

        self.reference_audio = audio
        self.reference_transcript = audio_transcript.get(Path(audio_path).name)
        if self.reference_transcript is None:
            raise RuntimeError(f"No transcript entry for key '{Path(audio_path).name}' in json.")

    def sample_from_logits(
        self,
        logits: torch.Tensor,
        method: str = "nucleus",
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Sample from logits using different methods."""
        return sample_from_logits(
            logits,
            method=method,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    def _min_steps_for_text(self, text: str) -> int:
        if not text or not text.strip():
            return self.min_steps_default
        letters_count = len(_SPEECH_TEXT_RE.findall(text))
        if letters_count == 0:
            return 0
        extra = max(0, (letters_count - self.min_steps_char_threshold) // self.min_steps_char_bucket)
        return min(self.min_steps_default + extra, self.min_steps_max)

    def append_texts(
        self,
        session_id: int,
        texts: List[str],
        speaker_id: int = 0,
        min_steps_overrides: Optional[List[int]] = None,
    ) -> bool:
        """
        Append additional text segments to an existing session's input queue.

        Each text is turned into a text-only chat turn.
        """
        with self.session_lock:
            if session_id not in self.sessions:
                return False
            # Update activity time
            self.sessions[session_id].last_activity_time = time.time()

        new_inputs: List[Dict[str, Any]] = []
        for idx, text in enumerate(texts):
            if min_steps_overrides and idx < len(min_steps_overrides):
                min_steps = max(0, int(min_steps_overrides[idx]))
            else:
                min_steps = self._min_steps_for_text(text)
            conversation = [
                {
                    "role": f"{speaker_id}",
                    "content": [{"type": "text", "text": text}],
                }
            ]
            cur_input = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
            ).to(self.device)
            new_inputs.append(
                {
                    "input_ids": cur_input["input_ids"],
                    "min_steps": min_steps,
                }
            )

        with self.session_lock:
            if session_id not in self.sessions:
                return False
            self.sessions[session_id].input_queue.extend(new_inputs)
            self.sessions[session_id].last_activity_time = time.time()

        return True

    def initialize_session(self, session_id: int, speaker_id: int = 0) -> bool:
        """
        Initialize a session:
        - runs ONLY the reference (text + audio) through the backbone
        - builds KV cache and h_last
        - no text chunks yet (those come via append_texts)
        """
        with self.session_lock:
            if len(self.sessions) >= self.max_concurrent_sessions:
                return False
            if session_id in self.sessions:
                # overwrite existing
                del self.sessions[session_id]

        if self.reference_audio is None or self.reference_transcript is None:
            raise RuntimeError("Reference audio/transcript not loaded.")

        # Build reference-only conversation
        conversation = [
            {
                "role": f"{speaker_id}",
                "content": [
                    {"type": "text", "text": self.reference_transcript},
                    {"type": "audio", "path": self.reference_audio.numpy()[0]},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(self.device)

        if "input_values" in inputs:
            inputs["input_values"] = inputs["input_values"].to(torch.bfloat16)

        model_inputs = self.model.prepare_inputs_for_generation(**inputs)
        ref_embeds = model_inputs["inputs_embeds"].to(device=self.device, dtype=torch.float16)

        ort_bb = ORTBackbonePastN(
            self.backbone_onnx_path,
            past_len=BACKBONE_PAST_LEN,
            device_id=self.device_id,
            sess=self.ort_bb_sess,
        )
        ort_dd = ORTDepthDecoder(
            self.depth_decoder_onnx_path,
            past_len=DEPTH_DECODER_PAST_LEN,
            device_id=self.device_id,
            sess=self.ort_dd_sess,
        )
        ort_bb.reset()

        logits_next, h_last, seq_len = prefill_backbone_with_onnx(ort_bb, ref_embeds)
        if logits_next is None or h_last is None:
            raise RuntimeError("Reference prefill produced no logits; check reference inputs.")

        eos = torch.tensor(
            [self.model.config.codebook_eos_token_id],
            device=self.device,
            dtype=torch.long,
        )

        state = GenerationState(
            ort_bb=ort_bb,
            ort_dd=ort_dd,
            h_last=h_last,
            logits_next=logits_next,
            seq_len=seq_len,
            cur_step=0,
            frames=[],
            input_queue=[],
            counter=0,               # next text index to inject
            eos=eos,
            has_active_text=False,   # no text chunk yet
            min_steps=0,
        )

        with self.session_lock:
            self.sessions[session_id] = state

        return True

    def step_session(self, 
                     session_id: int, 
                     max_steps: int = 1, 
                     temperature: Optional[float] = None,
                     topp: Optional[float] = None,
                     depth_temperature: Optional[float] = None,
                     depth_topp: Optional[float] = None,
                     ) -> bool:
        """
        Run generation steps for a session.

        - If there is no active text chunk but queued texts exist, inject the next one.
        - Then generate audio frames until EOS or max_steps.

        Returns:
            is_complete
            is_complete == True if no active chunk and no queued texts.
        """
        with self.session_lock:
            if session_id not in self.sessions:
                return True
            state = self.sessions[session_id]
            # Update activity time
            state.last_activity_time = time.time()
            
        steps_taken = 0
        if temperature:
            self.temperature = temperature
        if topp:
            self.top_p = topp
        if depth_temperature:
            self.decoder_temperature = depth_temperature
        if depth_topp:
            self.decoder_top_p = depth_topp

        with torch.no_grad():
            while steps_taken < max_steps:
                # 1) If no active text, try to inject next chunk
                if not state.has_active_text:
                    if state.counter < len(state.input_queue):
                        inject_inputs = state.input_queue[state.counter]
                        inject_embeds = self.text_embed(inject_inputs["input_ids"])
                        for i in range(inject_embeds.shape[1]):
                            token_embed = inject_embeds[:, i:i + 1, :]
                            ort_logits, h_last = state.ort_bb.step(
                                token_embed,
                                pos=state.seq_len,
                            )
                            state.seq_len += 1
                            state.logits_next = ort_logits[:, 0, :]
                            state.h_last = h_last
                        state.has_active_text = True
                        state.cur_step = 0
                        state.min_steps = int(inject_inputs.get("min_steps", 0))
                        state.counter += 1  # consumed this text chunk
                        continue
                    else:
                        # No active text and nothing new to inject
                        break

                # 2) We have an active text chunk: generate one frame
                logits = state.logits_next
                if state.cur_step < state.min_steps:
                    logits = logits.clone()
                    logits[..., int(state.eos.item())] = -1e9
                c0 = self.sample_from_logits(
                    logits,
                    method="nucleus",
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                c0_1d = c0.view(-1).to(torch.long)
                reached_eos = False
                if state.cur_step >= state.min_steps:
                    reached_eos = (c0_1d == state.eos.view(-1)).any()

                if bool(reached_eos):
                    # End of this text chunk; next loop may inject new chunk
                    state.has_active_text = False
                    state.cur_step = 0
                    steps_taken += 1
                    continue

                # Depth decoder for codebooks 1..N
                frame = state.ort_dd.generate_frame(
                    c0=c0_1d,
                    h_last=state.h_last,
                    method="nucleus",
                    temperature=self.decoder_temperature,
                    top_p=self.decoder_top_p,
                )
                state.frames.append(frame)

                # Feed new audio frame back into backbone
                frame_3d = frame.unsqueeze(1)  # [B,1,C]
                audio_embeds = self.audio_embed(frame_3d)
                ort_logits, h_last = state.ort_bb.step(audio_embeds, pos=state.seq_len)
                state.seq_len += 1
                state.logits_next = ort_logits[:, 0, :]
                state.h_last = h_last

                state.cur_step += 1
                steps_taken += 1

        # Complete when no active chunk and no queued texts left
        is_complete = (not state.has_active_text) and (
            state.counter >= len(state.input_queue)
        )
        return is_complete

    def stream_session_audio(
        self,
        session_id: int,
        chunk_len: int,
        max_steps: int,
        temperature: Optional[float] = None,
        topp: Optional[float] = None,
        depth_temperature: Optional[float] = None,
        depth_topp: Optional[float] = None,
    ) -> Generator[np.ndarray, None, None]:
        """
        Yield audio chunks for the queued text in a session.

        This runs the generation loop internally and emits the latest chunk
        after each newly generated frame.

        Assumes exclusive access to the session while streaming.
        """
        if chunk_len <= 0:
            return

        with self.session_lock:
            state = self.sessions.get(session_id)
        if state is None:
            return

        if temperature is not None:
            self.temperature = temperature
        if topp is not None:
            self.top_p = topp
        if depth_temperature is not None:
            self.decoder_temperature = depth_temperature
        if depth_topp is not None:
            self.decoder_top_p = depth_topp

        steps_for_chunk = 0

        with torch.no_grad():
            while True:
                frames = None
                state.last_activity_time = time.time()

                if not state.has_active_text:
                    if state.counter < len(state.input_queue):
                        inject_inputs = state.input_queue[state.counter]
                        inject_embeds = self.text_embed(inject_inputs["input_ids"])
                        for i in range(inject_embeds.shape[1]):
                            token_embed = inject_embeds[:, i:i + 1, :]
                            ort_logits, h_last = state.ort_bb.step(
                                token_embed,
                                pos=state.seq_len,
                            )
                            state.seq_len += 1
                            state.logits_next = ort_logits[:, 0, :]
                            state.h_last = h_last
                        state.has_active_text = True
                        state.cur_step = 0
                        state.min_steps = int(inject_inputs.get("min_steps", 0))
                        state.counter += 1
                        steps_for_chunk = 0
                    else:
                        return
                else:
                    logits = state.logits_next
                    if state.cur_step < state.min_steps:
                        logits = logits.clone()
                        logits[..., int(state.eos.item())] = -1e9
                    c0 = self.sample_from_logits(
                        logits,
                        method="nucleus",
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    c0_1d = c0.view(-1).to(torch.long)
                    reached_eos = False
                    if state.cur_step >= state.min_steps:
                        reached_eos = (c0_1d == state.eos.view(-1)).any()

                    if bool(reached_eos):
                        state.has_active_text = False
                        state.cur_step = 0
                        steps_for_chunk = 0
                    else:
                        frame = state.ort_dd.generate_frame(
                            c0=c0_1d,
                            h_last=state.h_last,
                            method="nucleus",
                            temperature=self.decoder_temperature,
                            top_p=self.decoder_top_p,
                        )
                        state.frames.append(frame)

                        frame_3d = frame.unsqueeze(1)  # [B,1,C]
                        audio_embeds = self.audio_embed(frame_3d)
                        ort_logits, h_last = state.ort_bb.step(audio_embeds, pos=state.seq_len)
                        state.seq_len += 1
                        state.logits_next = ort_logits[:, 0, :]
                        state.h_last = h_last

                        state.cur_step += 1
                        steps_for_chunk += 1

                        frames = state.frames

                if (not state.has_active_text) and (state.counter >= len(state.input_queue)):
                    return

                if frames is None:
                    continue

                audio_codes = torch.cat(frames[-2:], dim=0)
                audio_24k = self._decode_audio(audio_codes)
                if audio_24k is None:
                    continue
                audio_np = audio_24k.detach().cpu().float().numpy()
                if audio_np.size == 0:
                    continue

                chunk_np = audio_np[-chunk_len:].astype(np.float32)
                yield chunk_np

                if max_steps > 0 and steps_for_chunk >= max_steps and state.has_active_text:
                    state.has_active_text = False
                    state.cur_step = 0
                    steps_for_chunk = 0

    def increment_state_counter(self, session_id: int) -> bool:
        """
        Force the current text chunk to finish and allow the next one to inject.
        Useful when max_steps stops generation without EOS.
        """
        with self.session_lock:
            if session_id not in self.sessions:
                return False
            state = self.sessions[session_id]
            state.has_active_text = False
            state.cur_step = 0
            state.last_activity_time = time.time()
        return True

    def _decode_audio(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Decode audio codes to waveform."""
        audio_codes = audio_codes.unsqueeze(0).to(self.device)

        with torch.no_grad():
            audio_list = []
            for audio_codes_batch in audio_codes:
                eos_idxs = (
                    (audio_codes_batch == self.model.config.codebook_eos_token_id)
                    .all(dim=-1)
                    .nonzero()
                )
                if eos_idxs.numel() != 0:
                    cutoff_idx = eos_idxs.min()
                else:
                    cutoff_idx = audio_codes_batch.shape[0]
                audio_codes_batch = audio_codes_batch[:cutoff_idx]

                codec_decode_output = self.codec_model.decode(
                    audio_codes_batch.transpose(0, 1).unsqueeze(0)
                )
                audio_list.append(codec_decode_output.audio_values[0, 0])

        return audio_list[0] if audio_list else None

    def get_session_audio(
        self, session_id: int, return_codes: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Get all generated audio (codes or waveform) for a session so far.
        """
        with self.session_lock:
            if session_id not in self.sessions:
                return None
            state = self.sessions[session_id]
            if not state.frames:
                return None
            audio_codes = torch.cat(state.frames, dim=0)  # [T, C]

        if return_codes:
            return audio_codes
        else:
            return self._decode_audio(audio_codes)

    def end_session(self, session_id: int) -> Optional[torch.Tensor]:
        """
        End a session and return final decoded audio (full).
        """
        audio = self.get_session_audio(session_id, return_codes=False)
        with self.session_lock:
            state = self.sessions.pop(session_id, None)
        if state is not None:
            state.ort_bb.reset()
            state.ort_dd.reset()
        return audio

    def get_active_sessions(self) -> List[int]:
        with self.session_lock:
            return list(self.sessions.keys())


@dataclass
class TTSConfig:
    temperature: float = 0.01
    top_p: float = 0.999
    decoder_temperature: float = 0.01
    decoder_top_p: float = 0.999
    model_id: str = "local_models/tts_model/csm-1b-base"
    model_path: str = "local_models/tts_model/georgian-csm-1b"
    device: str = "cuda"
    max_steps: int = 125 // 2
    min_steps: int = 1
    min_steps_max: Optional[int] = None
    min_steps_char_threshold: int = 5
    min_steps_char_bucket: int = 6


def _split_text_for_streaming(text: str) -> List[str]:
    """
    Split text for TTS streaming with 2-word lookahead.

    Example: "გამარჯობა! როგორ შემიძლია დაგეხმაროთ დღეს?"
    Returns: ["გამარჯობა! როგორ შემიძლია", " დაგეხმაროთ", " დღეს?", "", ""]
    """
    text = text.replace("\n", " ").strip()
    words = text.split()

    result = [" ".join(words[:3])]
    for w in words[3:]:
        result.append(" " + w)
    result.extend(["", ""])
    return result


if __name__ == "__main__":
    config = TTSConfig()

    tts_generator = TTSGenerator(
        model_id=config.model_id,
        model_path=config.model_path,
        device=config.device,
        temperature=config.temperature,
        top_p=config.top_p,
        decoder_temperature=config.decoder_temperature,
        decoder_top_p=config.decoder_top_p,
        compile_model=True,
        reference_audio_path="local_models/tts_model/georgian-csm-1b/context_audio_for_inference.wav",
        reference_json_path="local_models/tts_model/georgian-csm-1b/context_text_for_inference.json",
        min_steps_default=config.min_steps,
        min_steps_max=config.min_steps_max,
        min_steps_char_threshold=config.min_steps_char_threshold,
        min_steps_char_bucket=config.min_steps_char_bucket,
    )

    assistant_texts = [
        "გამარჯობა! როგორ შემიძლია დაგეხმაროთ?",
        "თუ გსურთ ბალანსის შემოწმება, გთხოვთ მითხრათ.",
        "დავიწყოთ გადარიცხვა — მიუთითეთ თანხა და მიმღები.",
        "თქვენი მოთხოვნა მიღებულია და მალე დასრულდება.",
        "სხვა რამით ხომ არ დაგეხმაროთ?",
    ]

    output_sample_rate = 24_000
    stream_chunk_duration = 1920.0 / float(output_sample_rate)
    chunk_len = int(stream_chunk_duration * float(output_sample_rate))

    total_audio_samples = 0
    total_gen_time = 0.0

    for turn_idx, text in enumerate(assistant_texts, start=1):
        seq_id = 100 + turn_idx
        all_chunks = _split_text_for_streaming(text)
        words = text.replace("\n", " ").strip().split()

        ok = tts_generator.initialize_session(seq_id, speaker_id=0)
        if not ok:
            raise RuntimeError(f"Failed to initialize TTS session {seq_id}.")

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        first_chunk_time = None

        print(f"\n[Turn {turn_idx}] {text}")
        print(
            f"[Turn {turn_idx}] min_steps={config.min_steps} max={tts_generator.min_steps_max} "
            f"chars>={config.min_steps_char_threshold} "
            f"bucket={config.min_steps_char_bucket} "
            f"words={len(words)} chunks={len(all_chunks)}"
        )

        # NEW: collect streamed chunks here (each is length chunk_len)
        streamed_chunks: List[np.ndarray] = []

        for chunk_idx, chunk in enumerate(all_chunks, start=1):
            word_idx = chunk_idx - 1
            if word_idx < len(words):
                word_label = words[word_idx]
                word_min_steps = tts_generator._min_steps_for_text(word_label)
            else:
                word_label = "<empty>"
                word_min_steps = tts_generator.min_steps_default

            appended = tts_generator.append_texts(
                seq_id,
                [chunk],
                speaker_id=0,
                min_steps_overrides=[word_min_steps],
            )
            if not appended:
                raise RuntimeError(f"Failed to append text for session {seq_id}.")

            chunk_samples = 0
            for out_chunk in tts_generator.stream_session_audio(
                session_id=seq_id,
                chunk_len=chunk_len,
                max_steps=config.max_steps,
                temperature=config.temperature,
                topp=config.top_p,
                depth_temperature=config.decoder_temperature,
                depth_topp=config.decoder_top_p,
            ):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter() - start_time

                # NEW: store the chunk
                streamed_chunks.append(out_chunk)
                chunk_samples += out_chunk.shape[-1]

            chunk_ms = (chunk_samples / output_sample_rate) * 1000.0
            print(
                f"[Turn {turn_idx}] Word {chunk_idx}/{len(words)} '{word_label}': "
                f"min_steps={word_min_steps} {chunk_ms:.1f} ms ({chunk_samples} samples)"
            )

        torch.cuda.synchronize()
        gen_time = time.perf_counter() - start_time

        # NEW: assemble final audio from streamed chunks
        if not streamed_chunks:
            raise RuntimeError(f"No audio chunks generated for session {seq_id}.")

        final_audio_np = np.concatenate(streamed_chunks, axis=0).astype(np.float32)
        final_audio = torch.from_numpy(final_audio_np)  # 1D CPU tensor

        # Cleanup session state (NOTE: your end_session decodes full audio again, wasteful)
        _ = tts_generator.end_session(seq_id)  # ignore returned audio, just cleanup

        audio_samples = final_audio.shape[-1]
        audio_dur = audio_samples / output_sample_rate
        rtf = gen_time / max(audio_dur, 1e-9)

        total_audio_samples += audio_samples
        total_gen_time += gen_time

        first_chunk_ms = 0.0 if first_chunk_time is None else first_chunk_time * 1000.0
        print(f"[Turn {turn_idx}] First output latency: {first_chunk_ms:.1f} ms")
        print(f"[Turn {turn_idx}] Generation time: {gen_time:.3f} s")
        print(f"[Turn {turn_idx}] Audio duration: {audio_dur:.3f} s")
        print(f"[Turn {turn_idx}] RTF: {rtf:.3f}")

        out_path = f"standalone_turn_{turn_idx}.wav"
        sf.write(out_path, final_audio_np, output_sample_rate)
        print(f"[Turn {turn_idx}] Saved {out_path}")

    total_audio_dur = total_audio_samples / output_sample_rate
    total_rtf = total_gen_time / max(total_audio_dur, 1e-9)
    print("\n[Total] Generation time: {:.3f} s".format(total_gen_time))
    print("[Total] Audio duration: {:.3f} s".format(total_audio_dur))
    print("[Total] RTF: {:.3f}".format(total_rtf))
