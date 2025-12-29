import os
import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import torch
import torch.nn.functional as F
import soundfile as sf
from transformers import CsmForConditionalGeneration, AutoProcessor
from transformers.cache_utils import StaticCache

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


@dataclass
class PerfStats:
    bb_inject_ms: float = 0.0
    bb_audio_ms: float = 0.0
    dd_ms: float = 0.0
    lm_ms: float = 0.0
    codec_ms: float = 0.0
    inject_calls: int = 0
    audio_calls: int = 0
    dd_calls: int = 0
    lm_calls: int = 0
    codec_calls: int = 0


@dataclass
class GenerationState:
    """State for a single generation session."""
    bb_pkv: StaticCache
    dd_pkv: StaticCache
    attn_mask: torch.Tensor
    h_last: torch.Tensor
    seq_len: int
    cur_step: int
    frames: List[torch.Tensor]
    input_queue: List[Dict[str, Any]]
    counter: int                # index of next text chunk to inject
    eos: torch.Tensor
    cache_pos: torch.Tensor
    has_active_text: bool       # True if currently generating for a chunk
    perf: PerfStats = field(default_factory=PerfStats)
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
        max_concurrent_sessions: int = 4,
        idle_timeout_seconds: float = 300.0,  # 5 minutes default
    ):
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.decoder_temperature = decoder_temperature
        self.decoder_top_p = decoder_top_p
        self.max_concurrent_sessions = max_concurrent_sessions
        self.idle_timeout_seconds = idle_timeout_seconds
        self.enable_timing = os.environ.get("TTS_TIMING", "1") != "0"
        self.timing_sync_cuda = os.environ.get("TTS_TIMING_SYNC", "0") == "1"

        # Torch settings
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        # Processor + model
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CsmForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device,
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        self.model.eval()

        if compile_model:
            self.model = torch.compile(
                self.model,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
                backend="inductor",
            )

        self.model.config.use_cache = True
        self.model.backbone_model.config.use_cache = True

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
        self._session_pool: List[GenerationState] = []
        self._reference_speaker_id = 0
        self._bb_cache_len = 4096
        self._dd_cache_len = 32
        self._reference_bb_pkv: Optional[StaticCache] = None
        self._reference_attn_mask: Optional[torch.Tensor] = None
        self._reference_h_last: Optional[torch.Tensor] = None
        self._reference_seq_len: int = 0
        self._reference_eos: Optional[torch.Tensor] = None
        self._reference_cache_pos: Optional[torch.Tensor] = None
        
        # Start idle session cleanup thread
        self._cleanup_thread_stop = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._idle_session_cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        # Build reference cache once for all sessions
        self._build_reference_state(self._reference_speaker_id)

    def _maybe_sync(self):
        if self.enable_timing and self.timing_sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _record_perf(self, state: GenerationState, key: str, delta_ms: float, count_key: Optional[str] = None):
        if not self.enable_timing:
            return
        setattr(state.perf, key, getattr(state.perf, key) + delta_ms)
        if count_key:
            setattr(state.perf, count_key, getattr(state.perf, count_key) + 1)

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
            self._session_pool.clear()
        
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

    def get_session_perf(self, session_id: int, reset: bool = False) -> Optional[PerfStats]:
        with self.session_lock:
            state = self.sessions.get(session_id)
            if state is None:
                return None
            perf = state.perf
            if reset:
                state.perf = PerfStats()
            return perf

    def record_codec_time(self, session_id: int, delta_ms: float):
        if not self.enable_timing:
            return
        with self.session_lock:
            state = self.sessions.get(session_id)
            if state is None:
                return
            state.perf.codec_ms += delta_ms
            state.perf.codec_calls += 1

    def save_model(self, save_dir: str = "saved_csm_model"):
        os.makedirs(save_dir, exist_ok=True)
        print(f"[TTSGenerator] Saving model and processor to: {save_dir}")
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)

    def _load_reference(self, audio_path: str, json_path: str):
        import json

        audio, _ = sf.read(audio_path)
        audio = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
        with open(json_path) as f:
            audio_transcript = json.load(f)

        self.reference_audio = audio
        self.reference_transcript = audio_transcript.get(Path(audio_path).name)
        if self.reference_transcript is None:
            raise RuntimeError(f"No transcript entry for key '{Path(audio_path).name}' in json.")

    def _build_reference_state(self, speaker_id: int = 0) -> None:
        if self.reference_audio is None or self.reference_transcript is None:
            raise RuntimeError("Reference audio/transcript not loaded.")

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
        ref_embeds = model_inputs["inputs_embeds"]
        attn_mask = inputs["attention_mask"].clone()

        bb_pkv = StaticCache(
            config=self.model.config,
            batch_size=1,
            max_cache_len=self._bb_cache_len,
            device=self.model.device,
            dtype=self.model.dtype,
        )
        pos_ids = torch.arange(ref_embeds.shape[1], device=self.device).unsqueeze(0)

        with torch.no_grad():
            bb_out = self.model.backbone_model(
                inputs_embeds=ref_embeds,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                past_key_values=bb_pkv,
                use_cache=True,
                output_hidden_states=True,
            )

        h_last = bb_out.hidden_states[-1][:, -1, :]
        seq_len = ref_embeds.shape[1]

        eos = torch.tensor(
            [self.model.config.codebook_eos_token_id],
            device=self.model.device,
            dtype=torch.long,
        )
        cache_pos = torch.arange(0, 2, device=self.device)

        self._reference_bb_pkv = bb_out.past_key_values
        self._reference_attn_mask = attn_mask
        self._reference_h_last = h_last
        self._reference_seq_len = seq_len
        self._reference_eos = eos
        self._reference_cache_pos = cache_pos
        self._reference_speaker_id = speaker_id

    def _new_bb_cache(self) -> StaticCache:
        return StaticCache(
            config=self.model.config,
            batch_size=1,
            max_cache_len=self._bb_cache_len,
            device=self.model.device,
            dtype=self.model.dtype,
        )

    def _new_dd_cache(self) -> StaticCache:
        return StaticCache(
            config=self.model.depth_decoder.config,
            batch_size=1,
            max_cache_len=self._dd_cache_len,
            device=self.model.device,
            dtype=self.model.dtype,
        )

    def _copy_reference_cache(self, target: StaticCache) -> StaticCache:
        if self._reference_bb_pkv is None:
            raise RuntimeError("Reference cache not initialized.")
        src = self._reference_bb_pkv

        if (
            hasattr(src, "key_cache")
            and hasattr(src, "value_cache")
            and hasattr(target, "key_cache")
            and hasattr(target, "value_cache")
        ):
            for layer_idx in range(len(src.key_cache)):
                target.key_cache[layer_idx].copy_(src.key_cache[layer_idx])
                target.value_cache[layer_idx].copy_(src.value_cache[layer_idx])
            return target

        if hasattr(src, "layers") and hasattr(target, "layers"):
            copied_layers = 0
            for layer_idx, src_layer in enumerate(src.layers):
                if layer_idx >= len(target.layers):
                    break
                tgt_layer = target.layers[layer_idx]
                if not hasattr(src_layer, "keys") or not hasattr(src_layer, "values"):
                    break
                if getattr(tgt_layer, "is_initialized", True) is False:
                    if hasattr(tgt_layer, "lazy_initialization"):
                        tgt_layer.lazy_initialization(src_layer.keys)
                if hasattr(tgt_layer, "keys") and hasattr(tgt_layer, "values"):
                    tgt_layer.keys.copy_(src_layer.keys)
                    tgt_layer.values.copy_(src_layer.values)
                    copied_layers += 1
            if copied_layers == len(src.layers):
                return target

        return copy.deepcopy(src)

    def _reset_state_from_reference(self, state: GenerationState) -> None:
        if self._reference_attn_mask is None or self._reference_h_last is None:
            raise RuntimeError("Reference state not initialized.")
        state.bb_pkv = self._copy_reference_cache(state.bb_pkv)
        state.dd_pkv.reset()
        state.attn_mask = self._reference_attn_mask.clone()
        state.h_last = self._reference_h_last.clone()
        state.seq_len = self._reference_seq_len
        state.cur_step = 0
        state.frames = []
        state.input_queue = []
        state.counter = 0
        state.has_active_text = False
        state.perf = PerfStats()
        state.last_activity_time = time.time()

        if self._reference_eos is not None:
            state.eos = self._reference_eos
        if self._reference_cache_pos is not None:
            state.cache_pos = self._reference_cache_pos

    def _acquire_state(self) -> GenerationState:
        if self._session_pool:
            state = self._session_pool.pop()
        else:
            state = GenerationState(
                bb_pkv=self._new_bb_cache(),
                dd_pkv=self._new_dd_cache(),
                attn_mask=torch.zeros((1, 1), device=self.device),
                h_last=torch.zeros((1, self.model.config.hidden_size), device=self.device),
                seq_len=0,
                cur_step=0,
                frames=[],
                input_queue=[],
                counter=0,
                eos=self._reference_eos if self._reference_eos is not None else torch.zeros(1, device=self.device, dtype=torch.long),
                cache_pos=self._reference_cache_pos if self._reference_cache_pos is not None else torch.arange(0, 2, device=self.device),
                has_active_text=False,
            )
        self._reset_state_from_reference(state)
        return state

    def _release_state(self, state: GenerationState) -> None:
        state.frames = []
        state.input_queue = []
        state.has_active_text = False
        state.counter = 0
        state.cur_step = 0
        state.perf = PerfStats()
        state.last_activity_time = time.time()
        if len(self._session_pool) < self.max_concurrent_sessions:
            self._session_pool.append(state)

    def warmup(
        self,
        speaker_id: int = 0,
        text: Optional[str] = None,
        max_steps: int = 12,
        max_rounds: int = 2,
    ) -> bool:
        session_id = -1
        if session_id in self.sessions:
            return False
        if not self.initialize_session(session_id, speaker_id=speaker_id):
            return False
        try:
            if text is None:
                text = "გამარჯობა, მე ვარ თიბისი ბანკის ციფრული ასისტენტი."
            chunks = _split_text_for_streaming(text)
            self.append_texts(session_id, chunks, speaker_id=speaker_id)
            for _ in range(max(1, max_rounds)):
                is_complete = self.step_session(session_id, max_steps=max_steps)
                if is_complete:
                    break
            codes = self.get_session_audio(session_id, return_codes=True)
            if codes is not None and codes.shape[0] > 0:
                tail = min(2, codes.shape[0])
                _ = self._decode_audio(codes[-tail:])
            return True
        finally:
            self.end_session(session_id)

    def sample_from_logits(
        self,
        logits: torch.Tensor,
        method: str = "nucleus",
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Sample from logits using different methods."""
        if method == "greedy":
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

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

    def append_texts(self, session_id: int, texts: List[str], speaker_id: int = 0) -> bool:
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
        for text in texts:
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
            new_inputs.append(cur_input)

        with self.session_lock:
            if session_id not in self.sessions:
                return False
            self.sessions[session_id].input_queue.extend(new_inputs)
            self.sessions[session_id].last_activity_time = time.time()

        return True

    def initialize_session(self, session_id: int, speaker_id: int = 0) -> bool:
        """
        Initialize a session from the cached reference state.
        """
        with self.session_lock:
            if len(self.sessions) >= self.max_concurrent_sessions:
                return False
            if session_id in self.sessions:
                del self.sessions[session_id]

        if speaker_id != self._reference_speaker_id:
            print(f"[TTSGenerator] Speaker ID {speaker_id} differs from cached reference; rebuilding reference cache.")
            self._build_reference_state(speaker_id)
            self._session_pool.clear()

        state = self._acquire_state()
        with self.session_lock:
            self.sessions[session_id] = state

        return True

    def _step_backbone_with_cache(
        self,
        new_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
        pkv: StaticCache,
        seq_len: int,
    ):
        """Run backbone on new tokens with cache."""
        bsz, n_new, _ = new_embeds.shape
        new_pos = (
            torch.arange(seq_len, seq_len + n_new, device=new_embeds.device)
            .unsqueeze(0)
            .expand(bsz, n_new)
        )
        new_mask = torch.ones(
            bsz, n_new, device=attn_mask.device, dtype=attn_mask.dtype
        )
        attn_mask = torch.cat([attn_mask, new_mask], dim=1)

        out = self.model.backbone_model(
            inputs_embeds=new_embeds,
            attention_mask=attn_mask,
            position_ids=new_pos,
            past_key_values=pkv,
            use_cache=True,
            output_hidden_states=True,
        )
        pkv = out.past_key_values
        h_last = out.hidden_states[-1][:, -1, :]
        seq_len += n_new
        return h_last, pkv, seq_len, attn_mask

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
                        inject_embeds = self.model.embed_text_tokens(
                            inject_inputs["input_ids"]
                        )
                        if self.enable_timing:
                            self._maybe_sync()
                            t0 = time.perf_counter()
                        (
                            state.h_last,
                            state.bb_pkv,
                            state.seq_len,
                            state.attn_mask,
                        ) = self._step_backbone_with_cache(
                            inject_embeds,
                            state.attn_mask,
                            state.bb_pkv,
                            state.seq_len,
                        )
                        if self.enable_timing:
                            self._maybe_sync()
                            self._record_perf(
                                state,
                                "bb_inject_ms",
                                (time.perf_counter() - t0) * 1000.0,
                                "inject_calls",
                            )
                        state.has_active_text = True
                        state.cur_step = 0
                        state.counter += 1  # consumed this text chunk
                        continue
                    else:
                        # No active text and nothing new to inject
                        break

                # 2) We have an active text chunk: generate one frame
                if self.enable_timing:
                    self._maybe_sync()
                    t0 = time.perf_counter()
                logits0 = self.model.lm_head(state.h_last.unsqueeze(1))[:, -1:, :]
                c0 = self.sample_from_logits(
                    logits0[:, -1, :],
                    method="nucleus",
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                if self.enable_timing:
                    self._maybe_sync()
                    self._record_perf(
                        state,
                        "lm_ms",
                        (time.perf_counter() - t0) * 1000.0,
                        "lm_calls",
                    )

                reached_eos = (c0 == state.eos).all()

                if bool(reached_eos):
                    # End of this text chunk; next loop may inject new chunk
                    state.has_active_text = False
                    state.cur_step = 0
                    steps_taken += 1
                    continue

                # Depth decoder for codebooks 1..N
                depth_prompt = torch.nn.functional.pad(c0, (1, 0), value=0)
                if self.enable_timing:
                    self._maybe_sync()
                    t0 = time.perf_counter()
                dd_out = self.model.depth_decoder.generate(
                    input_ids=depth_prompt,
                    backbone_last_hidden_state=state.h_last.clone(),
                    max_new_tokens=self.model.config.num_codebooks - 1,
                    temperature=self.decoder_temperature,
                    top_p=self.decoder_top_p,
                    cache_position=state.cache_pos,
                    logits_to_keep=1,
                    use_cache=True,
                    past_key_values=state.dd_pkv,
                    return_dict_in_generate=False,
                )
                if self.enable_timing:
                    self._maybe_sync()
                    self._record_perf(
                        state,
                        "dd_ms",
                        (time.perf_counter() - t0) * 1000.0,
                        "dd_calls",
                    )
                frame = dd_out[:, 1:]  # [B, num_codebooks]
                state.frames.append(frame)

                # Feed new audio frame back into backbone
                frame_3d = frame.unsqueeze(1)  # [B,1,C]
                audio_embeds = self.model.backbone_model.embed_tokens(frame_3d)
                if self.enable_timing:
                    self._maybe_sync()
                    t0 = time.perf_counter()
                (
                    state.h_last,
                    state.bb_pkv,
                    state.seq_len,
                    state.attn_mask,
                ) = self._step_backbone_with_cache(
                    audio_embeds,
                    state.attn_mask,
                    state.bb_pkv,
                    state.seq_len,
                )
                if self.enable_timing:
                    self._maybe_sync()
                    self._record_perf(
                        state,
                        "bb_audio_ms",
                        (time.perf_counter() - t0) * 1000.0,
                        "audio_calls",
                    )

                state.cur_step += 1
                steps_taken += 1

        # Complete when no active chunk and no queued texts left
        is_complete = (not state.has_active_text) and (
            state.counter >= len(state.input_queue)
        )
        return is_complete

    def _decode_audio(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Decode audio codes to waveform."""
        audio_codes = audio_codes.unsqueeze(0).to(self.model.device)

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

                codec_decode_output = self.model.codec_model.decode(
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
            self._release_state(state)
        return audio

    def get_active_sessions(self) -> List[int]:
        with self.session_lock:
            return list(self.sessions.keys())


@dataclass
class TTSConfig:
    temperature: float = 0.8
    top_p: float = 0.9
    decoder_temperature: float = 0.8
    decoder_top_p: float = 0.9
    model_id: str = "local_models/tts_model/csm-1b-base"
    model_path: str = "local_models/tts_model/georgian-csm-1b"
    device: str = "cuda"
    max_steps: int = 125 // 2


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
        reference_audio_path="context_audio_for_inference.wav",
        reference_json_path="context_text_for_inference.json",
    )

    text = "გამარჯობა, რით შემიძლია დაგეხმაროთ?"
    all_chunks = _split_text_for_streaming(text)

    seq_id = 123

    # 1) Initialize session using ONLY reference (text + audio)
    ok = tts_generator.initialize_session(seq_id, speaker_id=0)
    if not ok:
        raise RuntimeError("Failed to initialize TTS session.")

    # 2) Add all text chunks via append_texts
    appended = tts_generator.append_texts(seq_id, all_chunks, speaker_id=0)
    if not appended:
        raise RuntimeError("Failed to append texts to session.")

    # 3) Stream generation
    all_audio_chunks: List[torch.Tensor] = []
    prev_num_frames = 0

    while True:
        is_complete, steps_taken = tts_generator.step_session(
            seq_id,
            max_steps=10,
        )

        codes = tts_generator.get_session_audio(seq_id, return_codes=True)
        if codes is not None:
            cur_num_frames = codes.shape[0]
            if cur_num_frames > prev_num_frames:
                delta_codes = codes[prev_num_frames:cur_num_frames]
                delta_audio = tts_generator._decode_audio(delta_codes)
                all_audio_chunks.append(delta_audio.cpu())
                prev_num_frames = cur_num_frames

        if is_complete:
            break
        # Optional: if steps_taken == 0, you may sleep or wait for new text.

    if not all_audio_chunks:
        raise RuntimeError("No audio was generated.")

    full_audio = torch.cat(all_audio_chunks, dim=-1)

    final_audio = tts_generator.end_session(seq_id)

    print(f"Streamed audio length: {full_audio.shape[-1]} samples")

    sr = 24_000
    sf.write("streamed_output.wav", full_audio.to(torch.float32).numpy(), sr)
    print("Saved streamed_output.wav")
