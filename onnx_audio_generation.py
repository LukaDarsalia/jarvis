#!/usr/bin/env python3
# generate_with_depth_decoder_onnx.py
#
# Uses:
#   - Backbone + codec in PyTorch
#   - Depth decoder one-step in ONNXRuntime (fixed past_len=31, k_len=32)
#
# Export expected: csm_depth_decoder_step_past31.onnx

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import onnxruntime as ort

from transformers import CsmForConditionalGeneration, AutoProcessor
from transformers.cache_utils import StaticCache

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# -------------------------
# Sampling
# -------------------------
def sample_from_logits(logits, method="greedy", temperature=1.0, top_k=0, top_p=1.0):
    # logits: [B, V]
    if method == "greedy":
        return torch.argmax(logits, dim=-1, keepdim=True)

    temperature = max(float(temperature), 1e-8)
    probs = F.softmax(logits / temperature, dim=-1)

    if method == "multinomial":
        return torch.multinomial(probs, num_samples=1)

    if method == "topk":
        k = int(top_k)
        topk_probs, topk_idx = torch.topk(probs, k=k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(topk_probs, num_samples=1)
        return topk_idx.gather(-1, sampled)

    if method == "nucleus":
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumprobs = sorted_probs.cumsum(dim=-1)
        mask = cumprobs <= float(top_p)
        mask[..., 0] = True
        filtered_probs = sorted_probs * mask
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(filtered_probs, num_samples=1)
        return sorted_idx.gather(-1, sampled)

    raise ValueError(f"Unknown method: {method}")

# -------------------------
# Decode audio from frames
# -------------------------
@torch.no_grad()
def generate_audio_from_frames(model, frames_list):
    x = torch.stack(frames_list, dim=1)  # [B,T,32]
    x = x.to(device=model.device, dtype=torch.long)

    eos_id = int(model.config.codebook_eos_token_id)
    eos_mask = (x == eos_id).all(dim=-1)  # [B,T]

    audios = []
    for b in range(x.shape[0]):
        if eos_mask[b].any():
            cutoff = int(torch.nonzero(eos_mask[b], as_tuple=False)[0].item())
        else:
            cutoff = x.shape[1]
        xb = x[b, :cutoff, :]                      # [T,32]
        codec_in = xb.transpose(0, 1).unsqueeze(0)  # [1,32,T]
        out = model.codec_model.decode(codec_in)
        audios.append(out.audio_values[0, 0])
    return audios

# -------------------------
# Backbone cached step
# -------------------------
@torch.no_grad()
def step_backbone_with_cache(model, new_embeds, attn_mask, pkv, seq_len):
    bsz, n_new, _ = new_embeds.shape
    new_pos = (
        torch.arange(seq_len, seq_len + n_new, device=new_embeds.device)
        .unsqueeze(0)
        .expand(bsz, n_new)
    )
    new_mask = torch.ones(bsz, n_new, device=attn_mask.device, dtype=attn_mask.dtype)
    attn_mask = torch.cat([attn_mask, new_mask], dim=1)

    out = model.backbone_model(
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

# -------------------------
# ORT Depth Decoder runner (past_len=31 fixed)
# -------------------------
class ORTDepthDecoderPast31:
    def __init__(self, onnx_path: str, device_id: int = 0):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # IMPORTANT: make ORT use Torch current stream
        torch_stream = torch.cuda.current_stream(device_id).cuda_stream

        self.sess = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=[
                ("CUDAExecutionProvider", {
                    "device_id": device_id,
                    "user_compute_stream": torch_stream,  # <--- fixes stream hazard
                }),
                # Optional: keep CPU EP as fallback, but CUDA should be first
                "CPUExecutionProvider",
            ],
        )

        ins = self.sess.get_inputs()
        outs = self.sess.get_outputs()

        self.L = (len(ins) - 4) // 2
        past0 = ins[4].shape
        B, KVH, P, HD = map(int, past0)
        assert P == 31

        logits_shape = outs[0].shape
        B2, S2, V = map(int, logits_shape)
        assert B2 == B and S2 == 1

        bb_shape = ins[1].shape
        B3, Hbb = map(int, bb_shape)
        assert B3 == B

        self.B, self.KVH, self.HD, self.V, self.Hbb = B, KVH, HD, V, Hbb

        self.token_buf = torch.zeros((B, 1), device="cuda", dtype=torch.long)
        self.bb_buf = torch.zeros((B, Hbb), device="cuda", dtype=torch.float16)
        self.cache_pos = torch.zeros((1,), device="cuda", dtype=torch.long)
        self.mask_buf = torch.zeros((B, 1, 1, 32), device="cuda", dtype=torch.float16)

        self.past = [torch.zeros((B, KVH, 31, HD), device="cuda", dtype=torch.float16) for _ in range(2 * self.L)]
        self.present = [torch.empty((B, KVH, 32, HD), device="cuda", dtype=torch.float16) for _ in range(2 * self.L)]
        self.out_logits = torch.empty((B, 1, V), device="cuda", dtype=torch.float16)

        self.io = self.sess.io_binding()
        self.io.bind_input("input_ids", "cuda", 0, np.int64, self.token_buf.shape, self.token_buf.data_ptr())
        self.io.bind_input("backbone_last_hidden_state", "cuda", 0, np.float16, self.bb_buf.shape, self.bb_buf.data_ptr())
        self.io.bind_input("attention_mask", "cuda", 0, np.float16, self.mask_buf.shape, self.mask_buf.data_ptr())
        self.io.bind_input("cache_position", "cuda", 0, np.int64, self.cache_pos.shape, self.cache_pos.data_ptr())

        for i in range(2 * self.L):
            t = self.past[i]
            self.io.bind_input(f"past_{i}", "cuda", 0, np.float16, t.shape, t.data_ptr())

        self.io.bind_output("logits", "cuda", 0, np.float16, self.out_logits.shape, self.out_logits.data_ptr())
        for i in range(2 * self.L):
            t = self.present[i]
            self.io.bind_output(f"present_{i}", "cuda", 0, np.float16, t.shape, t.data_ptr())

    @torch.no_grad()
    def _set_mask_for_pos(self, pos: int):
        start = 31 - int(pos)
        self.mask_buf.zero_()
        if start > 0:
            self.mask_buf[..., :start] = -1e4

    @torch.no_grad()
    def reset(self):
        for t in self.past:
            t.zero_()

    @torch.no_grad()
    def step(self, token_id: torch.Tensor, h_last_bf16: torch.Tensor, pos: int) -> torch.Tensor:
        if token_id.dim() == 1:
            token_id = token_id.view(self.B, 1)
        self.token_buf.copy_(token_id.to(torch.long))
        self.bb_buf.copy_(h_last_bf16.to(torch.float16))
        self.cache_pos.fill_(int(pos))
        self._set_mask_for_pos(pos)

        # If io_binding has explicit sync methods, call them.
        # With user_compute_stream set, this usually becomes unnecessary, but it's safe.
        if hasattr(self.io, "synchronize_inputs"):
            self.io.synchronize_inputs()

        self.sess.run_with_iobinding(self.io)

        if hasattr(self.io, "synchronize_outputs"):
            self.io.synchronize_outputs()

        for i in range(2 * self.L):
            self.past[i].copy_(self.present[i][:, :, 1:, :])

        return self.out_logits[:, 0, :]

    @torch.no_grad()
    def generate_frame(
        self,
        c0: torch.Tensor,          # [B] or [B,1]
        h_last: torch.Tensor,      # [B,Hbb] bf16
        method="nucleus",
        temperature=0.1,
        top_p=0.999,
    ) -> torch.Tensor:
        """
        Returns frame [B,32] = [c0,c1..c31].
        Resets decoder state per frame (matches HF generate per frame).
        """
        B = self.B
        if c0.dim() == 2:
            c0 = c0[:, 0]
        c0 = c0.to(torch.long).view(B, 1)

        self.reset()

        # pos=0 dummy token (embedding replaced by backbone_last_hidden_state inside wrapper logic)
        dummy = torch.zeros((B, 1), device="cuda", dtype=torch.long)
        _ = self.step(dummy, h_last, pos=0)

        # pos=1 consume c0 -> predict c1
        logits = self.step(c0, h_last, pos=1)
        c1 = sample_from_logits(logits, method=method, temperature=temperature, top_p=top_p)  # [B,1]

        outs = [c1]
        prev = c1

        # pos=2..31
        for pos in range(2, 32):
            logits = self.step(prev, h_last, pos=pos)
            nxt = sample_from_logits(logits, method=method, temperature=temperature, top_p=top_p)
            outs.append(nxt)
            prev = nxt

        rest = torch.cat(outs, dim=1)        # [B,31]
        return torch.cat([c0, rest], dim=1)  # [B,32]

# -------------------------
# Main
# -------------------------
def main():
    device = "cuda"

    ONNX_PATH = "csm_depth_decoder_step_past31.onnx"
    AUDIO_CONTEXT_PATH = "local_models/tts_model/georgian-csm-1b/context_audio_for_inference.wav"
    JSON_CONTEXT_PATH = "local_models/tts_model/georgian-csm-1b/context_text_for_inference.json"

    SAMPLE_METHOD = "nucleus"
    TEMPERATURE = 0.1
    TOP_P = 0.999
    MAX_STEPS = 125 // 2

    processor = AutoProcessor.from_pretrained("local_models/tts_model/csm-1b-processor")
    model = CsmForConditionalGeneration.from_pretrained(
        "local_models/tts_model/georgian-csm-1b",
        device_map=device,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    ).eval()

    ort_dd = ORTDepthDecoderPast31(ONNX_PATH, device_id=0)

    whole_text = "ქართველური ტომების გაერთიანების უმთავრეს მიზეზად ამ ტომთა საერთო საქმიანობა უნდა ჩავთვალოთ - სამთამადნო წარმოება, მეტალურგია და ლითონდამუშავება ."
    words = whole_text.split()
    texts = [" ".join(words[:3])] + [" " + w for w in words[3:]] + ["", ""]
    print("texts:", texts)

    audio_ctx, _ = sf.read(AUDIO_CONTEXT_PATH, dtype="float32")
    with open(JSON_CONTEXT_PATH) as f:
        audio_transcript = json.load(f)

    speaker_id = 0

    # Build inputs chunks
    all_inputs = []
    conversation = [
        {
            "role": f"{speaker_id}",
            "content": [
                {"type": "text", "text": audio_transcript[AUDIO_CONTEXT_PATH.split("/")[-1]]},
                {"type": "audio", "path": audio_ctx},
            ],
        },
        {"role": f"{speaker_id}", "content": [{"type": "text", "text": texts[0]}]},
    ]
    padded_inputs_1 = processor.apply_chat_template(conversation, tokenize=True, return_dict=True).to(device)
    padded_inputs_1["input_values"] = padded_inputs_1["input_values"].to(torch.bfloat16)
    all_inputs.append(padded_inputs_1)

    for t in texts[1:]:
        conversation = [{"role": f"{speaker_id}", "content": [{"type": "text", "text": t}]}]
        cur_input = processor.apply_chat_template(conversation, tokenize=True, return_dict=True).to(device)
        all_inputs.append(cur_input)

    # Init backbone cache
    counter = 0
    inputs = all_inputs[counter]

    text_ids = inputs["input_ids"]
    model_inputs = model.prepare_inputs_for_generation(**inputs)
    text_embeds = model_inputs["inputs_embeds"]

    attn_mask = inputs["attention_mask"].clone()
    pos_ids = torch.arange(text_ids.shape[1], device=device).unsqueeze(0)

    bb_pkv = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=4096,
        device=model.device,
        dtype=text_embeds.dtype,
    )

    bb_out = model.backbone_model(
        inputs_embeds=text_embeds,
        attention_mask=attn_mask,
        position_ids=pos_ids,
        past_key_values=bb_pkv,
        use_cache=True,
        output_hidden_states=True,
    )
    pkv = bb_out.past_key_values
    h_last = bb_out.hidden_states[-1][:, -1, :]
    seq_len = text_embeds.shape[1]

    eos = torch.tensor([model.config.codebook_eos_token_id], device=model.device, dtype=torch.long)

    frames = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    cur_step = 0
    while cur_step < MAX_STEPS:
        # c0 from backbone hidden
        logits0 = model.lm_head(h_last.unsqueeze(1))[:, -1, :]  # [B,V]
        c0 = sample_from_logits(logits0, method=SAMPLE_METHOD, temperature=TEMPERATURE, top_p=TOP_P)  # [B,1]
        c0_1d = c0.view(-1).to(torch.long)

        reached_eos = (c0_1d == eos.view(-1)).any()
        if bool(reached_eos) or cur_step == MAX_STEPS - 1:
            counter += 1
            if counter == len(all_inputs):
                break
            cur_step = 0
            inject_inputs = all_inputs[counter]
            inject_embeds = model.embed_text_tokens(inject_inputs["input_ids"])
            h_last, pkv, seq_len, attn_mask = step_backbone_with_cache(
                model, inject_embeds, attn_mask, pkv, seq_len
            )
            continue

        # ONNX depth decode frame
        frame = ort_dd.generate_frame(
            c0=c0_1d,
            h_last=h_last,  # bf16
            method=SAMPLE_METHOD,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )  # [B,32]
        frames.append(frame)

        # feed frame back to backbone
        frame_3d = frame.unsqueeze(1)  # [B,1,32]
        audio_embeds = model.backbone_model.embed_tokens(frame_3d)
        h_last, pkv, seq_len, attn_mask = step_backbone_with_cache(
            model, audio_embeds, attn_mask, pkv, seq_len
        )

        cur_step += 1

    end.record()
    torch.cuda.synchronize()
    gen_time = start.elapsed_time(end) / 1000.0

    audio = generate_audio_from_frames(model, frames)[0].to(torch.float32).cpu()

    sr = 24_000
    audio_dur = len(audio) / sr
    print(f"Generation time: {gen_time:.3f} s")
    print(f"Audio duration: {audio_dur:.3f} s")
    print(f"RTF: {gen_time / max(audio_dur, 1e-9):.3f}")

    sf.write("output_onnx_dd.wav", audio.numpy(), sr)
    print("Wrote output_onnx_dd.wav")

if __name__ == "__main__":
    main()
