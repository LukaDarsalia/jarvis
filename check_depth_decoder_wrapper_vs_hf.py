#!/usr/bin/env python3
# generate_with_depth_decoder_wrapper.py
#
# Goal: run full TTS generation but replace HF depth_decoder.generate with YOUR one-step wrapper (PyTorch),
# to isolate whether ONNX is the problem.
#
# Optional: compare wrapper output vs HF generate for the first frame (greedy) to validate wrapper correctness.

import os
import json
import torch
import torch.nn.functional as F
import soundfile as sf
from transformers import CsmForConditionalGeneration, AutoProcessor
from transformers.cache_utils import StaticCache

from csm_depth_decoder_wrapper import DepthDecoderOneStepExport  # <-- your wrapper

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
    # frames_list: list of [B,32]
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
    """
    new_embeds: [B, n_new, D]
    attn_mask:  [B, T]
    """
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
# Depth decoding with your wrapper (full 32-length past, PyTorch)
# -------------------------
@torch.no_grad()
def depth_decode_frame_with_wrapper(
    dd_wrapper: DepthDecoderOneStepExport,
    h_last: torch.Tensor,                 # [B,Hbb] backbone last hidden
    c0: torch.Tensor,                     # [B] or [B,1] first codebook token
    method="nucleus",
    temperature=0.1,
    top_p=0.999,
):
    """
    Produces one frame [B,32] = [c0, c1..c31] using wrapper step-by-step.
    This keeps full past (Tpast grows 0..31), so it should match HF semantics.
    """
    device = h_last.device
    dtype = next(dd_wrapper.parameters()).dtype

    B = int(h_last.shape[0])
    if c0.dim() == 1:
        c0 = c0.view(B, 1)
    else:
        c0 = c0.view(B, 1)

    # config
    L = len(dd_wrapper.layers)
    kv_heads = dd_wrapper.config.num_key_value_heads
    hd = dd_wrapper.layers[0].self_attn.head_dim

    # init empty past: Tpast = 0
    past = [
        torch.empty((B, kv_heads, 0, hd), device=device, dtype=dtype)
        for _ in range(2 * L)
    ]

    # step pos=0 (conditioning). logits ignored.
    tok0 = torch.zeros((B, 1), device=device, dtype=torch.long)
    cache_pos = torch.tensor([0], device=device, dtype=torch.long)
    attn_mask = torch.zeros((B, 1, 1, 1), device=device, dtype=dtype)
    _logits0, *past = dd_wrapper(tok0, h_last.to(dtype), attn_mask, cache_pos, *past)

    # step pos=1 (consume c0 -> predict c1)
    cache_pos = torch.tensor([1], device=device, dtype=torch.long)
    attn_mask = torch.zeros((B, 1, 1, 2), device=device, dtype=dtype)
    logits, *past = dd_wrapper(c0.to(torch.long), h_last.to(dtype), attn_mask, cache_pos, *past)
    c1 = sample_from_logits(logits[:, 0, :], method=method, temperature=temperature, top_p=top_p)  # [B,1]
    outs = [c1]

    prev = c1
    # pos=2..31
    for pos in range(2, 32):
        cache_pos = torch.tensor([pos], device=device, dtype=torch.long)
        attn_mask = torch.zeros((B, 1, 1, pos + 1), device=device, dtype=dtype)
        logits, *past = dd_wrapper(prev.to(torch.long), h_last.to(dtype), attn_mask, cache_pos, *past)
        nxt = sample_from_logits(logits[:, 0, :], method=method, temperature=temperature, top_p=top_p)
        outs.append(nxt)
        prev = nxt

    rest = torch.cat(outs, dim=1)          # [B,31]
    frame = torch.cat([c0, rest], dim=1)   # [B,32]
    return frame


# -------------------------
# Optional: compare wrapper vs HF generate for 1 frame (greedy)
# -------------------------
@torch.no_grad()
def hf_depth_decode_frame_greedy(model, h_last, c0):
    B = int(h_last.shape[0])
    if c0.dim() == 1:
        c0 = c0.view(B, 1)
    depth_prompt = F.pad(c0.to(torch.long), (1, 0), value=0)  # [B,2] = [dummy, c0]
    out = model.depth_decoder.generate(
        input_ids=depth_prompt,
        backbone_last_hidden_state=h_last.clone(),
        max_new_tokens=model.config.num_codebooks - 1,  # 31
        do_sample=False,                                # greedy
        use_cache=True,
        logits_to_keep=1,
        return_dict_in_generate=False,
    )
    return out[:, 1:]  # [B,32]


def main():
    device = "cuda"

    # Params
    SAMPLE_METHOD = "nucleus"  # "greedy" | "nucleus" | "topk" | "multinomial"
    TEMPERATURE = 0.1
    TOP_P = 0.999
    MAX_STEPS = 125 // 2

    COMPARE_FIRST_FRAME_GREEDY = True  # sanity check wrapper correctness

    AUDIO_CONTEXT_PATH = "local_models/tts_model/georgian-csm-1b/context_audio_for_inference.wav"
    JSON_CONTEXT_PATH = "local_models/tts_model/georgian-csm-1b/context_text_for_inference.json"

    processor = AutoProcessor.from_pretrained("local_models/tts_model/csm-1b-processor")
    model = CsmForConditionalGeneration.from_pretrained(
        "local_models/tts_model/georgian-csm-1b",
        device_map=device,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    ).eval()

    # Wrapper (PyTorch)
    dd_wrapper = DepthDecoderOneStepExport(model.depth_decoder).eval().to(device)

    whole_text = "ქართველური ტომების გაერთიანების უმთავრეს მიზეზად ამ ტომთა საერთო საქმიანობა უნდა ჩავთვალოთ - სამთამადნო წარმოება, მეტალურგია და ლითონდამუშავება ."
    words = whole_text.split()
    texts = [" ".join(words[:3])] + [" " + w for w in words[3:]] + ["", ""]
    print("texts:", texts)

    audio_ctx, _ = sf.read(AUDIO_CONTEXT_PATH, dtype="float32")
    with open(JSON_CONTEXT_PATH) as f:
        audio_transcript = json.load(f)

    speaker_id = 0

    # Build all_inputs (same as your working code)
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

    # ---- Init backbone cache from first chunk ----
    counter = 0
    inputs = all_inputs[counter]

    text_ids = inputs["input_ids"]
    model_inputs = model.prepare_inputs_for_generation(**inputs)
    text_embeds = model_inputs["inputs_embeds"]               # [B,T0,D]

    attn_mask = inputs["attention_mask"].clone()              # [B,T0]
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
    h_last = bb_out.hidden_states[-1][:, -1, :]               # [B,2048]
    seq_len = text_embeds.shape[1]

    eos = torch.tensor([model.config.codebook_eos_token_id], device=model.device, dtype=torch.long)

    frames = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    cur_step = 0
    first_frame_checked = False

    while cur_step < MAX_STEPS:
        # c0 from backbone
        logits0 = model.lm_head(h_last.unsqueeze(1))[:, -1, :]         # [B,V]
        c0 = sample_from_logits(logits0, method=SAMPLE_METHOD, temperature=TEMPERATURE, top_p=TOP_P)  # [B,1]
        c0_1d = c0.view(-1).to(torch.long)

        reached_eos = (c0_1d == eos.view(-1)).any()
        if bool(reached_eos) or cur_step == MAX_STEPS - 1:
            counter += 1
            if counter == len(all_inputs):
                break
            cur_step = 0
            inject_inputs = all_inputs[counter]
            inject_embeds = model.embed_text_tokens(inject_inputs["input_ids"])  # [B,L,D]
            h_last, pkv, seq_len, attn_mask = step_backbone_with_cache(
                model, inject_embeds, attn_mask, pkv, seq_len
            )
            continue

        # optional correctness check once (greedy compare)
        if COMPARE_FIRST_FRAME_GREEDY and not first_frame_checked:
            c0_g = torch.argmax(logits0, dim=-1, keepdim=True).to(torch.long)  # greedy c0 for check
            frame_hf = hf_depth_decode_frame_greedy(model, h_last, c0_g)
            frame_wr = depth_decode_frame_with_wrapper(
                dd_wrapper,
                h_last=h_last,
                c0=c0_g,
                method="greedy",
                temperature=1.0,
                top_p=1.0,
            )
            same = bool(torch.equal(frame_hf, frame_wr))
            print("COMPARE first frame (greedy):", "MATCH" if same else "MISMATCH")
            if not same:
                # show first few tokens for debugging
                print("hf[:10]:", frame_hf[0, :10].tolist())
                print("wr[:10]:", frame_wr[0, :10].tolist())
            first_frame_checked = True

        # depth decode with wrapper (sampled)
        frame = depth_decode_frame_with_wrapper(
            dd_wrapper,
            h_last=h_last,
            c0=c0_1d,
            method=SAMPLE_METHOD,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )  # [B,32]
        frames.append(frame)

        # feed frame into backbone
        frame_3d = frame.unsqueeze(1)                           # [B,1,32]
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

    sf.write("output_wrapper.wav", audio.numpy(), sr)
    print("Wrote output_wrapper.wav")


if __name__ == "__main__":
    main()
