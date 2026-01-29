from huggingface_hub import snapshot_download
import subprocess
import os


TOKEN = os.environ["HF_TOKEN"]
HF_HOME = os.environ.get("HF_HOME", "local_models/hf_cache")
os.makedirs(HF_HOME, exist_ok=True)


# LLM (English)
snapshot_download(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    local_dir="local_models/llm_model",
    resume_download=True,
    token=TOKEN,
)

# STT (English, NeMo)
snapshot_download(
    repo_id="nvidia/stt_en_fastconformer_hybrid_large_pc",
    local_dir="local_models/stt_model",
    resume_download=True,
    token=TOKEN,
)

# TTS (Pocket-TTS weights + voice presets cache)
snapshot_download(
    repo_id="kyutai/pocket-tts",
    cache_dir=HF_HOME,
    resume_download=True,
    token=TOKEN,
)
snapshot_download(
    repo_id="kyutai/pocket-tts-without-voice-cloning",
    cache_dir=HF_HOME,
    resume_download=True,
    token=TOKEN,
)

musetalk_install = f"""
# Set the checkpoints directory
CheckpointsDir="local_models/musetalk_model"

# Create necessary directories
mkdir -p local_models/musetalk_model/musetalk local_models/musetalk_model/musetalkV15 local_models/musetalk_model/syncnet local_models/musetalk_model/dwpose local_models/musetalk_model/face-parse-bisent local_models/musetalk_model/sd-vae local_models/musetalk_model/whisper


# Set HuggingFace mirror endpoint
export HF_ENDPOINT=https://hf-mirror.com

# Download MuseTalk V1.0 weights
hf download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin" \
  --token {TOKEN}

# Download MuseTalk V1.5 weights (unet.pth)
hf download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth" \
  --token {TOKEN}

# Download SD VAE weights
hf download stabilityai/sd-vae-ft-mse \
  --local-dir $CheckpointsDir/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin" \
  --token {TOKEN}

# Download Whisper weights
hf download openai/whisper-tiny \
  --local-dir $CheckpointsDir/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json" \
  --token {TOKEN}

# Download DWPose weights
hf download yzd-v/DWPose \
  --local-dir $CheckpointsDir/dwpose \
  --include "dw-ll_ucoco_384.pth" \
  --token {TOKEN}

# Download SyncNet weights
hf download ByteDance/LatentSync \
  --local-dir $CheckpointsDir/syncnet \
  --include "latentsync_syncnet.pt" \
  --token {TOKEN}

# Download Face Parse Bisent weights
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O $CheckpointsDir/face-parse-bisent/79999_iter.pth
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o $CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth

echo "✅ All weights have been downloaded successfully!" 
"""


# Run a command as a string
result = subprocess.run(musetalk_install, shell=True, capture_output=True, text=True)

# Access output, error, and return code
print(result.stdout)
print(result.returncode)
print(result.stderr)

os.makedirs("local_models/vad_model", exist_ok=True)

import torch
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)
model.eval()

# script (preferred over trace for models with control flow)
jit_model = torch.jit.script(model)

# save
jit_model.save("local_models/vad_model/silero_vad.jit")

print("Saved: silero_vad.jit")
