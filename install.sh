pip install \
    torch==2.9.0+cu130 \
    torchaudio==2.9.0+cu130 \
    torchvision==0.24.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130

# Install NeMo and other dependencies
pip install \
    "nemo-toolkit[asr]==2.5.3" \
    soundfile==0.13.1 \
    librosa==0.11.0

# Install transformers and dependencies for LLM
pip install  \
    transformers==4.57.1 \
    accelerate==1.12.0 \
    bitsandbytes==0.46.0 \
    sentencepiece>=0.2.1 \
    protobuf==5.29.5 \
    datasets==4.4.1 \
    peft>=0.11.0

pip install opencv-python-headless==4.10.0.84 \
    diffusers==0.30.2 \
    einops==0.8.1 \
    sounddevice==0.5.3 
