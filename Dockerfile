FROM nvcr.io/nvidia/tritonserver:25.11-py3


# Add venv to PATH so all subsequent commands use it
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

RUN pip install --no-cache \
    torch==2.9.0+cu128 \
    torchaudio==2.9.0+cu128 \
    torchvision==0.24.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install NeMo and other dependencies
RUN pip install --no-cache \
    "nemo-toolkit[asr]==2.5.3" \
    soundfile==0.13.1 \
    librosa==0.11.0

# Install transformers and dependencies for LLM
RUN pip install --no-cache \
    transformers==4.57.1 \
    accelerate==1.12.0 \
    bitsandbytes==0.46.0 \
    sentencepiece>=0.2.1 \
    protobuf==5.29.5 \
    datasets==4.4.1 \
    peft>=0.11.0

RUN pip install --no-cache opencv-python-headless==4.10.0.84 \
    diffusers==0.30.2 \
    einops==0.8.1 \
    sounddevice==0.5.3 

EXPOSE 8001

CMD ["tritonserver", "--model-repository", "/models", "--strict-model-config", "false", "--log-verbose", "0", "--log-info", "true", "--log-warning", "true", "--log-error", "true"]