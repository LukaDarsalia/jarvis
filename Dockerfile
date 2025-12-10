FROM nvcr.io/nvidia/tritonserver:25.11-py3

# Install uv for faster package installation
RUN pip install uv

RUN uv venv /opt/venv

# Add venv to PATH so all subsequent commands use it
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

RUN uv pip install --no-cache \
    torch==2.9.0+cu128 \
    torchaudio==2.9.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install NeMo and other dependencies
RUN uv pip install --no-cache \
    "nemo-toolkit[asr]==2.5.3" \
    soundfile==0.13.1 \
    librosa==0.11.0

# Install transformers and dependencies for LLM
RUN uv pip install --no-cache \
    transformers==4.57.1 \
    accelerate==1.12.0 \
    bitsandbytes==0.46.0 \
    sentencepiece>=0.2.1 \
    protobuf==5.29.5 \
    datasets==4.4.1 \
    peft>=0.11.0

# The model repository will be mounted at runtime
