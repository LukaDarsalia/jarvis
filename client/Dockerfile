FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    python3-venv \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY avatar_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a legacy venv for avatar creation (older torch stack)
ARG AVATAR_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
ARG AVATAR_INSTALL_MMLAB=0
RUN python -m venv /opt/avatar_venv \
    && /opt/avatar_venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/avatar_venv/bin/pip install --no-cache-dir \
        torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        --index-url ${AVATAR_TORCH_INDEX_URL} \
    && /opt/avatar_venv/bin/pip install --no-cache-dir -r avatar_requirements.txt \
    && if [ "${AVATAR_INSTALL_MMLAB}" = "1" ]; then \
         /opt/avatar_venv/bin/pip install --no-cache-dir -U openmim && \
         /opt/avatar_venv/bin/mim install mmengine && \
         /opt/avatar_venv/bin/mim install "mmcv==2.0.1" && \
         /opt/avatar_venv/bin/mim install "mmdet==3.1.0" && \
         /opt/avatar_venv/bin/mim install "mmpose==1.1.0" ; \
       fi

# Copy application code
COPY . .

# Expose the FastAPI port
EXPOSE 8080

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
