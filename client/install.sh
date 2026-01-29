#!/usr/bin/env bash

set -euo pipefail

# Client environment setup (non-docker)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-venv}"

if command -v apt-get >/dev/null 2>&1 && [ "${SKIP_APT:-0}" != "1" ]; then
  echo "Installing system dependencies (you can skip with SKIP_APT=1)..."
  sudo apt-get update -y
  PY_VER="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  PY_VENV_PKG="python${PY_VER}-venv"
  EXTRA_VENV_PKG=""
  if apt-cache show "$PY_VENV_PKG" >/dev/null 2>&1; then
    EXTRA_VENV_PKG="$PY_VENV_PKG"
  fi
  sudo apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    python3-venv \
    $EXTRA_VENV_PKG
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# Activate main venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo "Client venv ready: $SCRIPT_DIR/$VENV_DIR"

# Optional avatar venv (older torch stack)
if [ "${INSTALL_AVATAR_VENV:-0}" = "1" ]; then
  AVATAR_VENV_DIR="${AVATAR_VENV_DIR:-avatar_venv}"
  if [ ! -d "$AVATAR_VENV_DIR" ]; then
    echo "Creating avatar venv at $AVATAR_VENV_DIR"
    "$PYTHON_BIN" -m venv "$AVATAR_VENV_DIR"
  fi

  AVATAR_PIP="$AVATAR_VENV_DIR/bin/pip"
  "$AVATAR_PIP" install --upgrade pip setuptools wheel

  if [ "${AVATAR_TORCH_DEVICE:-cuda}" = "cpu" ]; then
    "$AVATAR_PIP" install \
      torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
      --index-url https://download.pytorch.org/whl/cpu
  else
    "$AVATAR_PIP" install \
      torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
      --index-url https://download.pytorch.org/whl/cu118
  fi

  "$AVATAR_PIP" install -r avatar_requirements.txt

  if [ "${INSTALL_MMLAB:-0}" = "1" ]; then
    "$AVATAR_PIP" install -U openmim
    "$AVATAR_VENV_DIR/bin/mim" install mmengine
    "$AVATAR_VENV_DIR/bin/mim" install "mmcv==2.0.1"
    "$AVATAR_VENV_DIR/bin/mim" install "mmdet==3.1.0"
    "$AVATAR_VENV_DIR/bin/mim" install "mmpose==1.1.0"
  fi

  echo "Avatar venv ready: $SCRIPT_DIR/$AVATAR_VENV_DIR"
  echo "Export to use it:" 
  echo "  export AVATAR_PYTHON=$SCRIPT_DIR/$AVATAR_VENV_DIR/bin/python"
  echo "  export AVATAR_CREATE_SCRIPT=$SCRIPT_DIR/../model_repository/musetalk/create_avatar.py"
  echo "  export AVATAR_MODEL_ROOT=$SCRIPT_DIR/../local_models/musetalk_model"
  echo "  export AVATAR_RESULT_DIR=$SCRIPT_DIR/../local_models/musetalk_model/testing_avatar_creation"
  echo "  export AVATAR_ROOT=$SCRIPT_DIR/../local_models/musetalk_model/testing_avatar_creation/v15/avatars"
  echo "  export AVATAR_VERSION=v15"
  echo "  export AVATAR_DEVICE=cpu"
  echo "  export AVATAR_MAX_SIDE=512"
fi

echo "Done. You can run the client with:"
echo "  source $VENV_DIR/bin/activate && python main.py"
