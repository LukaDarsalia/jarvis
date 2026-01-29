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

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "Venv activation script missing; recreating venv..."
  rm -rf "$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "Venv python not found at $VENV_PY"
  exit 1
fi

"$VENV_PY" -m pip install --upgrade pip setuptools wheel
"$VENV_PY" -m pip install -r requirements.txt

echo "Client venv ready: $SCRIPT_DIR/$VENV_DIR"

# Avatar venv (older torch stack) — enabled by default to match docker-compose
if [ "${SKIP_AVATAR_VENV:-0}" != "1" ]; then
  AVATAR_VENV_DIR="${AVATAR_VENV_DIR:-avatar_venv}"
  if [ -z "${AVATAR_DEVICE:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 || [ -e /dev/nvidia0 ]; then
      AVATAR_DEVICE="cuda"
    else
      AVATAR_DEVICE="cpu"
    fi
  fi
  AVATAR_TORCH_DEVICE="${AVATAR_TORCH_DEVICE:-$AVATAR_DEVICE}"
  AVATAR_PYTHON_BIN="${AVATAR_PYTHON_BIN:-}"
  AUTO_INSTALL_PY310="${AUTO_INSTALL_PY310:-1}"
  SUDO=""
  if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
  fi

  ensure_python310() {
    if command -v python3.10 >/dev/null 2>&1; then
      return 0
    fi
    if [ "$AUTO_INSTALL_PY310" != "1" ] || [ "${SKIP_APT:-0}" = "1" ]; then
      return 1
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
      return 1
    fi
    echo "Python 3.10 not found; attempting to install via apt..."
    $SUDO apt-get update -y
    if ! command -v add-apt-repository >/dev/null 2>&1; then
      $SUDO apt-get install -y --no-install-recommends software-properties-common
    fi
    if ! grep -q "deadsnakes" /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null; then
      $SUDO add-apt-repository -y ppa:deadsnakes/ppa
    fi
    $SUDO apt-get update -y
    $SUDO apt-get install -y --no-install-recommends python3.10 python3.10-venv python3.10-dev
    command -v python3.10 >/dev/null 2>&1
  }

  if [ -z "$AVATAR_PYTHON_BIN" ]; then
    if ensure_python310; then
      AVATAR_PYTHON_BIN="python3.10"
    elif command -v python3.11 >/dev/null 2>&1; then
      AVATAR_PYTHON_BIN="python3.11"
    else
      AVATAR_PYTHON_BIN="$PYTHON_BIN"
    fi
  fi
  if ! command -v "$AVATAR_PYTHON_BIN" >/dev/null 2>&1; then
    echo "Avatar venv requires python3.10 for torch==2.0.1."
    echo "Install python3.10 or set AVATAR_PYTHON_BIN to a compatible Python."
    exit 1
  fi
  if [ ! -d "$AVATAR_VENV_DIR" ]; then
    echo "Creating avatar venv at $AVATAR_VENV_DIR"
    "$AVATAR_PYTHON_BIN" -m venv "$AVATAR_VENV_DIR"
  fi

  AVATAR_PY="$AVATAR_VENV_DIR/bin/python"
  if [ ! -x "$AVATAR_PY" ]; then
    echo "Avatar venv python not found at $AVATAR_PY"
    exit 1
  fi

  AVATAR_PY_VER="$("$AVATAR_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  AVATAR_TORCH_VERSION="${AVATAR_TORCH_VERSION:-2.0.1}"
  AVATAR_TORCHVISION_VERSION="${AVATAR_TORCHVISION_VERSION:-0.15.2}"
  AVATAR_TORCHAUDIO_VERSION="${AVATAR_TORCHAUDIO_VERSION:-2.0.2}"

  REQUIREMENTS_FILE="$SCRIPT_DIR/avatar_requirements.txt"
  NEEDS_NUMPY_1235=0
  if [ -f "$REQUIREMENTS_FILE" ]; then
    if command -v rg >/dev/null 2>&1; then
      if rg -q "^numpy==1\\.23\\.5" "$REQUIREMENTS_FILE"; then
        NEEDS_NUMPY_1235=1
      fi
    else
      if grep -q "^numpy==1\\.23\\.5" "$REQUIREMENTS_FILE"; then
        NEEDS_NUMPY_1235=1
      fi
    fi
  fi

  if [ "$AVATAR_PY_VER" = "3.12" ]; then
    if [ "$AVATAR_TORCH_VERSION" = "2.0.1" ]; then
      echo "Avatar venv uses Python $AVATAR_PY_VER, but torch==$AVATAR_TORCH_VERSION wheels are not available."
      echo "Install Python 3.10/3.11 and rerun with AVATAR_PYTHON_BIN=python3.10"
      exit 1
    fi
    if [ "$NEEDS_NUMPY_1235" = "1" ]; then
      echo "avatar_requirements.txt pins numpy==1.23.5, which is not compatible with Python $AVATAR_PY_VER."
      echo "Install Python 3.10/3.11 and rerun with AVATAR_PYTHON_BIN=python3.10"
      exit 1
    fi
  fi

  "$AVATAR_PY" -m pip install --upgrade pip setuptools wheel

  if [ "$AVATAR_TORCH_DEVICE" = "cpu" ]; then
    if [[ "$AVATAR_TORCH_VERSION" != *"+"* ]]; then
      AVATAR_TORCH_VERSION="${AVATAR_TORCH_VERSION}+cpu"
    fi
    if [[ "$AVATAR_TORCHVISION_VERSION" != *"+"* ]]; then
      AVATAR_TORCHVISION_VERSION="${AVATAR_TORCHVISION_VERSION}+cpu"
    fi
    if [[ "$AVATAR_TORCHAUDIO_VERSION" != *"+"* ]]; then
      AVATAR_TORCHAUDIO_VERSION="${AVATAR_TORCHAUDIO_VERSION}+cpu"
    fi
    "$AVATAR_PY" -m pip install \
      torch=="$AVATAR_TORCH_VERSION" torchvision=="$AVATAR_TORCHVISION_VERSION" torchaudio=="$AVATAR_TORCHAUDIO_VERSION" \
      --index-url https://download.pytorch.org/whl/cpu
  else
    if [[ "$AVATAR_TORCH_VERSION" != *"+"* ]]; then
      AVATAR_TORCH_VERSION="${AVATAR_TORCH_VERSION}+cu118"
    fi
    if [[ "$AVATAR_TORCHVISION_VERSION" != *"+"* ]]; then
      AVATAR_TORCHVISION_VERSION="${AVATAR_TORCHVISION_VERSION}+cu118"
    fi
    if [[ "$AVATAR_TORCHAUDIO_VERSION" != *"+"* ]]; then
      AVATAR_TORCHAUDIO_VERSION="${AVATAR_TORCHAUDIO_VERSION}+cu118"
    fi
    "$AVATAR_PY" -m pip install \
      torch=="$AVATAR_TORCH_VERSION" torchvision=="$AVATAR_TORCHVISION_VERSION" torchaudio=="$AVATAR_TORCHAUDIO_VERSION" \
      --index-url https://download.pytorch.org/whl/cu118
  fi

  "$AVATAR_PY" -m pip install -r avatar_requirements.txt

  if [ "${INSTALL_MMLAB:-0}" = "1" ]; then
    "$AVATAR_PY" -m pip install -U openmim
    "$AVATAR_VENV_DIR/bin/mim" install mmengine
    "$AVATAR_VENV_DIR/bin/mim" install "mmcv==2.0.1"
    "$AVATAR_VENV_DIR/bin/mim" install "mmdet==3.1.0"
    "$AVATAR_VENV_DIR/bin/mim" install "mmpose==1.1.0"
  fi

  if ! "$AVATAR_PY" -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
    echo "Torch is not available in the avatar venv. Please check the install output."
    exit 1
  fi

  echo "Avatar venv ready: $SCRIPT_DIR/$AVATAR_VENV_DIR"
  AVATAR_ENV_FILE="${AVATAR_ENV_FILE:-avatar_env.sh}"
  cat > "$AVATAR_ENV_FILE" <<EOF
export AVATAR_PYTHON="$SCRIPT_DIR/$AVATAR_VENV_DIR/bin/python"
export AVATAR_CREATE_SCRIPT="$SCRIPT_DIR/../model_repository/musetalk/create_avatar.py"
export AVATAR_MODEL_ROOT="$SCRIPT_DIR/../local_models/musetalk_model"
export AVATAR_RESULT_DIR="$SCRIPT_DIR/../local_models/musetalk_model/testing_avatar_creation"
export AVATAR_ROOT="$SCRIPT_DIR/../local_models/musetalk_model/testing_avatar_creation/v15/avatars"
export AVATAR_VERSION="v15"
export AVATAR_DEVICE="$AVATAR_DEVICE"
export AVATAR_MAX_SIDE="\${AVATAR_MAX_SIDE:-512}"
export AVATAR_CREATE_FPS="\${AVATAR_CREATE_FPS:-25}"
export AVATAR_CREATE_DURATION_S="\${AVATAR_CREATE_DURATION_S:-0.25}"
export AVATAR_CREATE_BATCH_SIZE="\${AVATAR_CREATE_BATCH_SIZE:-24}"
EOF
  echo "Avatar env file written: $SCRIPT_DIR/$AVATAR_ENV_FILE"
fi

echo "Done. You can run the client with:"
if [ "${SKIP_AVATAR_VENV:-0}" != "1" ]; then
  echo "  source ${AVATAR_ENV_FILE:-avatar_env.sh} && $VENV_PY main.py"
else
  echo "  $VENV_PY main.py"
fi
