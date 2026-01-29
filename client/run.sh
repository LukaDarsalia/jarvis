#!/usr/bin/env bash

# Voice Assistant Client Startup Script

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🎤 Voice Assistant Client${NC}"
echo "=========================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-venv}"
VENV_PY="$VENV_DIR/bin/python"

# Ensure venv exists
if [ ! -x "$VENV_PY" ]; then
    echo -e "${YELLOW}Client venv not found. Running install.sh...${NC}"
    bash ./install.sh
fi

VENV_PY="$VENV_DIR/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo -e "${RED}✗ venv python not found at $VENV_PY${NC}"
    exit 1
fi

# Load avatar env if present (sets AVATAR_PYTHON, paths, etc.)
AVATAR_ENV_FILE="${AVATAR_ENV_FILE:-avatar_env.sh}"
if [ -f "$AVATAR_ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$AVATAR_ENV_FILE"
fi

# Check if Triton server is available
echo -e "${YELLOW}Checking Triton server connection...${NC}"
if curl -s http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Triton server is ready${NC}"
else
    echo -e "${RED}✗ Triton server is not available at localhost:8000${NC}"
    echo -e "${YELLOW}Please start the Triton server first:${NC}"
    echo "  cd .. && docker-compose up -d"
    echo ""
    echo -e "${YELLOW}Continuing anyway...${NC}"
fi

# Start the client
echo ""
echo -e "${GREEN}Starting Voice Assistant Client...${NC}"
echo -e "Web UI will be available at: ${GREEN}http://localhost:8080${NC}"
echo ""

"$VENV_PY" main.py
