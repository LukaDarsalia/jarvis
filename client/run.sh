#!/bin/bash

# Voice Assistant Client Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🎤 Voice Assistant Client${NC}"
echo "=========================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt -q

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

python main.py

