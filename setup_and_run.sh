#!/bin/bash
# setup_and_run.sh - Complete setup script for GATI prototype on Linux
# Python 3.14+ compatible
# Usage: chmod +x setup_and_run.sh && ./setup_and_run.sh

set -e  # Exit on error

echo "======================================================================"
echo "GATI — Government Acquisition & Transition Index"
echo "Complete Setup & Execution Script (Linux/macOS)"
echo "======================================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}[1/8]${NC} Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python: $python_version"

if ! python3 -c 'import sys; assert sys.version_info >= (3, 10)'; then
    echo -e "${RED}Error: Python 3.10+ required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Create virtual environment
echo -e "${BLUE}[2/8]${NC} Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}→ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip, setuptools, wheel
echo -e "${BLUE}[3/8]${NC} Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ Package managers updated${NC}"
echo ""

# Install dependencies
echo -e "${BLUE}[4/8]${NC} Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo -e "${GREEN}✓ All dependencies installed${NC}"
echo ""

# Create data directory
echo -e "${BLUE}[5/8]${NC} Creating data and models directories..."
mkdir -p data models
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Generate synthetic data
echo -e "${BLUE}[6/8]${NC} Generating synthetic data..."
echo "   This creates 2000 projects across 30 districts..."
python generate_data_survival.py
echo -e "${GREEN}✓ Synthetic data generated${NC}"
echo "   - data/projects.csv (2000 rows)"
echo "   - data/legal_events.csv"
echo "   - data/districts.json"
echo ""

# Train survival models
echo -e "${BLUE}[7/8]${NC} Training per-stage survival models (Random Survival Forests)..."
echo "   Training on 5 stages: notification, award, compensation, rnr, possession"
echo "   This may take 2-5 minutes..."
python train_survival_models.py
echo -e "${GREEN}✓ Survival models trained${NC}"
echo "   Outputs: models/{stage}_rsf.joblib"
echo ""

# Optional: Train SHAP surrogates
echo -e "${BLUE}[8/8]${NC} Training SHAP explainability surrogates..."
echo "   Creating surrogate models for feature importance..."
python explainability.py
echo -e "${GREEN}✓ SHAP surrogates trained${NC}"
echo "   Output: models/surrogates.joblib"
echo ""

# Final summary
echo "======================================================================"
echo -e "${GREEN}✓ SETUP COMPLETE!${NC}"
echo "======================================================================"
echo ""
echo -e "${YELLOW}To run the dashboard:${NC}"
echo ""
echo "  Option 1 - Unified Dashboard (All 4 Pillars):"
echo "  $ streamlit run app_gati.py"
echo ""
echo "  Option 2 - Legacy Survival Dashboard:"
echo "  $ streamlit run app_survival.py"
echo ""
echo -e "${BLUE}Dashboard will open at:${NC} http://localhost:8501"
echo ""
echo -e "${YELLOW}To deactivate virtual environment later:${NC}"
echo "  $ deactivate"
echo ""
echo "======================================================================"
