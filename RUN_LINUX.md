# RUN_LINUX.md - Complete Linux Setup & Execution Guide

> **For macOS users:** Use the same commands (`.sh` files work on both).

## Quick Start (5 minutes)

```bash
# 1. Make script executable
chmod +x setup_and_run.sh

# 2. Run everything automatically
./setup_and_run.sh

# 3. Open dashboard
streamlit run app_gati.py
```

**That's it!** The script handles:
✅ Python version check (3.10+)  
✅ Virtual environment setup  
✅ Dependency installation  
✅ Synthetic data generation  
✅ Model training  
✅ SHAP surrogate training  

---

## Step-by-Step Manual Setup

If you prefer manual control or the script doesn't work for you:

### Prerequisites

```bash
# Check Python version (need 3.10+)
python3 --version

# Install git (if not already installed)
sudo apt-get install git python3-venv python3-dev
```

### 1. Clone & Navigate

```bash
git clone https://github.com/aminul821/gati-prototype.git
cd gati-prototype
```

### 2. Create Virtual Environment

```bash
# Create
python3 -m venv .venv

# Activate
source .venv/bin/activate

# You should see (.venv) in your terminal prompt
```

### 3. Install Dependencies

```bash
# Upgrade package managers
pip install --upgrade pip setuptools wheel

# Install all packages
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn, sksurv, streamlit, shap; print('✓ All packages OK')"
```

### 4. Create Required Directories

```bash
mkdir -p data models
```

### 5. Generate Synthetic Data

```bash
python generate_data_survival.py
```

**Output:**
```
Wrote data/ with projects.csv, legal_events.csv, districts.json
```

**Verify:**
```bash
ls -lh data/
# Should show:
# - projects.csv (50-100 KB)
# - legal_events.csv (10-20 KB)
# - districts.json (2-5 KB)
```

### 6. Train Survival Models

```bash
python train_survival_models.py
```

**Expected output:**
```
Training RSF for stage notification on XXXX rows...
Saved models/notification_rsf.joblib
Training RSF for stage award on XXXX rows...
Saved models/award_rsf.joblib
...
Training complete.
```

**This takes 2-5 minutes.** Get coffee ☕

**Verify:**
```bash
ls -lh models/
# Should show 5 .joblib files:
# - notification_rsf.joblib
# - award_rsf.joblib
# - compensation_rsf.joblib
# - rnr_rsf.joblib
# - possession_rsf.joblib
```

### 7. (Optional) Train SHAP Surrogates

For explainability features in the dashboard:

```bash
python explainability.py
```

**Output:**
```
Training surrogate for notification on XXX samples...
Training surrogate for award on XXX samples...
...
Saved surrogates to models/surrogates.joblib
```

**Verify:**
```bash
ls -lh models/surrogates.joblib
# Should exist and be 1-5 MB
```

### 8. Launch Dashboard

**Option A: Unified Dashboard (Recommended)**
```bash
streamlit run app_gati.py
```

**Option B: Legacy Dashboard**
```bash
streamlit run app_survival.py
```

**Output:**
```
  You can now view your Streamlit app in your browser.

  URL: http://localhost:8501
```

**Open in browser:** http://localhost:8501

---

## Dashboard Features

### Tab 1: 📊 National Overview
- Top 20 high-risk projects
- District contention ranking
- Adjustable risk horizon (30-730 days)

### Tab 2: 🔍 Project Deep-Dive
- Per-stage survival curves
- SHAP feature explanations
- Risk scores
- Legal risk profile

### Tab 3: 🌐 District Contention Graph
- Bottleneck districts (load > 1.0)
- District capacity vs legal risk
- Full district statistics

### Tab 4: 🎛️ What-If Simulator
- Project-level: Adjust capacity/pending projects
- District-level: Simulate capacity changes
- See impact on risk & rank

### Tab 5: 📈 Legal Risk Analysis
- Legal events by type
- High-risk projects
- Litigation density by district

---

## Troubleshooting

### "Command 'python3' not found"

Install Python 3.10+ first:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.14 python3.14-venv python3.14-dev

# Then use python3.14 instead of python3
python3.14 -m venv .venv
```

### "ModuleNotFoundError: No module named 'pandas'"

Ensure virtual environment is activated:

```bash
source .venv/bin/activate
# Should show (.venv) in prompt

# Then reinstall
pip install -r requirements.txt
```

### "Permission denied" for setup_and_run.sh

Make it executable:

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

### "No models found" when launching dashboard

Train them first:

```bash
python generate_data_survival.py
python train_survival_models.py
```

Then launch dashboard.

### Streamlit port 8501 already in use

Use a different port:

```bash
streamlit run app_gati.py --server.port 8502
```

Open http://localhost:8502

### "SHAP explainer not available" in dashboard

Train surrogates (optional but recommended):

```bash
python explainability.py
```

### Memory/Performance Issues

If training is slow or crashes:

```bash
# Reduce synthetic data size (edit generate_data_survival.py)
# Change: N = 2000  →  N = 500

# Then regenerate:
python generate_data_survival.py
python train_survival_models.py
```

---

## Docker Alternative (Easier)

If you have Docker installed:

```bash
# Build image
docker build -t gati:latest .

# Run container
docker run -p 8501:8501 gati:latest

# Open http://localhost:8501
```

No virtual environment setup needed!

---

## Full Execution Flow Diagram

```
┌─────────────────────────────────────────┐
│ setup_and_run.sh (Automated)            │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ 1. Check Python 3.10+                   │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ 2. Create .venv                         │
│ 3. pip install requirements.txt         │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ 4. python generate_data_survival.py     │
│    → data/*.csv, *.json (2000 projects) │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ 5. python train_survival_models.py      │
│    → models/{stage}_rsf.joblib (5x)    │
│    Duration: 2-5 minutes                │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ 6. python explainability.py (optional)  │
│    → models/surrogates.joblib           │
│    Duration: 1-2 minutes                │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ streamlit run app_gati.py               │
│ Opens: http://localhost:8501            │
└─────────────────────────────────────────┘
```

---

## Deactivate Virtual Environment

When done:

```bash
deactivate
```

To reactivate later:

```bash
source .venv/bin/activate
streamlit run app_gati.py
```

---

## Files Created During Setup

```
gati-prototype/
├── .venv/                          ← Virtual environment
├── data/
│   ├── projects.csv               ← 2000 projects
│   ├── legal_events.csv           ← Legal disputes
│   └── districts.json             ← District metadata
├── models/
│   ├── notification_rsf.joblib    ← Trained model
│   ├── award_rsf.joblib
│   ├── compensation_rsf.joblib
│   ├── rnr_rsf.joblib
│   ├── possession_rsf.joblib
│   └── surrogates.joblib          ← SHAP explainers (optional)
└── [source files remain unchanged]
```

---

## Performance Notes

| Operation | Time | CPU | RAM |
|-----------|------|-----|-----|
| Data generation | 30s | Medium | 500MB |
| Model training (5 stages) | 2-5m | High | 2GB+ |
| SHAP surrogates | 1-2m | High | 1GB |
| Dashboard startup | 5-10s | Medium | 500MB |

Typical setup time: **5-10 minutes** on modern hardware.

---

## Next Steps

1. **Explore the dashboard** — all 5 tabs
2. **Modify synthetic data** — edit `generate_data_survival.py` to change:
   - Number of projects: `N = 2000`
   - Number of districts: `DISTRICTS = [...]`
   - Risk parameters and stage durations
3. **Integrate real data** — replace `data/*.csv` with production records
4. **Deploy** — use Docker for production deployment

---

**Questions?** Check the main [README.md](README.md) for more details.

**Author:** Aminul (aminul821)  
**Last Updated:** 2024-Q4  
**Python:** 3.10+
