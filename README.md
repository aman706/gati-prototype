# GATI — Government Acquisition & Transition Index (Enhanced Prototype)

**Python 3.14+ compatible survival analysis and decision support system** for monitoring delays in government acquisition projects across India.

## What's New: Four-Pillar Enhancement

This version integrates **all four pillars** of the GATI thesis:

### ✅ Pillar 1: Per-Stage Survival Models
- **File:** `train_survival_models.py`, `app_gati.py` (Tab 2)
- Random Survival Forests for each lifecycle stage: notification → award → compensation → R&R → possession
- Censoring-aware hazard estimation (accounts for incomplete projects)
- **Fixed:** Updated from broken `sksurv==0.15.2` to `scikit-survival>=0.21.0`

### ✅ Pillar 1b: SHAP-Based Explainability
- **File:** `explainability.py`, `app_gati.py` (Tab 2)
- Trains surrogate regression models per stage (since SHAP doesn't work on survival curves)
- Per-project ranked feature contributions: *"Why is this project's risk high?"*
- TreeExplainer outputs ranked SHAP values for all features

### ✅ Pillar 2: Legal Dispute Risk Scoring
- **File:** `legal_risk.py`, `app_gati.py` (Tab 5)
- Project-level legal risk with recency decay (events older than 2 years decay exponentially)
- Event severity weighting: notice (0.3) → petition (0.6) → injunction (1.0) → settlement (0.8)
- District-level litigation density (disputes per active project)

### ✅ Pillar 3: District Contention Graph (Network Analysis)
- **File:** `district_graph.py`, `app_gati.py` (Tab 3)
- Directed graph of districts with capacity + legal risk + load metrics
- Bottleneck ranking: load_score (projects/capacity) + legal_risk combined
- What-if simulation: *"What if this district hired 2 more verification teams?"*

### ✅ Pillar 4: Integrated What-If Simulator
- **File:** `app_gati.py` (Tab 4)
- **Project-level:** Adjust collector capacity, pending projects → see risk change
- **District-level:** Simulate capacity or project reductions → recalculate contention rank

---

## Quick Start

### Local Setup (Python 3.14+)

```bash
# 1. Clone repo
git clone https://github.com/aminul821/gati-prototype
cd gati-prototype

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Generate synthetic data
python generate_data_survival.py
# Outputs: data/projects.csv, data/legal_events.csv, data/districts.json

# 5. Train survival models (takes 2-5 minutes)
python train_survival_models.py
# Outputs: models/{notification,award,compensation,rnr,possession}_rsf.joblib

# 6. Train SHAP surrogates (optional, ~1 minute)
python explainability.py
# Outputs: models/surrogates.joblib

# 7. Launch unified dashboard
streamlit run app_gati.py
# Opens http://localhost:8501
```

### Docker Setup (Recommended)

```bash
# Build
docker build -t gati:latest .

# Run
docker run -p 8501:8501 --rm gati:latest

# Open http://localhost:8501
```

---

## Dashboard Tabs

| Tab | Purpose | Key Features |
|-----|---------|--------------|
| 📊 **National Overview** | Org-wide status | Top 20 high-risk projects, district contention ranking, stage selection |
| 🔍 **Project Deep-Dive** | Single project analysis | Survival curves per stage, risk-within-horizon, SHAP explanations |
| 🌐 **District Contention Graph** | Bottleneck identification | Bottleneck ranking, load vs legal risk scatter, all district stats |
| 🎛️ **What-If Simulator** | Scenario planning | Project-level & district-level counterfactuals with impact metrics |
| 📈 **Legal Risk Analysis** | Dispute trends | Events by type, high-risk projects, litigation density by district |

---

## Data Format

### `data/projects.csv`
2000 synthetic projects with:
- Project ID, district, type (Road/Rail/Irrigation/etc), land category
- Per-stage dates: entry, completion, duration (days), event (completed=1/censored=0)
- District context: collector capacity, verification teams, pending projects
- Legal risk (0–1 baseline, enriched in Pillar 2)

### `data/legal_events.csv`
Simulated legal events (notice, petition, injunction, settlement) linked to projects:
- Project ID, district, date, event type, severity (0–1)
- Used in Pillar 2 to compute legal_risk_score with recency decay

### `data/districts.json`
District-level metadata (collector capacity, verification teams, payroll cycle offset).

---

## Production Readiness Checklist

- [x] Censoring & survival estimation (Pillar 1)
- [x] SHAP explainability (Pillar 1b)
- [x] Legal risk scoring (Pillar 2)
- [x] District contention graph (Pillar 3)
- [x] What-if simulators (Pillar 4)
- [x] Python 3.14+ compatibility with PEP 695 type aliases
- [ ] **Replace synthetic data:** Use anonymized court acquisition records (NJDG via data.gov.in for district baseline, Kaggle/HuggingFace judgment datasets for legal events)
- [ ] **Validate counterfactuals:** Cross-check "what-if" predictions against historical interventions (e.g., past capacity hires)
- [ ] **Add uncertainty quantification:** Confidence intervals on survival curves (quantile regression forests)
- [ ] **Enable model monitoring:** Track prediction vs actual completion times, retrain monthly
- [ ] **Implement access control:** Role-based dashboards (judge, administrator, data officer)

---

## File Structure

```
gati-prototype/
├── generate_data_survival.py       Pillar 1: Synthetic data generation (2000 projects)
├── train_survival_models.py        Pillar 1: RandomSurvivalForest per stage (fixed: sparse_output)
├── legal_risk.py                   Pillar 2: Legal risk scoring + recency decay
├── district_graph.py               Pillar 3: Network-based contention analysis
├── explainability.py               Pillar 1b: SHAP surrogate training
├── app_survival.py                 (Legacy) Single-tab prototype
├── app_gati.py                     ⭐ Unified five-tab dashboard
├── requirements.txt                Python 3.14+ dependencies (scikit-survival>=0.21.0)
├── Dockerfile                      Python 3.14-slim with health checks
├── README.md                       (this file)
└── data/                           (generated)
    ├── projects.csv
    ├── legal_events.csv
    └── districts.json
└── models/                         (trained)
    ├── notification_rsf.joblib
    ├── award_rsf.joblib
    ├── compensation_rsf.joblib
    ├── rnr_rsf.joblib
    ├── possession_rsf.joblib
    └── surrogates.joblib           (SHAP explainers per stage)
```

---

## Key Improvements Over v0

| Aspect | Before | After |
|--------|--------|-------|
| **Python** | 3.10 | 3.14+ with PEP 695 type aliases |
| **scikit-survival** | Broken: `sksurv==0.15.2` | Fixed: `scikit-survival>=0.21.0` |
| **Explainability** | Feature importance only | SHAP per-feature contributions |
| **Legal Risk** | Static 0–1 value | Weighted by event type + recency decay |
| **District Analysis** | Flat pending_projects count | Full networkx graph + contention rank |
| **What-If Scope** | Project-level only | Project + district level |
| **Dashboard** | 1 basic tab | 5 integrated tabs |
| **Type Safety** | Minimal | Full PEP 695 annotations |

---

## Example: Running a What-If Scenario

**Scenario:** "District_5 is a bottleneck. What if we increase collector capacity from 8 to 12?"

1. Open dashboard → **What-If Simulator** tab
2. Select **District-Level** mode
3. Choose District_5
4. Set new capacity: 12
5. Click **Simulate District Change**
6. **Output:**
   ```
   Old Load: 2.50 (250% capacity)
   New Load: 1.67 (167% capacity)
   New Rank Position: #7 of 30 (previously #15)
   ```

---

## Troubleshooting

### "No module named 'sksurv'"
✅ **Fixed:** Now uses `scikit-survival>=0.21.0`. Run: `pip install --upgrade scikit-survival`

### "SHAP explainer not available"
Train surrogates: `python explainability.py` (requires trained RSF models first)

### Streamlit shows "data/projects.csv not found"
Generate data: `python generate_data_survival.py`

### Models trained but app shows "No RSF models"
Verify all 5 `.joblib` files exist in `models/` directory. Train if missing: `python train_survival_models.py`

---

## Citation

Built as enhanced prototype for GATI (Government Acquisition & Transition Index) thesis work on acquisition delay prediction using survival analysis and network contention models.

---

**Maintained by:** Aminul (aminul821)  
**Last updated:** 2024-Q4  
**Python:** 3.14+ | **License:** MIT
