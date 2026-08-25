# GATI — Survival prototype (branch b-survival-prototype)

This branch contains an enhanced prototype that uses per-stage survival models (Random Survival Forests) and a Dockerfile to run the Streamlit demo.

Quick start (local)
1. Create virtualenv (python 3.10+):
   python -m venv .venv && source .venv/bin/activate
2. Install requirements:
   pip install -r requirements.txt
3. Generate synthetic data:
   python generate_data_survival.py
4. Train survival models (this can take a few minutes):
   python train_survival_models.py
5. Run demo:
   streamlit run app_survival.py

Run with Docker (recommended for reproducibility)
1. Build image:
   docker build -t gati-survival:latest .
2. Run container:
   docker run -p 8501:8501 --rm gati-survival:latest
3. Open http://localhost:8501

Notes
- This is a demo-level prototype. The Random Survival Forest models are trained on synthetic data designed to reflect the GATI thesis (stage-wise delays, censoring, district resources, and legal events).
- For production: replace synthetic data with anonymized acquisition records, validate counterfactuals using historic interventions, and add robust explainability (SHAP surrogates or PDPs for survival outputs).
