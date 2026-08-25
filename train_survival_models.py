"""
train_survival_models.py

Trains a RandomSurvivalForest per RFCTLARR stage using scikit-survival.
Also computes a risk-within-T (default 180 days) for training rows and trains a surrogate
HistGradientBoostingRegressor per stage to predict that risk; the surrogate is used for
SHAP-based explainability in the app.
Saves pipelines to models/
"""
import os
import joblib
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/projects.csv")
STAGES = ["notification", "award", "compensation", "rnr", "possession"]

# features to use
cat_features = ["district", "project_type", "land_category"]
num_features = ["area_ha", "affected_families", "collector_capacity", "verification_teams", "pending_projects", "legal_risk"]
feature_cols = cat_features + num_features

# preprocessor reused for each stage (keeps one-hot mapping consistent)
preprocessor = ColumnTransformer([
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_features)
], remainder="passthrough")

RISK_HORIZON = 180  # days for risk-within-T surrogate target

for stage in STAGES:
    print(f"Preparing data for stage: {stage}")
    time_col = f"duration_{stage}"
    event_col = f"event_{stage}"
    mask = df[f"{stage}_entry"] != ""
    X_stage = df.loc[mask, feature_cols].reset_index(drop=True)
    times = df.loc[mask, time_col].to_numpy(dtype=float)
    events = df.loc[mask, event_col].to_numpy(dtype=bool)

    # structured survival target
    y = Surv.from_arrays(events, times)

    # RSF pipeline
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=5, random_state=42)
    pipeline_rsf = Pipeline([("pre", preprocessor), ("rsf", rsf)])

    print(f"Training RSF for stage {stage} on {len(X_stage)} rows...")
    pipeline_rsf.fit(X_stage, y)
    joblib.dump(pipeline_rsf, f"models/{stage}_rsf.joblib")
    print(f"Saved models/{stage}_rsf.joblib")

    # compute risk-within-T for each training row using RSF survival functions
    try:
        surv_funcs = pipeline_rsf.predict_survival_function(X_stage)
    except Exception as e:
        print("Could not compute survival functions for surrogate target:", e)
        surv_funcs = []

    risks = []
    expected_times = []
    for fn in surv_funcs:
        # fn may be a StepFunction-like with .x (times) and .y (survival probs)
        if hasattr(fn, 'x') and hasattr(fn, 'y'):
            times_arr = np.array(fn.x)
            probs = np.array(fn.y)
        else:
            # fallback to sequence of (t,p)
            arr = np.array(fn)
            times_arr = arr[:,0]
            probs = arr[:,1]
        # risk within horizon = 1 - S(horizon)
        if RISK_HORIZON <= times_arr[0]:
            S_h = 1.0
        else:
            idx = np.searchsorted(times_arr, RISK_HORIZON, side='right') - 1
            idx = max(0, min(idx, len(probs)-1))
            S_h = probs[idx]
        risks.append(float(1.0 - S_h))
        # expected time (restricted mean up to horizon): integrate S(t) from 0..RISK_HORIZON
        # numeric integration via step areas
        times_for_integ = np.concatenate(([0.0], times_arr[times_arr <= RISK_HORIZON], [RISK_HORIZON]))
        probs_for_integ = []
        for t in times_for_integ:
            if t <= times_arr[0]:
                probs_for_integ.append(1.0)
            else:
                idx = np.searchsorted(times_arr, t, side='right') - 1
                idx = max(0, min(idx, len(probs)-1))
                probs_for_integ.append(probs[idx])
        # trapz integration
        rmst = 0.0
        for k in range(len(times_for_integ)-1):
            dt = times_for_integ[k+1] - times_for_integ[k]
            rmst += probs_for_integ[k] * dt
        expected_times.append(float(rmst))

    # train surrogate regressor for risk-within-T
    y_risk = np.array(risks)
    surrogate = HistGradientBoostingRegressor(max_iter=200, random_state=42)
    pipeline_surrogate = Pipeline([("pre", preprocessor), ("reg", surrogate)])
    print(f"Training surrogate regressor for stage {stage}...")
    pipeline_surrogate.fit(X_stage, y_risk)
    joblib.dump(pipeline_surrogate, f"models/{stage}_surrogate.joblib")
    print(f"Saved models/{stage}_surrogate.joblib")

    # persist small metadata (optional)
    meta = {"n_train": len(X_stage), "risk_horizon": RISK_HORIZON}
    joblib.dump(meta, f"models/{stage}_meta.joblib")

print("All stages trained and surrogate models saved.")
