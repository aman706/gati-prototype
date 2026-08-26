"""
train_survival_models.py

Trains a RandomSurvivalForest per stage using scikit-survival.
Saves models to models/<stage>_rsf.joblib

Python 3.14+ compatible with modern type hints and correct sklearn parameter names.
"""
from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/projects.csv")
STAGES = ["notification", "award", "compensation", "rnr", "possession"]

# Type aliases (PEP 695 style)
FeatureList = list[str]
NDArray = np.ndarray[Any, np.dtype[Any]]

# features to use
cat_features: FeatureList = ["district", "project_type", "land_category"]
num_features: FeatureList = [
    "area_ha", "affected_families", "collector_capacity", 
    "verification_teams", "pending_projects", "legal_risk"
]

X = df[cat_features + num_features].copy()

for stage in STAGES:
    time_col = f"duration_{stage}"
    event_col = f"event_{stage}"
    
    # filter rows where entry exists (some later stages have blank entry)
    mask = df[f"{stage}_entry"] != ""
    X_stage = X[mask].reset_index(drop=True)
    times: NDArray = df.loc[mask, time_col].to_numpy(dtype=float)
    events: NDArray = df.loc[mask, event_col].to_numpy(dtype=bool)

    # build y structured array
    y = Surv.from_arrays(events, times)

    # pipeline: OHE for categorical features then RSF
    # FIXED: Use sparse_output instead of deprecated sparse parameter (scikit-learn 1.2+)
    preprocessor = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ], remainder="passthrough")

    rsf: RandomSurvivalForest = RandomSurvivalForest(
        n_estimators=100, 
        min_samples_split=10, 
        min_samples_leaf=5, 
        random_state=42
    )
    pipeline: Pipeline = Pipeline([("pre", preprocessor), ("rsf", rsf)])

    print(f"Training RSF for stage {stage} on {len(X_stage)} rows...")
    pipeline.fit(X_stage, y)
    joblib.dump(pipeline, f"models/{stage}_rsf.joblib")
    print(f"Saved models/{stage}_rsf.joblib")

print("Training complete.")
