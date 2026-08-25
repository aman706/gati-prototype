"""
train_survival_models.py

Trains a RandomSurvivalForest per stage using scikit-survival.
Saves models to models/<stage>_rsf.joblib
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

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/projects.csv")
STAGES = ["notification", "award", "compensation", "rnr", "possession"]

# features to use
cat_features = ["district", "project_type", "land_category"]
num_features = ["area_ha", "affected_families", "collector_capacity", "verification_teams", "pending_projects", "legal_risk"]

X = df[cat_features + num_features].copy()

for stage in STAGES:
    time_col = f"duration_{stage}"
    event_col = f"event_{stage}"
    # filter rows where entry exists (some later stages have blank entry)
    mask = df[f"{stage}_entry"] != ""
    X_stage = X[mask].reset_index(drop=True)
    times = df.loc[mask, time_col].to_numpy(dtype=float)
    events = df.loc[mask, event_col].to_numpy(dtype=bool)

    # build y structured array
    y = Surv.from_arrays(events, times)

    # pipeline: OHE for categorical features then RSF
    preprocessor = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_features)
    ], remainder="passthrough")

    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=5, random_state=42)
    pipeline = Pipeline([("pre", preprocessor), ("rsf", rsf)])

    print(f"Training RSF for stage {stage} on {len(X_stage)} rows...")
    pipeline.fit(X_stage, y)
    joblib.dump(pipeline, f"models/{stage}_rsf.joblib")
    print(f"Saved models/{stage}_rsf.joblib")

print("Training complete.")
