"""
explainability.py

Pillar 1b: Explainability for survival models using SHAP surrogates.

Since SHAP doesn't work directly on survival curves, we:
1. Train a surrogate regression model per stage that predicts risk-within-horizon from features
2. Run TreeExplainer on the surrogate to get feature importance + SHAP values
3. Return ranked explanation for why a project's risk is high

This gives interpretable answers: "Project X has high risk because:
  - Legal risk is high (contributes +0.35 to risk score)
  - Collector capacity is low (contributes +0.22 to risk score)
  - Pending projects is high (contributes +0.18 to risk score)"
"""

import os
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sksurv.util import Surv
import shap


class SurvivalExplainer:
    """Train surrogates and compute SHAP explanations for survival models."""
    
    def __init__(self, models_dir="models", data_dir="data"):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.surrogates = {}  # {stage: Pipeline with surrogate GB model}
        self.explainers = {}  # {stage: shap.TreeExplainer}
        self.feature_names = None
        self.stages = ["notification", "award", "compensation", "rnr", "possession"]
    
    def train_surrogates(self, projects_df, legal_df=None, risk_horizon=180, force_retrain=False):
        """
        Train surrogate regression models for each stage.
        
        Each surrogate predicts: risk_within_horizon = 1 - S(T) from features.
        
        Args:
            projects_df: Projects DataFrame
            legal_df: Legal events DataFrame (optional, for enrichment)
            risk_horizon: Horizon in days for risk computation
            force_retrain: If False, try to load from disk first
        
        Returns:
            dict {stage: pipeline}
        """
        from legal_risk import compute_project_legal_risk
        
        surrogates = {}
        
        # Load or train per-stage survival models
        rsf_models = {}
        for stage in self.stages:
            path = os.path.join(self.models_dir, f"{stage}_rsf.joblib")
            if os.path.exists(path):
                rsf_models[stage] = joblib.load(path)
        
        if not rsf_models:
            raise FileNotFoundError(f"No RSF models found in {self.models_dir}. Train them first.")
        
        # Enrich projects with legal risk if provided
        projects = projects_df.copy()
        if legal_df is not None and not legal_df.empty:
            project_legal_risk = compute_project_legal_risk(legal_df)
            projects["legal_risk_score"] = projects["project_id"].map(project_legal_risk).fillna(0)
        else:
            projects["legal_risk_score"] = projects["legal_risk"]
        
        cat_features = ["district", "project_type", "land_category"]
        num_features = [
            "area_ha", "affected_families", "collector_capacity", 
            "verification_teams", "pending_projects", "legal_risk_score"
        ]
        all_features = cat_features + num_features
        self.feature_names = all_features
        
        # Train surrogate for each stage
        for stage in self.stages:
            if stage not in rsf_models:
                continue
            
            time_col = f"duration_{stage}"
            event_col = f"event_{stage}"
            
            # Filter to projects that entered this stage
            mask = projects[f"{stage}_entry"] != ""
            X_stage = projects.loc[mask, all_features].reset_index(drop=True)
            times = projects.loc[mask, time_col].to_numpy(dtype=float)
            events = projects.loc[mask, event_col].to_numpy(dtype=bool)
            
            if len(X_stage) < 10:
                print(f"Skipping {stage}: too few samples ({len(X_stage)})")
                continue
            
            # Compute risk labels: 1 - S(risk_horizon) for each project
            rsf = rsf_models[stage]
            try:
                surv_funcs = rsf.predict_survival_function(X_stage)
                risks = []
                for fn in surv_funcs:
                    times_fn = np.array(fn.x) if hasattr(fn, 'x') else np.array([t for t, _ in fn])
                    probs = np.array(fn.y) if hasattr(fn, 'y') else np.array([p for _, p in fn])
                    
                    if risk_horizon <= times_fn[0]:
                        S_h = 1.0
                    else:
                        idx = np.searchsorted(times_fn, risk_horizon, side='right') - 1
                        idx = max(0, min(idx, len(probs) - 1))
                        S_h = probs[idx]
                    risks.append(max(0.0, min(1.0, 1.0 - S_h)))
                
                y_surrogate = np.array(risks)
            except Exception as e:
                print(f"Error predicting survival for {stage}: {e}")
                continue
            
            # Train surrogate: GB regressor on (X, y_risk)
            preprocessor = ColumnTransformer([
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
            ], remainder="passthrough")
            
            gb = GradientBoostingRegressor(
                n_estimators=50, 
                max_depth=3, 
                learning_rate=0.1,
                random_state=42
            )
            
            surrogate_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", gb)
            ])
            
            print(f"Training surrogate for {stage} on {len(X_stage)} samples...")
            surrogate_pipeline.fit(X_stage, y_surrogate)
            surrogates[stage] = surrogate_pipeline
            
            # Create SHAP explainer
            try:
                gb_model = surrogate_pipeline.named_steps['model']
                explainer = shap.TreeExplainer(gb_model)
                self.explainers[stage] = explainer
            except Exception as e:
                print(f"Could not create SHAP explainer for {stage}: {e}")
        
        self.surrogates = surrogates
        self.save_surrogates()
        return surrogates
    
    def explain_project(self, project_row, stage):
        """
        Generate SHAP-based explanation for a project at a given stage.
        
        Args:
            project_row: Single row from projects DataFrame
            stage: Stage name
        
        Returns:
            dict with SHAP values and ranked feature contributions
        """
        if stage not in self.surrogates:
            return {"error": f"No surrogate for stage {stage}"}
        
        surrogate = self.surrogates[stage]
        explainer = self.explainers.get(stage)
        
        cat_features = ["district", "project_type", "land_category"]
        num_features = [
            "area_ha", "affected_families", "collector_capacity",
            "verification_teams", "pending_projects", "legal_risk_score"
        ]
        all_features = cat_features + num_features
        
        # Build X for this project
        X = pd.DataFrame([project_row[all_features]])
        
        # Predict and explain
        try:
            pred = surrogate.predict(X)[0]
            
            if explainer is not None:
                # Get transformed features for SHAP
                preprocessor = surrogate.named_steps['preprocessor']
                X_transformed = preprocessor.transform(X)
                
                # Compute SHAP values
                shap_values = explainer.shap_values(X_transformed)
                
                # Get feature names after transformation
                ohe = preprocessor.named_transformers_['ohe']
                ohe_names = list(ohe.get_feature_names_out(cat_features))
                feature_names_transformed = ohe_names + num_features
                
                # Aggregate SHAP contributions (handle multi-output if needed)
                if isinstance(shap_values, list):
                    shap_vals = shap_values[0] if len(shap_values) > 0 else shap_values
                else:
                    shap_vals = shap_values
                
                # Rank contributions by absolute value
                contributions = pd.DataFrame({
                    "feature": feature_names_transformed,
                    "shap_value": shap_vals[0] if shap_vals.ndim > 1 else shap_vals,
                    "abs_impact": np.abs(shap_vals[0] if shap_vals.ndim > 1 else shap_vals)
                }).sort_values("abs_impact", ascending=False)
                
                return {
                    "stage": stage,
                    "predicted_risk": float(pred),
                    "top_contributors": contributions.head(10).to_dict("records"),
                    "base_value": float(explainer.expected_value) if hasattr(explainer, 'expected_value') else None
                }
            else:
                return {
                    "stage": stage,
                    "predicted_risk": float(pred),
                    "error": "SHAP explainer not available for this stage"
                }
        
        except Exception as e:
            return {
                "stage": stage,
                "error": f"Could not explain: {str(e)}"
            }
    
    def save_surrogates(self, path="models/surrogates.joblib"):
        """Save surrogates and explainers to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "surrogates": self.surrogates,
            "explainers": self.explainers,
            "feature_names": self.feature_names
        }, path)
        print(f"Saved surrogates to {path}")
    
    def load_surrogates(self, path="models/surrogates.joblib"):
        """Load surrogates from disk."""
        if os.path.exists(path):
            data = joblib.load(path)
            self.surrogates = data["surrogates"]
            self.explainers = data["explainers"]
            self.feature_names = data["feature_names"]
            print(f"Loaded surrogates from {path}")
            return True
        return False


if __name__ == "__main__":
    # Example usage
    projects_df = pd.read_csv("data/projects.csv")
    
    try:
        legal_df = pd.read_csv("data/legal_events.csv")
    except:
        legal_df = None
    
    explainer = SurvivalExplainer()
    
    # Train surrogates
    print("Training SHAP surrogates...")
    explainer.train_surrogates(projects_df, legal_df, risk_horizon=180)
    
    # Explain a random project
    if len(projects_df) > 0:
        test_proj = projects_df.iloc[0]
        for stage in ["notification", "award"]:
            print(f"\n=== Explanation for {test_proj['project_id']}, stage={stage} ===")
            explanation = explainer.explain_project(test_proj, stage)
            print(explanation)
