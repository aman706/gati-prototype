"""
app_survival.py

Streamlit app to inspect per-stage survival functions and run simple counterfactuals.
"""
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import json
import os

st.set_page_config(layout="wide", page_title="GATI - Survival Prototype")

@st.cache_data
def load_data():
    df = pd.read_csv("data/projects.csv")
    return df

@st.cache_resource
def load_models():
    stages = ["notification","award","compensation","rnr","possession"]
    models = {}
    for s in stages:
        path = f"models/{s}_rsf.joblib"
        if os.path.exists(path):
            models[s] = joblib.load(path)
    return models

@st.cache_data
def load_districts():
    with open("data/districts.json","r") as f:
        return json.load(f)

df = load_data()
models = load_models()
districts = load_districts()

st.title("GATI — Survival-model prototype")

# sidebar
st.sidebar.header("Controls")
selected_district = st.sidebar.selectbox("District", options=["All"] + sorted(df["district"].unique().tolist()))
selected_stage = st.sidebar.selectbox("Stage to inspect", options=list(models.keys()))
risk_horizon = st.sidebar.slider("Risk horizon (days)", 30, 730, 180)

# filter
if selected_district != "All":
    df_view = df[df["district"] == selected_district].copy()
else:
    df_view = df.copy()

st.header("District-level summary")
# compute mean risk-within-T for each district using model predictions
agg_rows = []
for d, group in df_view.groupby("district"):
    row = {"district": d, "n_projects": len(group)}
    for s, m in models.items():
        # build X for this model's features
        X = group[["district","project_type","land_category","area_ha","affected_families","collector_capacity","verification_teams","pending_projects","legal_risk"]]
        try:
            surv_funcs = m.predict_survival_function(X)
            # each surv_func is a (times, surv) step function represented as array of (time, prob)
            # compute risk within horizon as mean(1 - S(h))
            risks = []
            for fn in surv_funcs:
                # fn is an array-like of (time, prob) in older API; for pipeline we might get list of step arrays
                times = np.array(fn.x) if hasattr(fn, 'x') else np.array([t for t,_ in fn])
                probs = np.array(fn.y) if hasattr(fn, 'y') else np.array([p for _,p in fn])
                # find S(h)
                if risk_horizon <= times[0]:
                    S_h = 1.0
                else:
                    idx = np.searchsorted(times, risk_horizon, side='right') - 1
                    idx = max(0, min(idx, len(probs)-1))
                    S_h = probs[idx]
                risks.append(1.0 - S_h)
            row[f"risk_{s}"] = float(np.mean(risks))
        except Exception:
            row[f"risk_{s}"] = None
    agg_rows.append(row)
agg = pd.DataFrame(agg_rows).sort_values(f"risk_{selected_stage}", ascending=False)
st.dataframe(agg, use_container_width=True)

st.header("Project list and per-project view")
st.dataframe(df_view[["project_id","district","project_type","land_category","area_ha","affected_families","collector_capacity","pending_projects"]].head(200))

proj = st.selectbox("Select project", options=sorted(df_view["project_id"].tolist()))
row = df[df["project_id"] == proj].iloc[0]
st.subheader("Project profile")
st.write(row[["district","project_type","land_category","area_ha","affected_families","collector_capacity","pending_projects","legal_risk"]])

st.subheader("Per-stage risk and survival curve")
col1, col2 = st.columns([1,2])
with col1:
    st.write("Original features")
    st.write(row[["collector_capacity","pending_projects"]])
    # counterfactual sliders
    cf_capacity = st.slider("collector_capacity", min_value=1, max_value=20, value=int(row["collector_capacity"]))
    cf_pending = st.slider("pending_projects", min_value=0, max_value=20, value=int(row["pending_projects"]))
    apply_cf = st.button("Apply counterfactual")

with col2:
    fig, ax = plt.subplots(figsize=(6,4))
    T = np.linspace(0, risk_horizon, 100)
    for s, m in models.items():
        # build X for single project
        X = pd.DataFrame([row[["district","project_type","land_category","area_ha","affected_families","collector_capacity","verification_teams","pending_projects","legal_risk"]]])
        if apply_cf:
            X.loc[0, "collector_capacity"] = cf_capacity
            X.loc[0, "pending_projects"] = cf_pending
        try:
            fn = m.predict_survival_function(X)[0]
            times = np.array(fn.x) if hasattr(fn, 'x') else np.array([t for t,_ in fn])
            probs = np.array(fn.y) if hasattr(fn, 'y') else np.array([p for _,p in fn])
            # plot stepped survival
            ax.step(times, probs, where='post', label=s)
        except Exception as e:
            # fallback: skip
            pass
    ax.set_xlabel("Days since stage entry")
    ax.set_ylabel("Survival probability S(t)")
    ax.legend()
    st.pyplot(fig)

# show derived risk within horizon
st.subheader(f"Risk within {risk_horizon} days per stage (1 - S({risk_horizon}))")
risk_table = {}
for s, m in models.items():
    X = pd.DataFrame([row[["district","project_type","land_category","area_ha","affected_families","collector_capacity","verification_teams","pending_projects","legal_risk"]]])
    if apply_cf:
        X.loc[0, "collector_capacity"] = cf_capacity
        X.loc[0, "pending_projects"] = cf_pending
    try:
        fn = m.predict_survival_function(X)[0]
        times = np.array(fn.x) if hasattr(fn, 'x') else np.array([t for t,_ in fn])
        probs = np.array(fn.y) if hasattr(fn, 'y') else np.array([p for _,p in fn])
        if risk_horizon <= times[0]:
            S_h = 1.0
        else:
            idx = np.searchsorted(times, risk_horizon, side='right') - 1
            idx = max(0, min(idx, len(probs)-1))
            S_h = probs[idx]
        risk_table[s] = float(1.0 - S_h)
    except Exception:
        risk_table[s] = None

st.table(pd.DataFrame.from_dict(risk_table, orient='index', columns=['risk_within_T']))

st.markdown("Note: feature explainability for RSF is shown as variable importance (mean decrease in impurity proxy). For production we recommend SHAP-like surrogates or PDP/ICE visualizations.")
# show feature importances for the selected stage
st.subheader(f"Feature importances (stage: {selected_stage})")
if selected_stage in models:
    try:
        model = models[selected_stage]
        rsf = model.named_steps['rsf']
        # if OneHot encoding expanded features, importances align to transformed features; show raw numerical insides as proxy
        importances = rsf.feature_importances_
        # build feature names
        pre = model.named_steps['pre']
        ohe = pre.named_transformers_['ohe']
        cat_names = ohe.get_feature_names_out(["district","project_type","land_category"]).tolist()
        feature_names = list(cat_names) + ["area_ha","affected_families","collector_capacity","verification_teams","pending_projects","legal_risk"]
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(20)
        st.bar_chart(imp_df.set_index('feature')['importance'])
    except Exception as e:
        st.write("Could not compute importances:", e)
else:
    st.write("No model for selected stage loaded.")

