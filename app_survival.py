"""
app_survival.py (updated)

Adds:
- Expected time (restricted mean up to horizon) + delay probability display
- SHAP explainability using surrogate regressor per stage
- Case timeline visualization from data/legal_events.csv
- What-if simulator: feature adjustments + "expedite stage by X days" (shifts survival curve)
- RFCTLARR stage mapping labels
"""
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import shap

st.set_page_config(layout="wide", page_title="GATI - Survival Prototype (enhanced)")

@st.cache_data
def load_data():
    df = pd.read_csv("data/projects.csv")
    legal = pd.read_csv("data/legal_events.csv")
    return df, legal

@st.cache_resource
def load_models_and_surrogates():
    stages = ["notification","award","compensation","rnr","possession"]
    rsf_models = {}
    surrogates = {}
    metas = {}
    for s in stages:
        p_rsf = f"models/{s}_rsf.joblib"
        p_sur = f"models/{s}_surrogate.joblib"
        p_meta = f"models/{s}_meta.joblib"
        if os.path.exists(p_rsf):
            rsf_models[s] = joblib.load(p_rsf)
        if os.path.exists(p_sur):
            surrogates[s] = joblib.load(p_sur)
        if os.path.exists(p_meta):
            metas[s] = joblib.load(p_meta)
    return rsf_models, surrogates, metas

STAGE_LABELS = {
    "notification": "Notification (Section: Statutory notification)",
    "award": "Award (Determination of acquisition)",
    "compensation": "Compensation (Determination & disbursement)",
    "rnr": "R&R (Rehabilitation & Resettlement)",
    "possession": "Possession (Physical takeover)"
}

RISK_HORIZON = 180

(df, legal_df) = load_data()
rsf_models, surrogates, metas = load_models_and_surrogates()

st.title("GATI — Survival-model prototype (enhanced)")

# sidebar controls
st.sidebar.header("Controls")
selected_district = st.sidebar.selectbox("District", options=["All"] + sorted(df["district"].unique().tolist()))
selected_stage = st.sidebar.selectbox("Stage to inspect", options=list(rsf_models.keys()))
risk_horizon = st.sidebar.slider("Risk horizon (days)", 30, 730, RISK_HORIZON)

# filter dataset
if selected_district != "All":
    df_view = df[df["district"] == selected_district].copy()
else:
    df_view = df.copy()

st.header("Dashboard: district risk overview")
# compute mean risk-within-T for each district using surrogate if available, else RSF
agg = []
for d, group in df_view.groupby("district"):
    row = {"district": d, "n_projects": len(group)}
    for s in rsf_models.keys():
        try:
            if s in surrogates:
                X = group[["district","project_type","land_category","area_ha","affected_families","collector_capacity","verification_teams","pending_projects","legal_risk"]]
                preds = surrogates[s].predict(X)
                risk = float(np.nanmean(preds))
            else:
                # fallback: use RSF survival functions and compute mean risk
                X = group[["district","project_type","land_category","area_ha","affected_families","collector_capacity","verification_teams","pending_projects","legal_risk"]]
                survs = rsf_models[s].predict_survival_function(X)
                risks = []
                for fn in survs:
                    times_arr = np.array(fn.x) if hasattr(fn, 'x') else np.array([t for t,_ in fn])
                    probs = np.array(fn.y) if hasattr(fn, 'y') else np.array([p for _,p in fn])
                    if risk_horizon <= times_arr[0]:
                        S_h = 1.0
                    else:
                        idx = np.searchsorted(times_arr, risk_horizon, side='right') - 1
                        idx = max(0, min(idx, len(probs)-1))
                        S_h = probs[idx]
                    risks.append(1.0 - S_h)
                risk = float(np.mean(risks))
            row[f"risk_{s}"] = risk
        except Exception:
            row[f"risk_{s}"] = None
    agg.append(row)
agg_df = pd.DataFrame(agg).sort_values(f"risk_{selected_stage}", ascending=False)
st.dataframe(agg_df, use_container_width=True)

# show a simple national heat-like bar for selected stage
st.subheader(f"Top districts by risk (stage: {STAGE_LABELS.get(selected_stage, selected_stage)})")
fig, ax = plt.subplots(figsize=(10,4))
top = agg_df.head(12)
ax.bar(top['district'], top[f"risk_{selected_stage}"], color='C3')
ax.set_xticklabels(top['district'], rotation=45, ha='right')
ax.set_ylabel(f"Mean risk within {risk_horizon} days")
st.pyplot(fig)

# project list
st.header("Project list (district drill-down)")
st.dataframe(df_view[["project_id","district","project_type","land_category","area_ha","affected_families","collector_capacity","pending_projects"]].sort_values("pending_projects", ascending=False).head(300))

# project-detail panel
st.header("Project detail, timeline, explainability, and what-if simulator")
proj = st.selectbox("Select project", options=sorted(df_view["project_id"].tolist()))
row = df[df["project_id"] == proj].iloc[0]
st.subheader("Project profile")
st.write(row[["district","project_type","land_category","area_ha","affected_families","collector_capacity","pending_projects","legal_risk"]])

# timeline: stages + legal events
st.subheader("Case timeline (stages + legal events)")
# build stage points
stages = ["notification","award","compensation","rnr","possession"]
stage_points = []
for s in stages:
    ent = row.get(f"{s}_entry", "")
    comp = row.get(f"{s}_completion", "")
    if ent:
        stage_points.append((s, ent, 'entry'))
    if comp:
        stage_points.append((s, comp, 'completion'))

# legal events for project
proj_legal = legal_df[legal_df['project_id'] == proj].sort_values('date')

# plot timeline
fig2, ax2 = plt.subplots(figsize=(10,2))
y = 0
for s, date_str, typ in stage_points:
    try:
        dt = pd.to_datetime(date_str)
        ax2.plot([dt], [y], marker='o', label=f"{s}:{typ}")
        ax2.text(dt, y+0.02, f"{s[:3]}-{typ[0]}", rotation=45)
    except Exception:
        pass
for idx, ev in proj_legal.iterrows():
    dt = pd.to_datetime(ev['date'])
    ax2.plot([dt], [y-0.05], marker='x', color='red')
    ax2.text(dt, y-0.1, f"{ev['event_type']}({ev['severity']})", rotation=45, color='red')
ax2.get_yaxis().set_visible(False)
ax2.set_xlabel('Date')
st.pyplot(fig2)

# predicted survival curves, expected time, and risk-within-T per stage
st.subheader("Per-stage survival, expected time (RMST up to horizon) & risk-within-T")
col1, col2 = st.columns([1,1])

# what-if controls
with col1:
    st.markdown("### What-if: feature adjustments")
    cf_capacity = st.slider("collector_capacity", 1, 20, int(row['collector_capacity']))
    cf_pending = st.slider("pending_projects", 0, 20, int(row['pending_projects']))
    st.markdown("### Stage-level intervention")
    intervene_stage = st.selectbox("Expedite stage (shift survival curve by -X days)", options=['None'] + stages)
    expedite_days = st.slider("Expedite days (reduce expected duration)", 0, 120, 0)
    apply_cf = st.button("Apply what-if")

with col2:
    fig3, ax3 = plt.subplots(figsize=(6,4))
    for s, m in rsf_models.items():
        # build feature vector
        X = pd.DataFrame([row[["district","project_type","land_category","area_ha","affected_families","collector_capacity","verification_teams","pending_projects","legal_risk"]]])
        X.loc[0, 'collector_capacity'] = cf_capacity
        X.loc[0, 'pending_projects'] = cf_pending
        try:
            fn = m.predict_survival_function(X)[0]
            times = np.array(fn.x) if hasattr(fn, 'x') else np.array([t for t,_ in fn])
            probs = np.array(fn.y) if hasattr(fn, 'y') else np.array([p for _,p in fn])
            # apply expedite shift if applicable (approximation)
            if apply_cf and intervene_stage == s and expedite_days > 0:
                # shifting survival left: S_new(t) = S_old(t + expedite_days)
                times_shifted = times - expedite_days
                times_shifted[times_shifted < 0] = 0
                ax3.step(times_shifted, probs, where='post', label=f"{s} (expedited)")
            else:
                ax3.step(times, probs, where='post', label=s)
        except Exception:
            pass
    ax3.set_xlabel('Days since stage entry')
    ax3.set_ylabel('Survival S(t)')
    ax3.legend()
    st.pyplot(fig3)

# compute expected RMST and risk-within-T table
risk_table = {}
for s, m in rsf_models.items():
    X = pd.DataFrame([row[["district","project_type","land_category","area_ha","affected_families","collector_capacity","verification_teams","pending_projects","legal_risk"]]])
    X.loc[0, 'collector_capacity'] = cf_capacity
    X.loc[0, 'pending_projects'] = cf_pending
    try:
        fn = m.predict_survival_function(X)[0]
        times = np.array(fn.x) if hasattr(fn, 'x') else np.array([t for t,_ in fn])
        probs = np.array(fn.y) if hasattr(fn, 'y') else np.array([p for _,p in fn])
        # compute S(h)
        if risk_horizon <= times[0]:
            S_h = 1.0
        else:
            idx = np.searchsorted(times, risk_horizon, side='right') - 1
            idx = max(0, min(idx, len(probs)-1))
            S_h = probs[idx]
        risk = 1.0 - S_h
        # RMST up to horizon
        times_for_integ = np.concatenate(([0.0], times[times <= risk_horizon], [risk_horizon]))
        probs_for_integ = []
        for t in times_for_integ:
            if t <= times[0]:
                probs_for_integ.append(1.0)
            else:
                idx = np.searchsorted(times, t, side='right') - 1
                idx = max(0, min(idx, len(probs)-1))
                probs_for_integ.append(probs[idx])
        rmst = 0.0
        for k in range(len(times_for_integ)-1):
            dt = times_for_integ[k+1] - times_for_integ[k]
            rmst += probs_for_integ[k] * dt
        # if expedite applied and target stage matches, adjust rmst by subtracting expedite_days (simple approx)
        if apply_cf and intervene_stage == s and expedite_days > 0:
            rmst = max(0.0, rmst - expedite_days)
        risk_table[s] = {"risk_within_T": float(risk), "rmst_up_to_T": float(rmst)}
    except Exception:
        risk_table[s] = {"risk_within_T": None, "rmst_up_to_T": None}

st.table(pd.DataFrame(risk_table).T)

# Explainability via surrogate + SHAP
st.subheader("Explainability (surrogate model for risk-within-T)")
if selected_stage in surrogates:
    sur = surrogates[selected_stage]
    # prepare transformed data for SHAP background (sample)
    X_bg = df[["district","project_type","land_category","area_ha","affected_families","collector_capacity","verification_teams","pending_projects","legal_risk"]].sample(min(200, len(df)))
    # apply CF adjustments to project row
    x_row = pd.DataFrame([row[["district","project_type","land_category","area_ha","affected_families","collector_capacity","verification_teams","pending_projects","legal_risk"]]])
    x_row.loc[0,'collector_capacity'] = cf_capacity
    x_row.loc[0,'pending_projects'] = cf_pending
    try:
        # extract regressor from pipeline
        reg = sur.named_steps['reg']
        pre = sur.named_steps['pre']
        X_bg_trans = pre.transform(X_bg)
        x_row_trans = pre.transform(x_row)
        explainer = shap.Explainer(reg, X_bg_trans)
        shap_values = explainer(x_row_trans)
        st.write("SHAP values (top contributors):")
        # map transformed feature names
        try:
            ohe = pre.named_transformers_['ohe']
            cat_names = ohe.get_feature_names_out(["district","project_type","land_category"]).tolist()
            transformed_names = list(cat_names) + ["area_ha","affected_families","collector_capacity","verification_teams","pending_projects","legal_risk"]
        except Exception:
            transformed_names = [f"f{i}" for i in range(x_row_trans.shape[1])]
        sv = shap_values.values[0]
        df_shap = pd.DataFrame({"feature": transformed_names, "shap": sv})
        df_shap = df_shap.assign(abs_shap=df_shap['shap'].abs()).sort_values('abs_shap', ascending=False).head(12)
        st.table(df_shap[['feature','shap']])
        # small bar chart
        fig4, ax4 = plt.subplots(figsize=(6,3))
        ax4.barh(df_shap['feature'], df_shap['shap'])
        st.pyplot(fig4)
    except Exception as e:
        st.write("Explainability failed:", e)
        st.write("Fallback: model feature importances are shown in the dashboard.")
else:
    st.write("No surrogate explainability model available for this stage.")

st.markdown("""
Notes:
- Stage mapping follows RFCTLARR common terminology: Notification → Award → Compensation → R&R → Possession.
- Expected time shown is a restricted mean up to the selected risk horizon (RMST); production systems should estimate full expected time and uncertainty.
- The expedite intervention is an approximation (shifting the survival curve left). A production-level intervention model should be trained/tested on historical interventions.
""")
