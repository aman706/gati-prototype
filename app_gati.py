"""
app_gati.py

Unified GATI Streamlit dashboard (Python 3.14+ compatible).
Integrates all four pillars:
  1. Per-stage survival models (stage-aware hazard modeling)
  2. Legal dispute risk analysis
  3. District contention graph (network-based bottleneck analysis)
  4. What-if simulators (project-level and district-level)

Tabs:
  - National Overview: top-risk projects, district rankings
  - Project Deep-Dive: survival curves + SHAP explanation + legal risk
  - District Contention Graph: bottleneck districts, network visualization
  - What-If Simulator: counterfactual scenarios at project & district level
"""
from __future__ import annotations

import os
from typing import Optional, Any

import joblib
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

from legal_risk import (
    load_legal_events, 
    compute_project_legal_risk, 
    compute_district_legal_metrics,
    enrich_projects_with_legal_risk
)
from district_graph import DistrictContentionGraph
from explainability import SurvivalExplainer


# ============================================================================
# PAGE CONFIG & CACHING
# ============================================================================

st.set_page_config(
    layout="wide",
    page_title="GATI — Comprehensive Prototype",
    menu_items={"About": "GATI Pillar-integrated dashboard"}
)

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load projects CSV."""
    return pd.read_csv("data/projects.csv")

@st.cache_resource
def load_models() -> dict[str, Any]:
    """Load trained survival models."""
    stages = ["notification", "award", "compensation", "rnr", "possession"]
    models: dict[str, Any] = {}
    for s in stages:
        path = f"models/{s}_rsf.joblib"
        if os.path.exists(path):
            models[s] = joblib.load(path)
    return models

@st.cache_data
def load_districts() -> dict[str, dict[str, int]]:
    """Load district metadata."""
    with open("data/districts.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_legal_data() -> pd.DataFrame:
    """Load legal events."""
    try:
        return load_legal_events("data/legal_events.csv")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_resource
def setup_graph(projects_df: pd.DataFrame, legal_df: pd.DataFrame) -> DistrictContentionGraph:
    """Setup district contention graph."""
    legal_metrics = compute_district_legal_metrics(legal_df, projects_df)
    return DistrictContentionGraph(projects_df, legal_metrics=legal_metrics)

@st.cache_resource
def setup_explainer() -> SurvivalExplainer:
    """Setup SHAP explainer."""
    explainer = SurvivalExplainer()
    if not explainer.load_surrogates():
        st.warning("Surrogates not found. Run explainability training.")
    return explainer


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_project_risk_at_horizon(
    model: Any, 
    project_row: pd.Series[Any], 
    horizon: int
) -> Optional[float]:
    """Compute 1 - S(horizon) for a project using a model."""
    try:
        X = pd.DataFrame([project_row[[
            "district", "project_type", "land_category", "area_ha",
            "affected_families", "collector_capacity", "verification_teams",
            "pending_projects", "legal_risk"
        ]]])
        fn = model.predict_survival_function(X)[0]
        times = np.array(fn.x) if hasattr(fn, 'x') else np.array([t for t, _ in fn])
        probs = np.array(fn.y) if hasattr(fn, 'y') else np.array([p for _, p in fn])
        
        if horizon <= times[0]:
            return 0.0
        idx = np.searchsorted(times, horizon, side='right') - 1
        idx = max(0, min(idx, len(probs) - 1))
        return float(1.0 - probs[idx])
    except Exception:
        return None


def plot_survival_curve(model: Any, project_row: pd.Series[Any], horizon: int) -> plt.Figure:
    """Plot survival curve for a project."""
    try:
        X = pd.DataFrame([project_row[[
            "district", "project_type", "land_category", "area_ha",
            "affected_families", "collector_capacity", "verification_teams",
            "pending_projects", "legal_risk"
        ]]])
        fn = model.predict_survival_function(X)[0]
        times = np.array(fn.x) if hasattr(fn, 'x') else np.array([t for t, _ in fn])
        probs = np.array(fn.y) if hasattr(fn, 'y') else np.array([p for _, p in fn])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.step(times, probs, where='post', linewidth=2)
        ax.axvline(x=horizon, color='r', linestyle='--', alpha=0.5, label=f'Horizon ({horizon}d)')
        ax.set_xlabel("Days since stage entry")
        ax.set_ylabel("Survival probability S(t)")
        ax.set_title("Survival Curve")
        ax.legend()
        ax.grid(alpha=0.3)
        return fig
    except Exception as e:
        st.error(f"Could not plot: {e}")
        return None


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main app."""
    # Load data
    projects_df = load_data()
    models = load_models()
    districts_meta = load_districts()
    legal_df = load_legal_data()
    
    # Enrich with legal risk
    projects_df = enrich_projects_with_legal_risk(projects_df, legal_df)
    
    graph = setup_graph(projects_df, legal_df)
    explainer = setup_explainer()
    
    st.title("🏗️ GATI — Government Acquisition Delay Intelligence")
    st.markdown("""
    **Four-Pillar Analytics Dashboard**
    1. **Pillar 1:** Per-stage survival models (hazard prediction)
    2. **Pillar 2:** Legal dispute risk scoring & trends
    3. **Pillar 3:** District contention graph (bottleneck analysis)
    4. **Pillar 4:** What-if simulators (counterfactual scenarios)
    """)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 National Overview",
        "🔍 Project Deep-Dive",
        "🌐 District Contention Graph",
        "🎛️ What-If Simulator",
        "📈 Legal Risk Analysis"
    ])
    
    # ========================================================================
    # TAB 1: NATIONAL OVERVIEW
    # ========================================================================
    with tab1:
        st.header("National Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            risk_horizon = st.slider("Risk horizon (days)", 30, 730, 180, key="national_horizon")
        
        with col2:
            selected_stage = st.selectbox("Stage focus", options=list(models.keys()), key="national_stage")
        
        # Top-risk projects
        st.subheader(f"🔴 Top 20 High-Risk Projects (Stage: {selected_stage}, Horizon: {risk_horizon}d)")
        
        risk_scores: list[dict[str, Any]] = []
        for _, proj in projects_df.iterrows():
            if selected_stage in models:
                risk = compute_project_risk_at_horizon(models[selected_stage], proj, risk_horizon)
                if risk is not None:
                    risk_scores.append({
                        "project_id": proj["project_id"],
                        "district": proj["district"],
                        "risk": risk,
                        "legal_risk": proj["legal_risk"],
                        "affected_families": proj["affected_families"]
                    })
        
        if risk_scores:
            risk_df = pd.DataFrame(risk_scores).sort_values("risk", ascending=False).head(20)
            st.dataframe(risk_df, use_container_width=True)
        else:
            st.warning("No risk scores computed.")
        
        # District summary
        st.subheader("🏛️ District-Level Risk Ranking")
        district_rank = graph.contention_rank(normalize=True)
        district_summary = pd.DataFrame({
            "district": district_rank.index,
            "contention_score": district_rank.values
        }).sort_values("contention_score", ascending=False)
        
        fig = px.bar(
            district_summary.head(15),
            x="district",
            y="contention_score",
            title="Top 15 Bottleneck Districts",
            labels={"contention_score": "Contention Score (0-1)"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # TAB 2: PROJECT DEEP-DIVE
    # ========================================================================
    with tab2:
        st.header("Project Deep-Dive: Risk & Explanation")
        
        selected_project = st.selectbox(
            "Select project",
            options=sorted(projects_df["project_id"].tolist()),
            key="project_selector"
        )
        
        proj_row = projects_df[projects_df["project_id"] == selected_project].iloc[0]
        
        # Project profile
        st.subheader("📋 Project Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("District", proj_row["district"])
            st.metric("Project Type", proj_row["project_type"])
        with col2:
            st.metric("Land Category", proj_row["land_category"])
            st.metric("Area (ha)", f"{proj_row['area_ha']:.2f}")
        with col3:
            st.metric("Affected Families", int(proj_row["affected_families"]))
            st.metric("Legal Risk Score", f"{proj_row['legal_risk_score']:.3f}")
        
        # Survival curves
        st.subheader("📈 Per-Stage Survival Curves")
        stage_cols = st.columns(len(models))
        for idx, (stage, model) in enumerate(models.items()):
            with stage_cols[idx]:
                st.write(f"**{stage.capitalize()}**")
                fig = plot_survival_curve(model, proj_row, horizon=180)
                if fig:
                    st.pyplot(fig)
        
        # Risk within horizon
        st.subheader("⚠️ Risk-Within-Horizon (180 days per stage)")
        risk_table: dict[str, Optional[float]] = {}
        for stage, model in models.items():
            risk = compute_project_risk_at_horizon(model, proj_row, 180)
            risk_table[stage] = risk
        
        risk_df = pd.DataFrame.from_dict(risk_table, orient='index', columns=['Risk (1-S)'])
        st.dataframe(risk_df)
        
        # SHAP explanation
        st.subheader("🔬 SHAP Feature Explanation (Pillar 1b)")
        explain_stage = st.selectbox("Stage for explanation", list(models.keys()), key="explain_stage")
        
        if st.button("Generate SHAP Explanation"):
            # Ensure legal_risk_score exists
            if "legal_risk_score" not in proj_row.index:
                proj_row["legal_risk_score"] = proj_row.get("legal_risk", 0)
            
            explanation = explainer.explain_project(proj_row, explain_stage)
            
            if "error" not in explanation:
                st.success(f"Risk @ 180d: {explanation['predicted_risk']:.3f}")
                
                contributors = explanation.get("top_contributors", [])
                if contributors:
                    contrib_df = pd.DataFrame(contributors)[["feature", "shap_value", "abs_impact"]]
                    st.dataframe(contrib_df)
                    
                    fig = px.bar(
                        contrib_df.head(10),
                        x="abs_impact",
                        y="feature",
                        orientation="h",
                        title="Top 10 Contributing Features"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(explanation["error"])
    
    # ========================================================================
    # TAB 3: DISTRICT CONTENTION GRAPH
    # ========================================================================
    with tab3:
        st.header("District Contention Graph (Pillar 3)")
        
        st.subheader("🚨 Bottleneck Districts (Load > 1.0)")
        bottlenecks = graph.bottleneck_districts(load_threshold=1.0, top_n=15)
        
        if len(bottlenecks) > 0:
            st.dataframe(bottlenecks, use_container_width=True)
            
            # Visualization: load vs legal risk
            fig = px.scatter(
                bottlenecks,
                x="load_score",
                y="legal_risk",
                size="active_projects",
                hover_name="district",
                title="Bottleneck Districts: Load vs Legal Risk",
                labels={"load_score": "Load Score (projects/capacity)", "legal_risk": "Legal Risk"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No bottleneck districts detected (all load ≤ 1.0).")
        
        # Full district stats
        st.subheader("📊 All Districts")
        all_districts = graph.to_dataframe()
        st.dataframe(all_districts, use_container_width=True)
    
    # ========================================================================
    # TAB 4: WHAT-IF SIMULATOR
    # ========================================================================
    with tab4:
        st.header("What-If Simulator (Pillar 4)")
        
        simulator_mode = st.radio("Simulation mode", ["Project-Level", "District-Level"])
        
        if simulator_mode == "Project-Level":
            st.subheader("Project-Level Counterfactuals")
            
            proj_id = st.selectbox(
                "Select project",
                options=sorted(projects_df["project_id"].tolist()),
                key="cf_project"
            )
            proj = projects_df[projects_df["project_id"] == proj_id].iloc[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Features:**")
                st.metric("Collector Capacity", int(proj["collector_capacity"]))
                st.metric("Pending Projects", int(proj["pending_projects"]))
            
            with col2:
                st.write("**Counterfactual Adjustments:**")
                cf_capacity = st.slider("New collector capacity", 1, 20, int(proj["collector_capacity"]))
                cf_pending = st.slider("New pending projects", 0, 20, int(proj["pending_projects"]))
            
            if st.button("Simulate"):
                # Create modified row
                proj_cf = proj.copy()
                proj_cf["collector_capacity"] = cf_capacity
                proj_cf["pending_projects"] = cf_pending
                
                st.subheader("Counterfactual Results")
                for stage, model in models.items():
                    orig_risk = compute_project_risk_at_horizon(model, proj, 180)
                    cf_risk = compute_project_risk_at_horizon(model, proj_cf, 180)
                    
                    if orig_risk is not None and cf_risk is not None:
                        delta = cf_risk - orig_risk
                        st.metric(
                            f"{stage.capitalize()} Risk",
                            f"{cf_risk:.3f}",
                            delta=f"{delta:+.3f}"
                        )
        
        else:  # District-Level
            st.subheader("District-Level Counterfactuals")
            
            bottlenecks = graph.bottleneck_districts(load_threshold=0.8, top_n=20)
            
            if len(bottlenecks) > 0:
                district_id = st.selectbox(
                    "Select district",
                    options=bottlenecks["district"].tolist(),
                    key="cf_district"
                )
                
                current_capacity = int(bottlenecks[bottlenecks["district"] == district_id]["collector_capacity"].iloc[0])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Current: {district_id}**")
                    st.metric("Current Capacity", current_capacity)
                
                with col2:
                    st.write("**Simulation**")
                    new_capacity = st.slider("New capacity", 1, 30, current_capacity)
                
                if st.button("Simulate District Change"):
                    # Create a fresh graph copy for simulation
                    graph_sim = DistrictContentionGraph(
                        projects_df, 
                        legal_metrics=compute_district_legal_metrics(legal_df, projects_df)
                    )
                    
                    result = graph_sim.simulate_capacity_change(district_id, new_capacity)
                    
                    if result:
                        st.success("Simulation Result")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Old Load", f"{result['old_load']:.3f}")
                            st.metric("New Load", f"{result['new_load']:.3f}")
                        with col2:
                            st.metric(
                                "Load Change",
                                f"{result['new_load'] - result['old_load']:+.3f}"
                            )
                        with col3:
                            st.metric(
                                "New Rank Position",
                                f"#{result['rank_position']} of {len(graph_sim.G.nodes())}"
                            )
            else:
                st.info("No bottleneck districts. All districts have load ≤ 0.8.")
    
    # ========================================================================
    # TAB 5: LEGAL RISK ANALYSIS
    # ========================================================================
    with tab5:
        st.header("Legal Risk Analysis (Pillar 2)")
        
        if not legal_df.empty:
            st.subheader("📋 Legal Events Overview")
            st.metric("Total Events", len(legal_df))
            
            event_counts = legal_df["event_type"].value_counts()
            fig = px.bar(
                x=event_counts.index,
                y=event_counts.values,
                title="Legal Events by Type",
                labels={"x": "Event Type", "y": "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🔴 Projects with Highest Legal Risk Scores")
            high_risk = projects_df.nlargest(15, "legal_risk_score")[
                ["project_id", "district", "legal_risk_score", "affected_families"]
            ]
            st.dataframe(high_risk, use_container_width=True)
            
            st.subheader("🏛️ Districts with Highest Litigation Density")
            district_metrics = compute_district_legal_metrics(legal_df, projects_df)
            top_districts = district_metrics.nlargest(10, "litigation_density")
            st.dataframe(top_districts, use_container_width=True)
        else:
            st.warning("No legal events data available.")


if __name__ == "__main__":
    main()
