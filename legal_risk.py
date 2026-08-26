"""
legal_risk.py

Pillar 2: Legal-dispute pattern risk scoring.

Computes per-project legal risk scores (weighted by event severity + recency decay)
and per-district litigation density (mean risk + dispute frequency).
Used as input features for survival models and for district contention analysis.

Python 3.10+ compatible with proper type hints.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd


def load_legal_events(filepath: str = "data/legal_events.csv") -> pd.DataFrame:
    """Load legal events CSV."""
    if not os.path.exists(filepath):
        return pd.DataFrame()
    return pd.read_csv(filepath)


def compute_project_legal_risk(
    legal_df: pd.DataFrame, 
    reference_date: Optional[datetime | str] = None
) -> pd.Series:
    """
    Compute per-project legal risk score.
    
    Risk = sum of (severity * recency_decay) for all events in a project.
    Recency decay: events older than 2 years decay exponentially.
    
    Args:
        legal_df: DataFrame with columns [project_id, date, event_type, severity]
        reference_date: date to measure recency from (default: today)
    
    Returns:
        Series indexed by project_id with risk scores [0, 1+]
    """
    if legal_df.empty:
        return pd.Series(dtype=float)
    
    if reference_date is None:
        reference_date = datetime.now()
    elif isinstance(reference_date, str):
        reference_date = datetime.fromisoformat(reference_date)
    
    legal_df = legal_df.copy()
    legal_df["date"] = pd.to_datetime(legal_df["date"])
    
    # recency decay: exp(-days / 730) so at 2 years we're at ~0.37
    legal_df["days_old"] = (reference_date - legal_df["date"]).dt.days.clip(lower=0)
    legal_df["recency_decay"] = np.exp(-legal_df["days_old"] / 730.0)
    
    # event type severity multiplier
    severity_mult: dict[str, float] = {
        "notice": 0.3, 
        "petition": 0.6, 
        "injunction": 1.0, 
        "settlement": 0.8
    }
    legal_df["event_severity"] = legal_df["event_type"].map(severity_mult).fillna(0.5)
    
    # weighted risk per event
    legal_df["weighted_risk"] = (
        legal_df["severity"] * 
        legal_df["recency_decay"] * 
        legal_df["event_severity"]
    )
    
    # aggregate per project
    project_risk: pd.Series = legal_df.groupby("project_id")["weighted_risk"].sum()
    return project_risk


def compute_district_legal_metrics(
    legal_df: pd.DataFrame, 
    projects_df: Optional[pd.DataFrame] = None,
    reference_date: Optional[datetime | str] = None
) -> pd.DataFrame:
    """
    Compute per-district legal metrics: litigation density and mean risk.
    
    Args:
        legal_df: Legal events DataFrame
        projects_df: Projects DataFrame (to normalize by district size)
        reference_date: Reference date for decay
    
    Returns:
        DataFrame indexed by district with columns [litigation_density, mean_legal_risk]
    """
    if legal_df.empty:
        if projects_df is not None:
            return pd.DataFrame(
                {"litigation_density": 0.0, "mean_legal_risk": 0.0},
                index=projects_df["district"].unique()
            )
        return pd.DataFrame()
    
    # count disputes per district
    dispute_counts = legal_df.groupby("district").size()
    
    # project-level risk and merge to projects
    project_risk = compute_project_legal_risk(legal_df, reference_date)
    if projects_df is not None:
        projects_risk = projects_df[["project_id", "district"]].copy()
        projects_risk["legal_risk_score"] = projects_risk["project_id"].map(project_risk).fillna(0)
        mean_risk: pd.Series = projects_risk.groupby("district")["legal_risk_score"].mean()
    else:
        mean_risk = pd.Series(dtype=float)
    
    # Combine metrics
    result = pd.DataFrame(
        {
            "dispute_count": dispute_counts,
            "mean_legal_risk": mean_risk
        }
    ).fillna(0)
    
    # litigation density: disputes per 1000 project-days (proxy: disputes per district)
    if projects_df is not None:
        project_counts = projects_df.groupby("district").size()
        result["litigation_density"] = result["dispute_count"] / (project_counts + 1)
    else:
        result["litigation_density"] = result["dispute_count"]
    
    return result


def enrich_projects_with_legal_risk(
    projects_df: pd.DataFrame, 
    legal_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge project-level legal risk into projects DataFrame.
    Updates or creates 'legal_risk_score' column.
    """
    projects_df = projects_df.copy()
    project_risk = compute_project_legal_risk(legal_df)
    projects_df["legal_risk_score"] = projects_df["project_id"].map(project_risk).fillna(0)
    return projects_df


if __name__ == "__main__":
    # Example usage
    legal_df = load_legal_events()
    projects_df = pd.read_csv("data/projects.csv")
    
    print("\n=== Project-level legal risk (top 10) ===")
    project_risk = compute_project_legal_risk(legal_df)
    print(project_risk.nlargest(10))
    
    print("\n=== District-level legal metrics ===")
    district_metrics = compute_district_legal_metrics(legal_df, projects_df)
    print(district_metrics.sort_values("mean_legal_risk", ascending=False).head(10))
    
    print("\n=== Enriching projects with legal_risk_score ===")
    projects_enriched = enrich_projects_with_legal_risk(projects_df, legal_df)
    print(projects_enriched[["project_id", "legal_risk_score"]].head(10))
