"""
district_graph.py

Pillar 3: Cross-project contention graph.

Builds a directed graph where:
- Nodes = districts, with attributes: collector_capacity, verification_teams, active_projects, legal_risk
- Edges = resource dependencies (future: which districts send appeals/escalations to others)
- Scoring: "load" = active_projects / collector_capacity (flag over-capacity)

Provides:
1. Per-district contention rank (load + legal risk normalized)
2. What-if functions to simulate capacity/resource changes
"""

import pandas as pd
import numpy as np
import networkx as nx
import json
import os


class DistrictContentionGraph:
    """Graph-based district contention analysis."""
    
    def __init__(self, projects_df, districts_json_path="data/districts.json", legal_metrics=None):
        """
        Initialize the graph from projects and district metadata.
        
        Args:
            projects_df: Projects DataFrame
            districts_json_path: Path to districts.json with capacity info
            legal_metrics: District legal metrics (from legal_risk.py)
        """
        self.projects_df = projects_df
        self.legal_metrics = legal_metrics or pd.DataFrame()
        self.G = nx.DiGraph()
        
        # Load district capacity data
        if os.path.exists(districts_json_path):
            with open(districts_json_path, "r") as f:
                self.district_data = json.load(f)
        else:
            self.district_data = {}
        
        self._build_graph()
    
    def _build_graph(self):
        """Build district graph with nodes and attributes."""
        # Group projects by district
        district_agg = self.projects_df.groupby("district").agg({
            "project_id": "count",
            "legal_risk": "mean",
            "affected_families": "sum",
            "area_ha": "sum"
        }).rename(columns={"project_id": "active_projects"})
        
        for district in self.projects_df["district"].unique():
            self.G.add_node(district)
            
            # Set node attributes
            stats = district_agg.loc[district]
            capacity_data = self.district_data.get(district, {})
            
            collector_capacity = capacity_data.get("collector_capacity", 8)
            verification_teams = capacity_data.get("verification_teams", 2)
            active_projects = int(stats["active_projects"])
            legal_risk = float(stats["legal_risk"])
            
            # Load score: projects / capacity (flagged if > 1)
            load_score = active_projects / max(1, collector_capacity)
            
            # Legal risk from district metrics
            if not self.legal_metrics.empty and district in self.legal_metrics.index:
                district_legal_risk = float(self.legal_metrics.loc[district, "mean_legal_risk"])
            else:
                district_legal_risk = legal_risk
            
            self.G.nodes[district].update({
                "collector_capacity": collector_capacity,
                "verification_teams": verification_teams,
                "active_projects": active_projects,
                "legal_risk": district_legal_risk,
                "affected_families": int(stats["affected_families"]),
                "load_score": load_score,
                "over_capacity": load_score > 1.0
            })
        
        # Add simple edges: if district A has many pending projects, it might "pull" resources
        # For now, create edges to highest-capacity neighbor (future: escalation patterns)
        districts = list(self.G.nodes())
        for d in districts:
            if len(districts) > 1:
                # Light edge to next district in sorted order (placeholder for escalation)
                idx = districts.index(d)
                next_d = districts[(idx + 1) % len(districts)]
                self.G.add_edge(d, next_d, weight=0.5)
    
    def contention_rank(self, normalize=True):
        """
        Rank districts by contention: load_score + legal_risk (normalized).
        
        Args:
            normalize: If True, scales to [0, 1]
        
        Returns:
            Series indexed by district, sorted descending by contention
        """
        scores = {}
        for node in self.G.nodes():
            load = self.G.nodes[node]["load_score"]
            legal = self.G.nodes[node]["legal_risk"]
            # Simple weighting: 60% load, 40% legal risk
            contention = 0.6 * load + 0.4 * legal
            scores[node] = contention
        
        rank = pd.Series(scores).sort_values(ascending=False)
        
        if normalize and rank.max() > 0:
            rank = rank / rank.max()
        
        return rank
    
    def bottleneck_districts(self, load_threshold=1.0, top_n=5):
        """
        Identify bottleneck districts: load > threshold, ranked by legal risk.
        
        Args:
            load_threshold: Flag districts with load >= this value
            top_n: Return top N bottlenecks
        
        Returns:
            DataFrame with district stats, sorted by contention
        """
        bottlenecks = []
        for node in self.G.nodes():
            if self.G.nodes[node]["load_score"] >= load_threshold:
                bottlenecks.append({
                    "district": node,
                    "collector_capacity": self.G.nodes[node]["collector_capacity"],
                    "verification_teams": self.G.nodes[node]["verification_teams"],
                    "active_projects": self.G.nodes[node]["active_projects"],
                    "load_score": self.G.nodes[node]["load_score"],
                    "legal_risk": self.G.nodes[node]["legal_risk"],
                    "affected_families": self.G.nodes[node]["affected_families"]
                })
        
        df = pd.DataFrame(bottlenecks).sort_values(
            ["load_score", "legal_risk"], ascending=False
        ).head(top_n)
        
        return df
    
    def simulate_capacity_change(self, district, new_capacity):
        """
        Simulate what happens if a district's collector capacity changes.
        Returns updated load_score and contention rank.
        
        Args:
            district: District name
            new_capacity: New collector capacity value
        
        Returns:
            dict with old/new load, new contention rank
        """
        if district not in self.G.nodes():
            return None
        
        old_capacity = self.G.nodes[district]["collector_capacity"]
        old_load = self.G.nodes[district]["load_score"]
        
        active_projects = self.G.nodes[district]["active_projects"]
        new_load = active_projects / max(1, new_capacity)
        
        # Update node
        self.G.nodes[district]["collector_capacity"] = new_capacity
        self.G.nodes[district]["load_score"] = new_load
        self.G.nodes[district]["over_capacity"] = new_load > 1.0
        
        # Recompute contention
        new_rank = self.contention_rank()
        
        return {
            "district": district,
            "old_capacity": old_capacity,
            "new_capacity": new_capacity,
            "old_load": old_load,
            "new_load": new_load,
            "new_contention_rank": new_rank[district],
            "rank_position": list(new_rank.index).index(district) + 1
        }
    
    def simulate_project_reduction(self, district, reduction_count):
        """
        Simulate reducing active projects in a district.
        
        Args:
            district: District name
            reduction_count: Number of projects to reduce
        
        Returns:
            dict with impact on load and contention
        """
        if district not in self.G.nodes():
            return None
        
        old_active = self.G.nodes[district]["active_projects"]
        new_active = max(0, old_active - reduction_count)
        
        capacity = self.G.nodes[district]["collector_capacity"]
        old_load = self.G.nodes[district]["load_score"]
        new_load = new_active / max(1, capacity)
        
        self.G.nodes[district]["active_projects"] = new_active
        self.G.nodes[district]["load_score"] = new_load
        self.G.nodes[district]["over_capacity"] = new_load > 1.0
        
        new_rank = self.contention_rank()
        
        return {
            "district": district,
            "old_active_projects": old_active,
            "new_active_projects": new_active,
            "old_load": old_load,
            "new_load": new_load,
            "new_contention_rank": new_rank[district],
            "rank_position": list(new_rank.index).index(district) + 1
        }
    
    def to_dataframe(self):
        """Convert graph nodes to DataFrame for display."""
        rows = []
        for node in self.G.nodes():
            rows.append({
                "district": node,
                **self.G.nodes[node]
            })
        return pd.DataFrame(rows).sort_values("load_score", ascending=False)


if __name__ == "__main__":
    # Example usage
    projects_df = pd.read_csv("data/projects.csv")
    
    graph = DistrictContentionGraph(projects_df)
    
    print("\n=== District Contention Ranking ===")
    rank = graph.contention_rank()
    print(rank.head(10))
    
    print("\n=== Bottleneck Districts (load > 1.0) ===")
    bottlenecks = graph.bottleneck_districts(load_threshold=1.0, top_n=10)
    print(bottlenecks)
    
    print("\n=== Simulate: increase capacity in bottleneck district ===")
    if len(bottlenecks) > 0:
        test_district = bottlenecks.iloc[0]["district"]
        result = graph.simulate_capacity_change(test_district, new_capacity=15)
        print(result)
