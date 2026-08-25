"""
generate_data_survival.py

Generates a richer synthetic dataset suitable for training per-stage survival models.
Outputs:
 - data/projects.csv  (per-project features + per-stage entry/completion + durations + event flags)
 - data/legal_events.csv (simulated legal events linked to projects)
 - data/districts.json  (optional summary of district resources)

This generator adds censoring (some projects don't complete by dataset end) so we can use survival models.
"""
import json
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

OUT_DIR = "data"

N = 2000
DISTRICTS = [f"District_{i}" for i in range(1, 31)]
PROJECT_TYPES = ["Road", "Rail", "Irrigation", "Industrial", "Urban"]
LAND_CATS = ["Agricultural", "Govt", "Forest", "PrivateCommon"]
STAGES = ["notification", "award", "compensation", "rnr", "possession"]

start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 1, 1)

rows = []
legal_events = []

district_resources = {}
for d in DISTRICTS:
    # simulate district-level resources that will influence contention
    district_resources[d] = {
        "collector_capacity": int(max(3, np.random.normal(8, 2))),
        "verification_teams": int(max(1, np.random.poisson(2))),
        "payroll_cycle_offset_days": int(np.random.randint(0, 30))
    }

for i in range(N):
    pid = f"PRJ_{i+1}"
    district = random.choice(DISTRICTS)
    proj_type = random.choice(PROJECT_TYPES)
    land_cat = random.choice(LAND_CATS)
    area = float(max(0.05, np.random.lognormal(mean=2.0, sigma=1.0)))
    fam = int(max(0, np.random.poisson(6)))

    # district-level state
    dr = district_resources[district]
    # simulate contention as number of other active projects sampled from Poisson with mean related to district size
    pending_projects = int(np.random.poisson(3))
    legal_risk = {"Agricultural": 0.25, "Govt": 0.05, "Forest": 0.45, "PrivateCommon": 0.15}[land_cat]

    # base complexity
    base_complex = np.log1p(area) + (fam * 0.01) + (0.4 if proj_type in ["Rail","Industrial"] else 0.0)

    # simulate stage entry and completion times with delays influenced by features
    # entry time: random offset from start_date
    entry = start_date + timedelta(days=int(np.random.exponential(30)))
    stage_entry = {}
    stage_completion = {}
    durations = {}
    events = {}

    t_cur = entry
    for stage in STAGES:
        stage_entry[stage] = t_cur.date().isoformat()
        # base expected duration in days
        base_days = int(np.random.normal(60, 20))
        # modifiers: contention (+), low collector capacity (+), legal risk (esp. for later stages)
        contention_penalty = int(12 * pending_projects)
        capacity_penalty = int(max(0, 8 - dr["collector_capacity"]) * 6)
        legal_penalty = int(40 * legal_risk) if stage in ["compensation","rnr","possession"] else int(10 * legal_risk)
        stage_noise = int(np.random.normal(0, 10))
        duration = max(1, base_days + contention_penalty + capacity_penalty + legal_penalty + stage_noise)

        # Some projects will be censored (not completed by dataset end). Probability increases for later stages and with contention
        censor_prob = 0.08 + 0.02 * pending_projects + (0.05 if stage in ["rnr","possession"] else 0.0) + legal_risk*0.2
        completed = random.random() > censor_prob and (t_cur + timedelta(days=duration)) < end_date

        if completed:
            t_cur = t_cur + timedelta(days=duration)
            stage_completion[stage] = t_cur.date().isoformat()
            durations[stage] = duration
            events[stage] = 1
        else:
            # censored at dataset end or left open
            stage_completion[stage] = ""
            # censored duration: time from entry to dataset end
            durations[stage] = (end_date - t_cur).days
            events[stage] = 0
            # if censored, subsequent stages are not entered; fill remaining with blanks/censors
            for later in STAGES[STAGES.index(stage)+1:]:
                stage_entry[later] = ""
                stage_completion[later] = ""
                durations[later] = 0
                events[later] = 0
            break
        # small gap to next stage
        t_cur = t_cur + timedelta(days=int(np.random.normal(10, 4)))

    # assemble row
    row = {
        "project_id": pid,
        "district": district,
        "project_type": proj_type,
        "land_category": land_cat,
        "area_ha": round(area,3),
        "affected_families": fam,
        "collector_capacity": dr["collector_capacity"],
        "verification_teams": dr["verification_teams"],
        "payroll_cycle_offset_days": dr["payroll_cycle_offset_days"],
        "pending_projects": pending_projects,
        "legal_risk": legal_risk,
        "entry_date": entry.date().isoformat()
    }
    for s in STAGES:
        row[f"{s}_entry"] = stage_entry.get(s, "")
        row[f"{s}_completion"] = stage_completion.get(s, "")
        row[f"duration_{s}"] = durations.get(s, 0)
        row[f"event_{s}"] = events.get(s, 0)

    rows.append(row)

    # generate a few legal events proportional to legal_risk
    n_legal = np.random.poisson(legal_risk * 2)
    for k in range(n_legal):
        ev_date = entry + timedelta(days=int(np.random.exponential(300)))
        if ev_date > end_date:
            continue
        ev_type = random.choice(["notice", "petition", "injunction", "settlement"])
        severity = random.random() * (1.0 if ev_type in ["injunction"] else 0.6)
        legal_events.append({
            "project_id": pid,
            "district": district,
            "date": ev_date.date().isoformat(),
            "event_type": ev_type,
            "severity": round(severity,3)
        })

# write outputs
import os
os.makedirs(OUT_DIR, exist_ok=True)
projects_df = pd.DataFrame(rows)
projects_df.to_csv(os.path.join(OUT_DIR, "projects.csv"), index=False)
legal_df = pd.DataFrame(legal_events)
legal_df.to_csv(os.path.join(OUT_DIR, "legal_events.csv"), index=False)
with open(os.path.join(OUT_DIR, "districts.json"), "w") as f:
    json.dump(district_resources, f, indent=2)

print("Wrote data/ with projects.csv, legal_events.csv, districts.json")
