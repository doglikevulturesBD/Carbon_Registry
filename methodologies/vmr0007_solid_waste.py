# methodologies/vmr0007_waste.py
# ------------------------------------------------------------
# VMR0007 – Solid Waste Recovery & Recycling (MVP)
# Cloud-safe + saves into the Carbon Registry SQLite ledger.
#
# IMPORTANT:
# - Default EFs here are placeholders for demo only.
# - For any external discussion, you must replace with cited sources.
# ------------------------------------------------------------

import streamlit as st
import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import pandas as pd


# ------------------------------------------------------------
# DB (SQLite) — Cloud-safe
# ------------------------------------------------------------
DB_PATH = Path("data/carbon_registry.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def db_exec(query: str, params: Tuple = ()) -> None:
    conn = get_conn()
    conn.execute(query, params)
    conn.commit()

def db_query(query: str, params: Tuple = ()) -> pd.DataFrame:
    conn = get_conn()
    rows = conn.execute(query, params).fetchall()
    return pd.DataFrame([dict(r) for r in rows])

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def ensure_schema() -> None:
    db_exec(
        """
        CREATE TABLE IF NOT EXISTS calc_runs (
            calc_id TEXT PRIMARY KEY,
            project_id TEXT,
            calc_type TEXT NOT NULL,      -- 'scope' | 'methodology'
            calc_name TEXT NOT NULL,
            scope_label TEXT,             -- optional
            period_start TEXT,
            period_end TEXT,
            baseline_tco2e REAL,
            project_tco2e REAL,
            reduction_tco2e REAL,
            inputs_json TEXT,
            outputs_json TEXT,
            factor_source TEXT,           -- required to save
            status TEXT DEFAULT 'final',  -- 'final' | 'draft'
            actor TEXT,
            created_at TEXT NOT NULL
        );
        """
    )

ensure_schema()


# ------------------------------------------------------------
# Registry helpers
# ------------------------------------------------------------
def list_projects() -> pd.DataFrame:
    try:
        df = db_query(
            """
            SELECT project_id, project_code, project_name, status, updated_at
            FROM projects
            ORDER BY updated_at DESC
            """
        )
        if df.empty:
            return df
        df["project_code"] = df["project_code"].fillna("")
        df["project_name"] = df["project_name"].fillna("")
        df["label"] = df["project_code"] + " — " + df["project_name"]
        return df
    except Exception:
        return pd.DataFrame(columns=["project_id", "project_code", "project_name", "status", "updated_at", "label"])

def save_calc_run(
    *,
    project_id: str,
    calc_name: str,
    period_start: Optional[str],
    period_end: Optional[str],
    baseline_tco2e: float,
    project_tco2e: float,
    reduction_tco2e: float,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    factor_source: str,
    status: str = "final",
) -> str:
    calc_id = str(uuid.uuid4())
    actor = st.session_state.get("actor_name", "unknown")
    ts = now_iso()

    db_exec(
        """
        INSERT INTO calc_runs (
            calc_id, project_id, calc_type, calc_name, scope_label,
            period_start, period_end,
            baseline_tco2e, project_tco2e, reduction_tco2e,
            inputs_json, outputs_json, factor_source,
            status, actor, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            calc_id,
            project_id,
            "methodology",
            calc_name,
            "VMR0007",
            period_start,
            period_end,
            float(baseline_tco2e),
            float(project_tco2e),
            float(reduction_tco2e),
            json.dumps(inputs, ensure_ascii=False),
            json.dumps(outputs, ensure_ascii=False),
            factor_source.strip(),
            status,
            actor,
            ts,
        ),
    )
    return calc_id


# ------------------------------------------------------------
# EF TABLES (MVP placeholders)
# Units:
# - BASELINE_EF: tCO2e per tonne material disposed (baseline)
# - PROJECT_EF:  tCO2e per tonne material recycled/processed (project)
# - TRANSPORT_EF: tCO2e per tonne-km
# - DIESEL_EF: tCO2e per litre
# - ELECTRICITY_EF: tCO2e per kWh
# ------------------------------------------------------------
BASELINE_EF = {
    "Plastic": 1.3,
    "Paper": 0.9,
    "Metal": 1.5,
    "Glass": 0.4,
}

PROJECT_EF = {
    "Plastic": 0.55,
    "Paper": 0.35,
    "Metal": 0.20,
    "Glass": 0.15,
}

TRANSPORT_EF = 0.000102
DIESEL_EF = 0.0027
ELECTRICITY_EF = 0.0009


# ------------------------------------------------------------
# Core calculations
# ------------------------------------------------------------
def calculate_baseline(material: str, tons: float, baseline_ef: float) -> float:
    # tCO2e/yr
    return tons * baseline_ef

def calculate_project_emissions(
    *,
    material: str,
    tons_recycled: float,
    contamination_pct: float,
    baseline_ef: float,
    project_ef: float,
    diesel_litres: float,
    diesel_ef: float,
    kwh: float,
    electricity_ef: float,
    distance_km: float,
    transport_ef: float,
    round_trip: bool,
) -> Dict[str, float]:
    clean_tons = tons_recycled * (1 - contamination_pct / 100.0)
    residue_tons = tons_recycled - clean_tons

    # If round trip, double distance
    km_effective = distance_km * (2.0 if round_trip else 1.0)

    pe_recycling = clean_tons * project_ef
    pe_residuals = residue_tons * baseline_ef  # residues treated as baseline disposal
    pe_transport = clean_tons * km_effective * transport_ef
    pe_energy = (diesel_litres * diesel_ef) + (kwh * electricity_ef)

    total_pe = pe_recycling + pe_residuals + pe_transport + pe_energy

    return {
        "clean_tons": clean_tons,
        "residual_tons": residue_tons,
        "km_effective": km_effective,
        "pe_recycling": pe_recycling,
        "pe_residuals": pe_residuals,
        "pe_transport": pe_transport,
        "pe_energy": pe_energy,
        "total_pe": total_pe,
    }


# ------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------
def vmr0007_app() -> None:
    st.title("♻️ VMR0007 – Solid Waste Recovery & Recycling (MVP)")
    st.caption("Education-focused demo: baseline disposal vs project recycling + optional save to MRV ledger.")

    with st.expander("📘 Method overview (VMR0007-style framing)", expanded=False):
        st.markdown(
            """
**Simplified structure (MVP):**
- **Baseline:** all material is disposed → `BE = tons × EF_baseline`
- **Project:** material is processed/recycled with contamination (residuals disposed)
  - recycling process emissions
  - residual disposal emissions
  - transport emissions (tonne-km)
  - energy emissions (diesel + electricity)

**Important:** Default factors shown are placeholders. Replace with cited sources.
            """.strip()
        )

    # -------------------------------------------------
    # Project selection (for saving)
    # -------------------------------------------------
    st.markdown("## 0️⃣ Save Context (optional)")
    projs = list_projects()
    if projs.empty:
        st.warning("No projects found. Create a project in the Registry page first (📂 Projects). Saving disabled.")
        save_enabled = False
        pid = None
    else:
        save_enabled = True
        active_pid = st.session_state.get("active_project_id")
        options = projs["project_id"].tolist()
        default_idx = options.index(active_pid) if active_pid in options else 0
        pid = st.selectbox(
            "Select project (for saving results)",
            options=options,
            index=default_idx,
            format_func=lambda x: projs.loc[projs.project_id == x, "label"].values[0],
        )

    st.divider()

    # -------------------------------------------------
    # Inputs
    # -------------------------------------------------
    st.markdown("## 1️⃣ Inputs")

    c1, c2 = st.columns(2)
    with c1:
        material = st.selectbox("Material type", list(BASELINE_EF.keys()))
        tons = st.number_input("Total material processed (tonnes/year)", min_value=0.0, value=100.0, step=0.1)
        contamination = st.slider("Contamination rate (%)", min_value=0, max_value=40, value=10, step=1)

    with c2:
        diesel_l = st.number_input("Diesel used (litres/year)", min_value=0.0, value=500.0, step=10.0)
        kwh = st.number_input("Electricity use (kWh/year)", min_value=0.0, value=30000.0, step=100.0)
        distance_km = st.number_input("Transport distance (km, one-way)", min_value=0.0, value=15.0, step=1.0)
        round_trip = st.checkbox("Treat distance as round-trip (×2)", value=False)

    st.divider()

    # -------------------------------------------------
    # Factors (editable + visible unit discipline)
    # -------------------------------------------------
    st.markdown("## 2️⃣ Emission factors (editable, demo defaults)")
    st.caption("Units shown explicitly. Replace defaults + cite sources if you save/publish.")

    bcol, pcol, tcol = st.columns(3)
    with bcol:
        baseline_ef = st.number_input(
            f"Baseline EF (tCO₂e / tonne) — {material}",
            min_value=0.0,
            value=float(BASELINE_EF[material]),
            step=0.01,
        )
    with pcol:
        project_ef = st.number_input(
            f"Project EF (tCO₂e / tonne) — {material}",
            min_value=0.0,
            value=float(PROJECT_EF[material]),
            step=0.01,
        )
    with tcol:
        transport_ef = st.number_input(
            "Transport EF (tCO₂e / tonne-km)",
            min_value=0.0,
            value=float(TRANSPORT_EF),
            step=0.000001,
            format="%.6f",
        )

    ecol1, ecol2 = st.columns(2)
    with ecol1:
        diesel_ef = st.number_input(
            "Diesel EF (tCO₂e / litre)",
            min_value=0.0,
            value=float(DIESEL_EF),
            step=0.0001,
            format="%.4f",
        )
    with ecol2:
        electricity_ef = st.number_input(
            "Electricity EF (tCO₂e / kWh)",
            min_value=0.0,
            value=float(ELECTRICITY_EF),
            step=0.0001,
            format="%.4f",
        )

    st.divider()

    # -------------------------------------------------
    # Period metadata (optional)
    # -------------------------------------------------
    st.markdown("## 3️⃣ Period (metadata)")
    pc1, pc2 = st.columns(2)
    with pc1:
        period_start = st.text_input("Period start (YYYY-MM-DD)", value="")
    with pc2:
        period_end = st.text_input("Period end (YYYY-MM-DD)", value="")

    st.divider()

    # -------------------------------------------------
    # Calculations
    # -------------------------------------------------
    st.markdown("## 4️⃣ Results")

    baseline_emissions = calculate_baseline(material, tons, baseline_ef)

    project_data = calculate_project_emissions(
        material=material,
        tons_recycled=tons,
        contamination_pct=float(contamination),
        baseline_ef=baseline_ef,
        project_ef=project_ef,
        diesel_litres=diesel_l,
        diesel_ef=diesel_ef,
        kwh=kwh,
        electricity_ef=electricity_ef,
        distance_km=distance_km,
        transport_ef=transport_ef,
        round_trip=round_trip,
    )
    project_emissions = project_data["total_pe"]
    er = baseline_emissions - project_emissions

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline emissions (tCO₂e/yr)", f"{baseline_emissions:,.3f}")
    c2.metric("Project emissions (tCO₂e/yr)", f"{project_emissions:,.3f}")
    c3.metric("Net ER (tCO₂e/yr)", f"{er:,.3f}")

    st.markdown("### Breakdown")
    breakdown = pd.DataFrame({
        "Parameter": [
            "Clean recyclable tonnes",
            "Residual tonnes (disposed)",
            "Project – recycling emissions (tCO₂e)",
            "Project – residual disposal (tCO₂e)",
            "Project – transport (tCO₂e)",
            "Project – energy (diesel + electricity) (tCO₂e)",
            "Effective transport distance (km)",
        ],
        "Value": [
            f"{project_data['clean_tons']:.3f}",
            f"{project_data['residual_tons']:.3f}",
            f"{project_data['pe_recycling']:.3f}",
            f"{project_data['pe_residuals']:.3f}",
            f"{project_data['pe_transport']:.3f}",
            f"{project_data['pe_energy']:.3f}",
            f"{project_data['km_effective']:.1f}",
        ]
    })
    st.dataframe(breakdown, use_container_width=True, hide_index=True)

    with st.expander("🧾 Show equations", expanded=False):
        st.markdown(
            f"""
**Baseline**  
BE = tons × EF_baseline  
= {tons:,.6g} × {baseline_ef:,.6g} = **{baseline_emissions:,.6g} tCO₂e/yr**

**Project**  
Clean = tons × (1 − contamination) = {tons:,.6g} × (1 − {float(contamination)/100:.4f}) = **{project_data['clean_tons']:.6g} t**  
Residual = tons − Clean = **{project_data['residual_tons']:.6g} t**

PE_recycling = Clean × EF_project = {project_data['clean_tons']:.6g} × {project_ef:,.6g} = {project_data['pe_recycling']:.6g}  
PE_residuals = Residual × EF_baseline = {project_data['residual_tons']:.6g} × {baseline_ef:,.6g} = {project_data['pe_residuals']:.6g}  
PE_transport = Clean × km × EF_transport = {project_data['clean_tons']:.6g} × {project_data['km_effective']:.6g} × {transport_ef:,.6g} = {project_data['pe_transport']:.6g}  
PE_energy = diesel×EF + kWh×EF = {diesel_l:,.6g}×{diesel_ef:,.6g} + {kwh:,.6g}×{electricity_ef:,.6g} = {project_data['pe_energy']:.6g}

PE_total = **{project_emissions:,.6g} tCO₂e/yr**  
ER = BE − PE = **{er:,.6g} tCO₂e/yr**
            """.strip()
        )

    st.divider()

    # -------------------------------------------------
    # Save to ledger
    # -------------------------------------------------
    st.markdown("## 5️⃣ Save result to Carbon Registry ledger")

    factor_source = st.text_area(
        "Factor source / references (required to save)",
        placeholder=(
            "Cite baseline & project factors and energy/transport factors.\n"
            "- Baseline EF source (dataset, year/version, geography)\n"
            "- Project EF source (recycling/avoided virgin assumptions)\n"
            "- Transport EF source (tonne-km EF)\n"
            "- Diesel & electricity EF sources\n"
        ),
        height=120,
    )

    status = st.selectbox("Save status", ["final", "draft"], index=0)

    can_save = bool(pid) and save_enabled and bool(factor_source.strip())

    if st.button("💾 Save to Ledger", use_container_width=True, disabled=not can_save):
        inputs: Dict[str, Any] = {
            "methodology": "VMR0007 (MVP)",
            "period_start": period_start.strip() or None,
            "period_end": period_end.strip() or None,
            "material": material,
            "tons_processed": tons,
            "contamination_pct": float(contamination),
            "distance_km_one_way": distance_km,
            "round_trip": round_trip,
            "diesel_litres": diesel_l,
            "electricity_kwh": kwh,
            "efs": {
                "baseline_ef_tco2e_per_tonne": baseline_ef,
                "project_ef_tco2e_per_tonne": project_ef,
                "transport_ef_tco2e_per_tonne_km": transport_ef,
                "diesel_ef_tco2e_per_litre": diesel_ef,
                "electricity_ef_tco2e_per_kwh": electricity_ef,
            },
            "units": {
                "baseline_project_ef": "tCO2e/tonne",
                "transport_ef": "tCO2e/tonne-km",
                "diesel_ef": "tCO2e/litre",
                "electricity_ef": "tCO2e/kWh",
            },
        }

        outputs: Dict[str, Any] = {
            "baseline_tco2e_per_year": baseline_emissions,
            "project_tco2e_per_year": project_emissions,
            "reduction_tco2e_per_year": er,
            "breakdown": project_data,
        }

        calc_id = save_calc_run(
            project_id=pid,
            calc_name="VMR0007 — Recycling & Recovery (MVP)",
            period_start=period_start.strip() or None,
            period_end=period_end.strip() or None,
            baseline_tco2e=float(baseline_emissions),
            project_tco2e=float(project_emissions),
            reduction_tco2e=float(er),
            inputs=inputs,
            outputs=outputs,
            factor_source=factor_source,
            status=status,
        )
        st.success(f"Saved ✅ calc_id = {calc_id}")

    st.caption(
        "Launch note: this MVP is designed to demonstrate structure + traceability. "
        "Replace placeholder EFs and document factor provenance for any real claims."
    )


# Allow running directly:
# streamlit run methodologies/vmr0007_waste.py
if __name__ == "__main__":
    vmr0007_app()
