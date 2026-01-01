# methodologies/am0124_hydrogen.py
# ------------------------------------------------------------
# AM0124 – Hydrogen Electrolysis Emissions Calculator (MVP)
# "Cloud-safe, single file" + saves into the same SQLite ledger used by your Registry/Scope pages.
#
# What this version fixes vs your draft:
# ✅ Uses the SAME SQLite DB at data/carbon_registry.db
# ✅ Creates/uses calc_runs table (same schema pattern as Scope Calculator)
# ✅ Project selector pulls from `projects`
# ✅ Grid share check uses grid/(grid+captive) (more defensible than grid/captive)
# ✅ Adds "EF source" requirement before saving
# ✅ Supports saving as draft/non-compliant if you want (toggle)
# ✅ Stores full inputs/outputs JSON for audit-style traceability
#
# Notes:
# - This is an educational MVP aligned to a "AM0124-style" structure.
# - It does NOT ship official default factors. If you use defaults, cite sources.
# ------------------------------------------------------------

import streamlit as st
import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np


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
    # projects table should exist from Registry page; we won't recreate it here.
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
# Helpers
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
            "AM0124",
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

def metric_row(be: float, pe: float, er: float, total_er: float) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline emissions (tCO₂e/yr)", f"{be:,.3f}")
    c2.metric("Project emissions (tCO₂e/yr)", f"{pe:,.3f}")
    c3.metric("Net ERᵧ (tCO₂e/yr)", f"{er:,.3f}")
    c4.metric("Total ER (tCO₂e)", f"{total_er:,.3f}")

def require_positive(name: str, x: float) -> bool:
    if x <= 0:
        st.error(f"{name} must be > 0.")
        return False
    return True


# ------------------------------------------------------------
# UI + Core calc
# ------------------------------------------------------------
def am0124_hydrogen_app() -> None:
    st.title("🟦 AM0124 – Hydrogen Electrolysis Emissions Calculator (MVP)")
    st.caption("Educational & demonstration version: baseline vs project emissions + ledger save (SQLite).")

    with st.expander("📘 Method overview (AM0124-style framing)", expanded=False):
        st.markdown(
            """
**Core idea (simplified):**
- **Baseline emissions (BEy):** hydrogen produced × baseline emissions factor (tCO₂e per tH₂)
- **Project emissions (PEy):**
  - grid electricity emissions
  - optional on-site fossil fuel emissions
  - transport emissions (optional)
  - hydrogen leakage (scenario-based CO₂e equivalence)

- **Net ERᵧ:** BEy − PEy  
- **Total ER:** ERᵧ × project years

**Important:** This MVP does not ship “official” factors. If you use defaults, cite sources.
            """.strip()
        )

    # -------------------------------------------------
    # Project selection (for saving)
    # -------------------------------------------------
    st.markdown("## 0️⃣ Save Context (optional)")
    projs = list_projects()
    if projs.empty:
        st.warning("No projects found. Create a project in the Registry page first (📂 Projects).")
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
    st.markdown("## 1️⃣ Project Inputs")

    col1, col2 = st.columns(2)

    with col1:
        mh2 = st.number_input("Hydrogen Produced (t H₂ / year)", min_value=0.0, value=1000.0, step=10.0)

        baseline_tech = st.selectbox(
            "Baseline Technology",
            ["Coal (gasification + SMR)", "Natural Gas (SMR)", "Oil (gasification + SMR)"]
        )

        # Electricity
        grid_mwh = st.number_input("Grid Electricity Used (MWh / year)", min_value=0.0, value=500.0, step=10.0)
        captive_mwh = st.number_input("Captive Renewable Electricity (MWh / year)", min_value=0.0, value=9500.0, step=10.0)
        grid_ef = st.number_input("Grid Emission Factor (t CO₂e / MWh)", min_value=0.0, value=1.3, step=0.01)

    with col2:
        fossil_gj = st.number_input("On-site Fossil Fuel Use (GJ / year)", min_value=0.0, value=0.0, step=10.0)
        fossil_ef = st.number_input(
            "CO₂e Factor of Fuel (t CO₂e / GJ)",
            min_value=0.0, value=0.0, step=0.001,
            help="Example: Diesel ≈ 0.074 tCO₂/GJ (CO₂ only); adjust if using CO₂e factors."
        )

        transport_t = st.number_input(
            "Transport Emissions (t CO₂e / year)",
            min_value=0.0, value=0.0, step=1.0,
            help="Can be 0 if pipeline is powered by captive plant; include if applicable."
        )

        leak_pct = st.number_input("Hydrogen Leak Rate (%)", min_value=0.0, value=5.0, step=0.5)
        gwp_h2 = st.number_input(
            "Hydrogen leakage CO₂e equivalence (t CO₂e / t H₂ leaked)",
            min_value=0.0, value=5.8, step=0.1,
            help="Scenario parameter; choose value aligned to your reference."
        )

        years = st.number_input("Project Duration (years)", min_value=1, value=10, step=1)
        years_i = int(years)

    st.divider()

    # -------------------------------------------------
    # Period metadata (optional)
    # -------------------------------------------------
    st.markdown("## 2️⃣ Period (metadata)")
    c1, c2 = st.columns(2)
    with c1:
        period_start = st.text_input("Period start (YYYY-MM-DD)", value="")
    with c2:
        period_end = st.text_input("Period end (YYYY-MM-DD)", value="")

    st.divider()

    # -------------------------------------------------
    # Baseline EF: default + user-editable + source
    # -------------------------------------------------
    st.markdown("## 3️⃣ Baseline emissions factor")
    st.caption("Defaults are demo values only. For any serious use, replace and cite a source.")

    if "Coal" in baseline_tech:
        default_ef_bl = 19.0
    else:
        default_ef_bl = 9.0

    ef_bl = st.number_input(
        "Baseline EF (t CO₂e / t H₂)",
        min_value=0.0,
        value=float(default_ef_bl),
        step=0.1,
        help="Editable. Replace with referenced baseline factor aligned to your standard/assumptions."
    )

    st.divider()

    # -------------------------------------------------
    # Compliance-style check (grid share)
    # -------------------------------------------------
    st.markdown("## 4️⃣ Electricity mix check (screening)")
    total_mwh = grid_mwh + captive_mwh
    grid_share = (grid_mwh / total_mwh) if total_mwh > 0 else np.inf

    c1, c2, c3 = st.columns(3)
    c1.metric("Grid electricity (MWh/yr)", f"{grid_mwh:,.1f}")
    c2.metric("Captive electricity (MWh/yr)", f"{captive_mwh:,.1f}")
    c3.metric("Grid share", f"{grid_share:.3f}")

    compliant = bool(grid_share < 0.1) if np.isfinite(grid_share) else False
    if not np.isfinite(grid_share):
        st.warning("Electricity total is zero; cannot compute grid share.")
    elif not compliant:
        st.warning(f"⚠️ Screening check: grid share should be < 0.1 for a 'strict' interpretation. Current grid share = {grid_share:.3f}")
    else:
        st.success("✅ Screening check passed (grid share < 0.1).")

    st.divider()

    # -------------------------------------------------
    # Calculations
    # -------------------------------------------------
    st.markdown("## 5️⃣ Calculations")

    calc = st.button("🧮 Calculate", use_container_width=True)

    if not calc:
        st.info("Enter inputs and click **Calculate**.")
        st.stop()

    # Basic validation
    if not require_positive("Hydrogen Produced (t H₂ / year)", mh2):
        st.stop()
    if not require_positive("Baseline EF (tCO₂e/tH₂)", ef_bl):
        st.stop()

    # Baseline emissions (tCO2e/yr)
    be_y = mh2 * ef_bl

    # Project emissions:
    pe_ec = grid_mwh * grid_ef
    pe_fc = fossil_gj * fossil_ef if (fossil_gj > 0 and fossil_ef > 0) else 0.0
    pe_leak = mh2 * (leak_pct / 100.0) * gwp_h2
    pe_tr = transport_t

    pe_y = pe_ec + pe_fc + pe_tr + pe_leak

    # Net ER
    er_y = be_y - pe_y
    total_er = er_y * years_i

    # -------------------------------------------------
    # Results
    # -------------------------------------------------
    st.markdown("## 6️⃣ Results")
    metric_row(be_y, pe_y, er_y, total_er)

    st.divider()
    st.markdown("### 🔍 Project emissions breakdown (tCO₂e/yr)")
    df = pd.DataFrame({
        "Component": ["Grid Electricity", "On-site Fossil Fuel", "Transport", "Hydrogen Leaks"],
        "Emissions (tCO₂e/yr)": [pe_ec, pe_fc, pe_tr, pe_leak],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(
        "Note: Hydrogen leakage CO₂e equivalence is scenario-based. "
        "Baseline EF should be replaced with cited sources for any external use."
    )

    # -------------------------------------------------
    # Save to Registry (SQLite ledger)
    # -------------------------------------------------
    st.divider()
    st.markdown("## 7️⃣ Save result to Carbon Registry ledger")

    factor_source = st.text_area(
        "Factor source / references (required to save)",
        placeholder=(
            "Provide references for baseline EF and project factors.\n"
            "- Baseline EF source (dataset/standard, year/version, boundary)\n"
            "- Grid EF source (utility/operator, year)\n"
            "- Fuel EF source (if used)\n"
            "- Hydrogen leakage equivalence source (if used)\n"
        ),
        height=120
    )

    allow_save_noncompliant = st.checkbox(
        "Allow saving even if screening check fails (save as draft)",
        value=True
    )

    # Decide status
    status = "final" if compliant else ("draft" if allow_save_noncompliant else "blocked")

    can_save = save_enabled and pid and bool(factor_source.strip()) and (status != "blocked")

    if not save_enabled:
        st.info("Create/select a project in the Registry page to enable saving.")
    elif status == "blocked":
        st.error("Saving disabled because grid share screening check failed and 'Allow saving non-compliant' is off.")
    else:
        st.caption(f"Save status will be: **{status}**")

    if st.button("💾 Save to Ledger", use_container_width=True, disabled=not can_save):
        inputs: Dict[str, Any] = {
            "methodology": "AM0124 (MVP)",
            "period_start": period_start.strip() or None,
            "period_end": period_end.strip() or None,
            "mh2_t_per_year": mh2,
            "baseline_tech": baseline_tech,
            "baseline_ef_tco2e_per_th2": ef_bl,
            "grid_mwh": grid_mwh,
            "captive_mwh": captive_mwh,
            "grid_ef_tco2e_per_mwh": grid_ef,
            "grid_share": float(grid_share) if np.isfinite(grid_share) else None,
            "screening_compliant_grid_share_lt_0_1": compliant,
            "fossil_gj": fossil_gj,
            "fossil_ef_tco2e_per_gj": fossil_ef,
            "transport_tco2e_per_year": transport_t,
            "leak_pct": leak_pct,
            "h2_leak_co2e_equiv_t_per_t_h2": gwp_h2,
            "years": years_i,
        }

        outputs: Dict[str, Any] = {
            "baseline_emissions_tco2e_per_year": be_y,
            "project_emissions_tco2e_per_year": pe_y,
            "er_tco2e_per_year": er_y,
            "total_er_tco2e": total_er,
            "project_breakdown_tco2e_per_year": {
                "grid_electricity": pe_ec,
                "on_site_fossil": pe_fc,
                "transport": pe_tr,
                "h2_leaks": pe_leak,
            },
        }

        calc_id = save_calc_run(
            project_id=pid,
            calc_name="AM0124 — Hydrogen Electrolysis (MVP)",
            period_start=period_start.strip() or None,
            period_end=period_end.strip() or None,
            baseline_tco2e=float(be_y),
            project_tco2e=float(pe_y),
            reduction_tco2e=float(er_y),  # annual reduction (matches calc_runs convention)
            inputs=inputs,
            outputs=outputs,
            factor_source=factor_source,
            status=status,
        )
        st.success(f"Saved ✅ calc_id = {calc_id}")

    # -------------------------------------------------
    # Transparency panel
    # -------------------------------------------------
    with st.expander("🧾 Show equations (transparency)", expanded=False):
        st.markdown(
            f"""
**Baseline:**  
BEy = MH₂ × EF_BL  
= {mh2:,.6g} × {ef_bl:,.6g} = **{be_y:,.6g} tCO₂e/yr**

**Project:**  
PEy = (E_grid × EF_grid) + (Fuel_GJ × EF_fuel) + Transport + Leaks  
- Grid: {grid_mwh:,.6g} × {grid_ef:,.6g} = {pe_ec:,.6g}  
- Fossil: {fossil_gj:,.6g} × {fossil_ef:,.6g} = {pe_fc:,.6g}  
- Transport: {pe_tr:,.6g}  
- Leaks: {mh2:,.6g} × ({leak_pct:,.6g}/100) × {gwp_h2:,.6g} = {pe_leak:,.6g}  

PEy = **{pe_y:,.6g} tCO₂e/yr**

**Net:** ERy = BEy − PEy = **{er_y:,.6g} tCO₂e/yr**  
**Total:** ER_total = ERy × years = {er_y:,.6g} × {years_i} = **{total_er:,.6g} tCO₂e**
            """.strip()
        )


# ------------------------------------------------------------
# If you want Streamlit to run this file directly:
# streamlit run methodologies/am0124_hydrogen.py
# ------------------------------------------------------------
if __name__ == "__main__":
    am0124_hydrogen_app()
