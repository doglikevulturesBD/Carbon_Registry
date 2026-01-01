# methodologies/vm0038_ev.py
# ------------------------------------------------------------
# VM0038-style EV Charging Methodology (DEMO / Workbench)
#
# Integrates with the existing Carbon Registry SQLite DB:
#   data/carbon_registry.db
#
# Writes results to:
#   calc_runs  (created by Scope Calculator page)
#
# NOTE:
# - This is a "VM0038-style" educational implementation, not an official verifier tool.
# - Users must record EF sources / assumptions for audit readiness.
# ------------------------------------------------------------

import streamlit as st
from datetime import date, datetime
from pathlib import Path
import sqlite3
import json
import uuid
import pandas as pd
import altair as alt
from typing import Tuple, Optional, Dict, Any


# ---------------------------
# DB (SQLite) — same as Registry page
# ---------------------------
DB_PATH = Path("data/carbon_registry.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
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


# ---------------------------
# Ensure calc_runs exists (same schema as Scope Calculator page)
# ---------------------------
def ensure_calc_runs_schema() -> None:
    db_exec(
        """
        CREATE TABLE IF NOT EXISTS calc_runs (
            calc_id TEXT PRIMARY KEY,
            project_id TEXT,
            calc_type TEXT NOT NULL,      -- 'scope' | 'methodology'
            calc_name TEXT NOT NULL,
            scope_label TEXT,
            period_start TEXT,
            period_end TEXT,
            baseline_tco2e REAL,
            project_tco2e REAL,
            reduction_tco2e REAL,
            inputs_json TEXT,
            outputs_json TEXT,
            factor_source TEXT,
            status TEXT DEFAULT 'final',
            actor TEXT,
            created_at TEXT NOT NULL
        );
        """
    )

ensure_calc_runs_schema()


# ---------------------------
# Default emission factors (DEMO VALUES)
# IMPORTANT: These are placeholders for demo. For any real MRV, users must cite sources.
# ---------------------------
FUEL_EF = {
    "Petrol": 2.31,   # kg CO2e / litre
    "Diesel": 2.68,
    "LPG": 1.51,
    "Other": None,
}

WTT_EF = {
    "Petrol": 0.52,   # kg CO2e / litre (upstream)
    "Diesel": 0.58,
    "LPG": 0.21,
}

FUEL_ENERGY_MJ = {
    "Petrol": 34.2,
    "Diesel": 38.6,
    "LPG": 26.8,
    "Other": 0.0,
}

RENEWABLE_EF = 0.0


# ---------------------------
# Project listing (reads your registry 'projects' table)
# ---------------------------
def list_projects() -> pd.DataFrame:
    try:
        df = db_query(
            """
            SELECT project_id, project_code, project_name, status, updated_at
            FROM projects
            ORDER BY updated_at DESC
            """
        )
        if not df.empty:
            df["project_code"] = df["project_code"].fillna("")
            df["project_name"] = df["project_name"].fillna("")
            df["label"] = df["project_code"] + " — " + df["project_name"]
        return df
    except Exception:
        return pd.DataFrame(columns=["project_id", "project_code", "project_name", "status", "updated_at", "label"])


def save_methodology_run(
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
    status: str = "final"
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
            "VM0038-style",
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


# ---------------------------
# Main methodology UI
# ---------------------------
def vm0038_ev():
    st.title("⚡ VM0038 – EV Charging System Methodology (Style / Demo)")
    st.caption("Baseline: Internal combustion vehicles | Project: Electric vehicles with grid/renewable electricity")

    with st.expander("📘 Methodology overview (VM0038-style)", expanded=False):
        st.markdown(
            """
This module demonstrates a **VM0038-style structure** (educational workbench):

- **Baseline emissions (BEy)**: avoided ICE fuel  
  \\( BE_y = FC_{fuel,y} \\times (EF_{fuel} + EF_{WTT}) \\)

- **Project emissions (PEy)**: EV charging electricity  
  \\( PE_y = E_{EV,y} \\times EF_{grid,y} \\)  
  adjusted for charging efficiency + renewable share + grid decarbonisation.

- **Net emission reduction**  
  \\( ER_y = BE_y - PE_y \\) and \\( ER_{total} = \\sum ER_y \\)

Includes:
- Charger fleet aggregation
- Energy equivalence (MJ/year)
- Uncertainty bands (screening-level)
- Annual table + charts + CSV
- Save results into registry ledger (`calc_runs`)
            """
        )

    st.divider()

    # ---------------------------
    # 1) Project selection
    # ---------------------------
    projs = list_projects()
    if projs.empty:
        st.error("No projects found. Create a project in the Registry page first.")
        return

    active_pid = st.session_state.get("active_project_id")
    options = projs["project_id"].tolist()
    default_idx = options.index(active_pid) if active_pid in options else 0

    project_id = st.selectbox(
        "Save VM0038 results to project:",
        options=options,
        index=default_idx,
        format_func=lambda pid: projs.loc[projs.project_id == pid, "label"].values[0],
    )

    st.divider()

    # ---------------------------
    # Optional period metadata
    # ---------------------------
    p1, p2 = st.columns(2)
    with p1:
        period_start = st.text_input("Period start (YYYY-MM-DD)", value="")
    with p2:
        period_end = st.text_input("Period end (YYYY-MM-DD)", value="")

    st.divider()

    # ---------------------------
    # 2) Baseline – fuel avoided
    # ---------------------------
    st.subheader("1. Baseline – Fuel Consumption Avoided (Internal Combustion Vehicles)")

    col1, col2 = st.columns(2)
    with col1:
        fuel_type = st.selectbox("Baseline fuel type:", list(FUEL_EF.keys()))
        fuel_use_l = st.number_input("Fuel consumption avoided (litres/year):", min_value=0.0, step=0.01)
    with col2:
        default_fuel_ef = FUEL_EF[fuel_type] if FUEL_EF[fuel_type] is not None else 0.0
        ef_fuel = st.number_input("Fuel EF (kg CO₂e/litre):", value=float(default_fuel_ef), min_value=0.0, step=0.0001)
        include_wtt = st.checkbox("Include Well-to-Tank (upstream) emissions", value=True)

    wtt_ef = WTT_EF.get(fuel_type, 0.0) if include_wtt else 0.0

    energy_mj_year = fuel_use_l * FUEL_ENERGY_MJ.get(fuel_type, 0.0)
    e1, e2 = st.columns(2)
    with e1:
        st.info(f"Fuel energy equivalent: **{energy_mj_year:,.1f} MJ/year**")
    with e2:
        st.caption("Energy equivalence is for diagnostics; not used in CO₂e accounting.")

    baseline_uncert_pct = st.slider("Baseline emission uncertainty (%):", 0.0, 20.0, 5.0, step=0.5)

    BEy_kg = fuel_use_l * (ef_fuel + wtt_ef)
    st.write(f"**Baseline emissions (BEy): {BEy_kg:,.2f} kg CO₂e/year**")

    st.divider()

    # ---------------------------
    # 3) Project emissions – electricity
    # ---------------------------
    st.subheader("2. Project – EV Charging Electricity Use")

    mode = st.radio(
        "How do you want to define project electricity use?",
        ["Direct annual kWh", "From charger fleet parameters"],
        horizontal=True,
    )

    if mode == "Direct annual kWh":
        c1, c2 = st.columns(2)
        with c1:
            kwh_year = st.number_input("EV charging electricity (kWh/year):", min_value=0.0, step=0.01)
        with c2:
            charge_eff = st.slider("Charging system efficiency (%):", 70, 100, 90)
    else:
        st.markdown("**Charger fleet parameters**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            n_chargers = st.number_input("Number of chargers:", min_value=1, value=4, step=1)
        with c2:
            sessions_per_day = st.number_input("Sessions per charger per day:", min_value=0.0, value=4.0, step=0.5)
        with c3:
            kwh_per_session = st.number_input("kWh delivered per session:", min_value=0.0, value=20.0, step=0.5)
        with c4:
            operating_days = st.number_input("Operating days per year:", min_value=0, value=300, step=1)

        kwh_year = n_chargers * sessions_per_day * kwh_per_session * operating_days
        st.info(f"Derived annual EV charging electricity: **{kwh_year:,.1f} kWh/year**")
        charge_eff = st.slider("Charging system efficiency (%):", 70, 100, 90)

    g1, g2 = st.columns(2)
    with g1:
        ef_grid = st.number_input("Grid EF (kg CO₂e/kWh):", min_value=0.0, value=0.9, step=0.0001)
    with g2:
        renewable_fraction = st.slider("Renewable supply fraction (% of total electricity):", 0, 100, 0, step=5)

    project_uncert_pct = st.slider("Project emission uncertainty (%):", 0.0, 20.0, 5.0, step=0.5)

    eff_grid_ef = ef_grid * (1 - renewable_fraction / 100.0) + RENEWABLE_EF * (renewable_fraction / 100.0)
    useful_kwh = kwh_year * (charge_eff / 100.0)

    PEy_kg = useful_kwh * eff_grid_ef
    st.write(f"**Project emissions (PEy, year 1): {PEy_kg:,.2f} kg CO₂e/year**")

    st.divider()

    # ---------------------------
    # 4) Duration + grid decarb + annual series
    # ---------------------------
    st.subheader("3. Project Duration, Grid Decarbonisation & Annual Emissions")

    d1, d2 = st.columns(2)
    with d1:
        years = st.number_input("Project duration (years):", min_value=1, value=10, step=1)
    with d2:
        grid_decarb = st.slider("Annual grid EF reduction (%/year):", 0.0, 10.0, 2.0, step=0.5)

    if BEy_kg <= 0 or kwh_year <= 0:
        st.warning("Enter non-zero baseline fuel and project electricity data to compute reductions.")
        return

    records = []
    current_ef = eff_grid_ef

    u_b = baseline_uncert_pct / 100.0
    u_p = project_uncert_pct / 100.0
    combined_u = (u_b**2 + u_p**2) ** 0.5

    for y in range(1, int(years) + 1):
        year_PEy_kg = useful_kwh * current_ef
        year_BEy_kg = BEy_kg
        year_ER_kg = year_BEy_kg - year_PEy_kg

        bey_unc_kg = year_BEy_kg * u_b
        pey_unc_kg = year_PEy_kg * u_p
        ery_unc_kg = abs(year_ER_kg) * combined_u

        records.append(
            {
                "Year": y,
                "BEy (kg CO₂e)": year_BEy_kg,
                "BEy ± (kg)": bey_unc_kg,
                "PEy (kg CO₂e)": year_PEy_kg,
                "PEy ± (kg)": pey_unc_kg,
                "Net Reduction (kg)": year_ER_kg,
                "Net Reduction ± (kg)": ery_unc_kg,
                "BEy (t)": year_BEy_kg / 1000.0,
                "PEy (t)": year_PEy_kg / 1000.0,
                "Net Reduction (t)": year_ER_kg / 1000.0,
                "Net Reduction ± (t)": ery_unc_kg / 1000.0,
            }
        )

        current_ef *= (1 - grid_decarb / 100.0)

    df = pd.DataFrame(records)

    total_reduction_kg = float(df["Net Reduction (kg)"].sum())
    total_reduction_t = total_reduction_kg / 1000.0

    total_reduction_unc_kg = float((df["Net Reduction ± (kg)"] ** 2).sum() ** 0.5)
    total_reduction_unc_t = total_reduction_unc_kg / 1000.0

    st.markdown("### Annual emission results")
    st.dataframe(
        df[["Year", "BEy (t)", "PEy (t)", "Net Reduction (t)", "Net Reduction ± (t)"]],
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"**Total emission reductions over {int(years)} years:** "
        f"{total_reduction_t:,.3f} ± {total_reduction_unc_t:,.3f} t CO₂e"
    )

    # ---------------------------
    # 5) Charts
    # ---------------------------
    st.markdown("### Emission trajectories")
    chart_data = df[["Year", "BEy (t)", "PEy (t)", "Net Reduction (t)"]].melt(
        id_vars="Year",
        var_name="Series",
        value_name="tCO2e",
    )

    chart = (
        alt.Chart(chart_data)
        .mark_line(point=True)
        .encode(
            x="Year:O",
            y="tCO2e:Q",
            color="Series:N",
            tooltip=["Year", "Series", "tCO2e"],
        )
    )
    st.altair_chart(chart, use_container_width=True)

    # ---------------------------
    # 6) Download CSV
    # ---------------------------
    st.markdown("### Download annual results")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download VM0038 annual results (CSV)",
        data=csv_bytes,
        file_name="vm0038_ev_annual_results.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()

    # ---------------------------
    # 7) Save to Registry ledger (calc_runs)
    # ---------------------------
    st.subheader("4. Save to Carbon Registry ledger (recommended)")

    st.caption(
        "This saves a methodology run into the registry ledger (`calc_runs`). "
        "For demo: it proves the pipeline from methodology → record → audit-ready JSON."
    )

    factor_source = st.text_area(
        "Emission factor source / reference (required to save)",
        placeholder="Cite your fuel EF + grid EF sources (dataset name, year, geography, link/doc).",
        height=90,
    )
    status = st.selectbox("Record status", ["final", "draft"], index=0)

    can_save = bool(factor_source.strip())
    if st.button("✅ Save methodology run", use_container_width=True, disabled=not can_save):
        baseline_tco2e = BEy_kg / 1000.0
        project_tco2e = (PEy_kg) / 1000.0  # year 1 project (for header fields)
        reduction_tco2e = total_reduction_t

        inputs = {
            "methodology": "VM0038-style",
            "project_id": project_id,
            "period_start": period_start.strip() or None,
            "period_end": period_end.strip() or None,
            "baseline": {
                "fuel_type": fuel_type,
                "fuel_use_l_per_year": fuel_use_l,
                "ef_fuel_kg_per_l": ef_fuel,
                "include_wtt": include_wtt,
                "wtt_ef_kg_per_l": wtt_ef,
                "baseline_uncert_pct": baseline_uncert_pct,
                "energy_mj_year": energy_mj_year,
            },
            "project": {
                "mode": mode,
                "kwh_year": kwh_year,
                "charge_eff_pct": charge_eff,
                "ef_grid_kg_per_kwh": ef_grid,
                "renewable_fraction_pct": renewable_fraction,
                "effective_grid_ef_kg_per_kwh": eff_grid_ef,
                "useful_kwh": useful_kwh,
                "project_uncert_pct": project_uncert_pct,
            },
            "duration": {
                "years": int(years),
                "grid_decarb_pct_per_year": grid_decarb,
            }
        }

        outputs = {
            "year1": {
                "BEy_kg": BEy_kg,
                "PEy_kg": PEy_kg,
                "ERy_kg": (BEy_kg - PEy_kg),
            },
            "totals": {
                "total_reduction_t": total_reduction_t,
                "total_reduction_unc_t": total_reduction_unc_t,
            },
            "annual_table": df.to_dict(orient="records"),
        }

        calc_id = save_methodology_run(
            project_id=project_id,
            calc_name="VM0038-style — EV Charging",
            period_start=period_start.strip() or None,
            period_end=period_end.strip() or None,
            baseline_tco2e=baseline_tco2e,
            project_tco2e=project_tco2e,
            reduction_tco2e=reduction_tco2e,
            inputs=inputs,
            outputs=outputs,
            factor_source=factor_source,
            status=status
        )
        st.success(f"Saved ✅  calc_id = {calc_id}")

    # ---------------------------
    # 8) Monitoring & QA/QC guidance
    # ---------------------------
    with st.expander("🔍 Monitoring, data & QA/QC hints"):
        st.markdown(
            """
**Suggested monitoring parameters**
- Baseline fuel:
  - Fuel purchase records (litres)
  - Vehicle kilometres travelled (if applicable)
  - Fuel type breakdown (petrol/diesel)
- Project electricity:
  - Metered kWh per charger / per site
  - Charging sessions and duration
  - EV fleet size and utilisation

**QA/QC suggestions**
- Cross-check fuel with invoices and logbooks
- Check charger meter calibration where applicable
- Reconcile station kWh with utility bills
- Document all emission factors with source + vintage + geography

**Uncertainty**
- Slider uncertainty uses a simple root-sum-square approximation.
- Verification-grade uncertainty analysis must follow the chosen standard/methodology.
            """
        )
