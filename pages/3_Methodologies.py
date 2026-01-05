# pages/3_📘_Methodologies.py
# ------------------------------------------------------------
# Carbon Registry • Methodology Calculators (Single-file, launch-ready)
#
# Embedded CSS version:
# - No dependency on assets/style.css
# - Ensures consistent styling even if file paths / load_css / caching causes issues
#
# Contains 3 MVP demos:
# - VM0038 (EV charging)  [demo-style, not official EF values]
# - AM0124 (Hydrogen electrolysis) [demo-style applicability + ER]
# - VMR0007 (Solid waste recovery & recycling) [demo-style ER]
# ------------------------------------------------------------

import streamlit as st

# ------------------------------------------------------------
# PAGE CONFIG (must be first Streamlit call)
# ------------------------------------------------------------
st.set_page_config(page_title="Carbon Registry • Methodologies", page_icon="📘", layout="wide")


# ------------------------------------------------------------
# EMBEDDED CSS (full)
# ------------------------------------------------------------
EMBEDDED_CSS = r"""
/* ============================================================
   GLOBAL VARIABLES
============================================================ */
:root {
    --bg-dark: #020c08;
    --bg-card: rgba(10, 25, 15, 0.55);
    --bg-card-strong: rgba(10, 25, 15, 0.85);

    --green-neon: #39ff9f;
    --green-mid: #00b46f;
    --green-soft: #86ffcf;

    --text-light: #e8fff2;
    --text-mid: #b3ffdd;

    --border-glow: 0 0 12px rgba(57, 255, 159, 0.45);
    --shadow-card: 0 0 20px rgba(0, 255, 138, 0.15);

    --radius: 18px;
    --transition: 0.25s ease-in-out;
}

/* ============================================================
   RESET / BASE
============================================================ */
html, body {
    margin: 0 !important;
    padding: 0 !important;
    background: var(--bg-dark) !important;
    font-family: "Segoe UI", Roboto, sans-serif;
    color: var(--text-light);
}

h1, h2, h3, h4 {
    color: var(--green-soft) !important;
    letter-spacing: 0.6px;
    text-shadow: 0 0 8px rgba(57, 255, 159, 0.25);
}

p, label, span, div {
    color: var(--text-mid) !important;
}

/* ============================================================
   STREAMLIT OVERRIDES
============================================================ */
.stApp {
    background: var(--bg-dark) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(5, 20, 10, 0.9) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(57, 255, 159, 0.15);
    box-shadow: 4px 0 20px rgba(0, 255, 138, 0.08);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--green-soft) !important;
}

/* Sidebar buttons */
.stButton>button {
    background: var(--green-mid) !important;
    color: black !important;
    font-weight: 600;
    border-radius: var(--radius);
    border: none;
    box-shadow: var(--border-glow);
    transition: var(--transition);
}

.stButton>button:hover {
    background: var(--green-neon) !important;
    transform: translateY(-2px);
    box-shadow: 0 0 18px rgba(57, 255, 159, 0.8);
}

/* ============================================================
   CARDS / CONTAINERS
============================================================ */
.im-card, .stContainer, .stMarkdown {
    background: var(--bg-card) !important;
    padding: 18px 22px !important;
    border-radius: var(--radius);
    border: 1px solid rgba(57, 255, 159, 0.2);
    box-shadow: var(--shadow-card);
    backdrop-filter: blur(40px) saturate(120%);
}

/* General input styling */
input, textarea, select {
    background: rgba(5, 20, 10, 0.5) !important;
    border: 1px solid rgba(57, 255, 159, 0.25) !important;
    border-radius: var(--radius) !important;
    color: var(--text-light) !important;
}

input:focus, textarea:focus, select:focus {
    border-color: var(--green-neon) !important;
    box-shadow: var(--border-glow);
}

/* Metric outputs */
.stSuccess, .stAlert {
    border-left: 4px solid var(--green-neon) !important;
    background: rgba(0, 255, 138, 0.08) !important;
    color: var(--green-soft) !important;
}

/* ============================================================
   GLOW SEPARATORS
============================================================ */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(
        90deg,
        rgba(57, 255, 159, 0) 0%,
        rgba(57, 255, 159, 0.6) 50%,
        rgba(57, 255, 159, 0) 100%
    );
    margin: 25px 0;
}

/* ============================================================
   CUSTOM UTILITY CLASSES
============================================================ */
.green-glow {
    text-shadow: 0 0 12px var(--green-neon);
    color: var(--green-neon) !important;
    font-weight: 700;
}

.card {
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 20px;
    border: 1px solid rgba(57, 255, 159, 0.25);
    box-shadow: var(--shadow-card);
    backdrop-filter: blur(14px);
}

.glass-box {
    border-radius: var(--radius);
    background: rgba(10, 25, 15, 0.45);
    border: 1px solid rgba(57, 255, 159, 0.25);
    padding: 25px;
    box-shadow: var(--shadow-card);
    backdrop-filter: blur(30px);
}

/* ============================================================
   EXPANDERS
============================================================ */
.streamlit-expanderHeader {
    background: rgba(10, 30, 15, 0.6) !important;
    color: var(--green-soft) !important;
    border-radius: var(--radius) !important;
}

/* ============================================================
   TABLES
============================================================ */
tbody, thead, tr, th, td {
    color: var(--text-light) !important;
    background: rgba(10, 30, 15, 0.25) !important;
    border-color: rgba(57, 255, 159, 0.2) !important;
}

/* ============================================================
   SUCCESS, WARNING, INFO TEXT
============================================================ */
.stSuccess, .stWarning, .stInfo, .stError {
    padding: 12px 18px !important;
    border-radius: var(--radius) !important;
    border-left: 4px solid var(--green-neon) !important;
}

/* ============================================================
   BUTTONS (global)
============================================================ */
button[kind="primary"], .stDownloadButton button {
    background: var(--green-mid) !important;
    color: black !important;
    border-radius: var(--radius) !important;
    border: none !important;
    box-shadow: var(--border-glow) !important;
    transition: var(--transition);
    font-weight: 600 !important;
}

button[kind="primary"]:hover,
.stDownloadButton button:hover {
    background: var(--green-neon) !important;
    transform: translateY(-2px);
    box-shadow: 0 0 20px rgba(57, 255, 159, 0.7) !important;
}

/* ============================================================
   FOOTER HIDE
============================================================ */
footer { visibility: hidden !important; }

/* ============================================================
   ALTAIR / VEGA EMBED BACKGROUND (dark-friendly)
============================================================ */
.vega-embed, .vega-embed details, .vega-embed summary {
    background: rgba(10, 25, 15, 0.35) !important;
    border: 1px solid rgba(57, 255, 159, 0.20) !important;
    box-shadow: 0 0 20px rgba(0, 255, 138, 0.12) !important;
    border-radius: 18px !important;
    padding: 10px !important;
}

/* ============================================================
   END OF FILE
============================================================ */
"""
st.markdown(f"<style>{EMBEDDED_CSS}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------
# Imports (after CSS)
# ------------------------------------------------------------
import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime, date
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
import altair as alt


# ------------------------------------------------------------
# HERO (same vibe as your other pages)
# ------------------------------------------------------------
st.markdown(
    """
<div class="glass-box" style="padding: 26px 26px 14px 26px; margin-bottom: 14px;">
<h1 style="margin:0; color:#86ffcf; text-shadow:0 0 10px #39ff9f;">
Methodology Calculators
</h1>
<p style="font-size:18px; margin-top:10px; color:#b3ffdd;">
Worked examples (demo style) with saving into the emissions ledger.
</p>
<p style="font-size:14px; margin-top:10px; color:#b3ffdd; opacity:0.85;">
These prove structure + data flow + audit-ready saving — not official crediting outputs.
</p>
</div>
    """,
    unsafe_allow_html=True,
)


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
    # projects table should already exist from Registry page, but we guard anyway
    db_exec("""
    CREATE TABLE IF NOT EXISTS projects (
        project_id TEXT PRIMARY KEY,
        project_code TEXT,
        project_name TEXT,
        status TEXT DEFAULT 'active',
        updated_at TEXT
    );
    """)

    # A simple emissions ledger for methodology saves
    db_exec("""
    CREATE TABLE IF NOT EXISTS emissions (
        emission_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        methodology TEXT NOT NULL,
        record_date TEXT NOT NULL,
        quantity_tco2e REAL NOT NULL,
        notes TEXT,
        inputs_json TEXT,
        outputs_json TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(project_id) REFERENCES projects(project_id)
    );
    """)

ensure_schema()

def list_projects() -> pd.DataFrame:
    df = db_query("""
        SELECT project_id, project_code, project_name, status
        FROM projects
        ORDER BY updated_at DESC
    """)
    if df.empty:
        return df
    df["project_code"] = df["project_code"].fillna("")
    df["project_name"] = df["project_name"].fillna("")
    df["label"] = df["project_code"] + " — " + df["project_name"]
    df.loc[df["label"].str.strip() == "—", "label"] = df["project_id"]
    return df

def save_emission(
    project_id: str,
    methodology: str,
    quantity_tco2e: float,
    record_date: str,
    notes: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
) -> str:
    emission_id = str(uuid.uuid4())
    db_exec(
        """
        INSERT INTO emissions (
            emission_id, project_id, methodology, record_date,
            quantity_tco2e, notes, inputs_json, outputs_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            emission_id,
            project_id,
            methodology,
            record_date,
            float(quantity_tco2e),
            notes,
            json.dumps(inputs, ensure_ascii=False),
            json.dumps(outputs, ensure_ascii=False),
            now_iso(),
        )
    )
    return emission_id


# ------------------------------------------------------------
# Shared: Save panel for methodology results
# ------------------------------------------------------------
def render_save_panel(
    *,
    methodology: str,
    total_tco2e: float,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    notes_default: str = "",
):
    st.divider()
    with st.expander("💾 Save result to Carbon Registry (optional)", expanded=False):
        projs = list_projects()
        if projs.empty:
            st.error("No projects found. Create a project on the Registry page first.")
            return

        active_pid = st.session_state.get("active_project_id")
        options = projs["project_id"].tolist()
        default_idx = options.index(active_pid) if active_pid in options else 0

        pid = st.selectbox(
            "Select project",
            options=options,
            index=default_idx,
            format_func=lambda x: projs.loc[projs.project_id == x, "label"].values[0],
            key=f"{methodology}_save_pid",
        )

        c1, c2 = st.columns(2)
        with c1:
            rec_date = st.date_input("Record date", value=date.today(), key=f"{methodology}_date").isoformat()
        with c2:
            qty = st.number_input(
                "Quantity (tCO₂e)",
                value=float(total_tco2e),
                step=0.001,
                key=f"{methodology}_qty",
            )

        notes = st.text_area("Notes (optional)", value=notes_default, height=100, key=f"{methodology}_notes")

        if st.button("✅ Save to emissions ledger", use_container_width=True, key=f"{methodology}_save_btn"):
            eid = save_emission(
                project_id=pid,
                methodology=methodology,
                quantity_tco2e=qty,
                record_date=rec_date,
                notes=notes,
                inputs=inputs,
                outputs=outputs,
            )
            st.success(f"Saved ✅ emission_id = {eid}")


# ============================================================
# 1) VM0038 – EV Charging (demo-style)
# ============================================================
FUEL_EF = {"Petrol": 2.31, "Diesel": 2.68, "LPG": 1.51, "Other": None}              # kg CO2e/L (demo defaults)
WTT_EF  = {"Petrol": 0.52, "Diesel": 0.58, "LPG": 0.21}                             # kg CO2e/L (demo defaults)
FUEL_ENERGY_MJ = {"Petrol": 34.2, "Diesel": 38.6, "LPG": 26.8, "Other": 0.0}        # MJ/L
RENEWABLE_EF = 0.0                                                                  # kg CO2e/kWh (assumed)

def vm0038_ev():
    st.subheader("⚡ VM0038 (demo-style) — EV Charging")

    with st.expander("📘 Overview", expanded=False):
        st.markdown(
            """
This is a **VM0038-style demonstration**:
- Baseline: ICE fuel avoided (litres/year) × (EF + optional WTT)
- Project: EV charging electricity (kWh/year) × grid EF, adjusted for renewable fraction and charging efficiency
- Optional grid decarbonisation over time
Outputs are **screening/demo** unless you align inputs, boundaries, and factors to the official methodology.
            """
        )

    st.markdown("### 1) Baseline — Fuel avoided")
    c1, c2 = st.columns(2)
    with c1:
        fuel_type = st.selectbox("Fuel type", list(FUEL_EF.keys()), key="vm0038_fuel_type")
        fuel_use_l = st.number_input("Fuel avoided (litres/year)", min_value=0.0, step=0.01, key="vm0038_fuel_l")
    with c2:
        default_ef = float(FUEL_EF[fuel_type]) if FUEL_EF[fuel_type] is not None else 0.0
        ef_fuel = st.number_input("Fuel EF (kg CO₂e/litre)", min_value=0.0, value=default_ef, step=0.0001, key="vm0038_ef_fuel")
        include_wtt = st.checkbox("Include WTT (upstream) EF", value=True, key="vm0038_include_wtt")

    wtt = float(WTT_EF.get(fuel_type, 0.0)) if include_wtt else 0.0
    energy_mj_year = fuel_use_l * float(FUEL_ENERGY_MJ.get(fuel_type, 0.0))
    st.info(f"Fuel energy equivalent (diagnostic): **{energy_mj_year:,.1f} MJ/year**")

    baseline_uncert_pct = st.slider("Baseline uncertainty (%)", 0.0, 20.0, 5.0, 0.5, key="vm0038_u_base")

    BEy_kg = fuel_use_l * (ef_fuel + wtt)
    st.write(f"**BEy:** {BEy_kg:,.2f} kg CO₂e/year")

    st.markdown("### 2) Project — Electricity for EV charging")
    mode = st.radio(
        "Electricity definition",
        ["Direct annual kWh", "From charger fleet parameters"],
        horizontal=True,
        key="vm0038_mode",
    )

    if mode == "Direct annual kWh":
        cc1, cc2 = st.columns(2)
        with cc1:
            kwh_year = st.number_input("EV charging electricity (kWh/year)", min_value=0.0, step=0.01, key="vm0038_kwh_year")
        with cc2:
            charge_eff = st.slider("Charging efficiency (%)", 70, 100, 90, key="vm0038_eff")
    else:
        st.markdown("**Charger fleet**")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            n_chargers = st.number_input("Chargers", min_value=1, value=4, step=1, key="vm0038_n")
        with a2:
            sessions_per_day = st.number_input("Sessions/charger/day", min_value=0.0, value=4.0, step=0.5, key="vm0038_spd")
        with a3:
            kwh_per_session = st.number_input("kWh/session", min_value=0.0, value=20.0, step=0.5, key="vm0038_kps")
        with a4:
            operating_days = st.number_input("Operating days/year", min_value=0, value=300, step=1, key="vm0038_days")

        kwh_year = float(n_chargers) * sessions_per_day * kwh_per_session * float(operating_days)
        st.info(f"Derived annual electricity: **{kwh_year:,.1f} kWh/year**")
        charge_eff = st.slider("Charging efficiency (%)", 70, 100, 90, key="vm0038_eff2")

    g1, g2 = st.columns(2)
    with g1:
        ef_grid = st.number_input("Grid EF (kg CO₂e/kWh)", min_value=0.0, value=0.9, step=0.0001, key="vm0038_ef_grid")
    with g2:
        renewable_fraction = st.slider("Renewable fraction (%)", 0, 100, 0, 5, key="vm0038_ren_frac")

    project_uncert_pct = st.slider("Project uncertainty (%)", 0.0, 20.0, 5.0, 0.5, key="vm0038_u_proj")

    eff_grid_ef = ef_grid * (1 - renewable_fraction / 100.0) + RENEWABLE_EF * (renewable_fraction / 100.0)
    useful_kwh = kwh_year * (charge_eff / 100.0)   # unchanged
    PEy_kg = useful_kwh * eff_grid_ef
    st.write(f"**PEy (year 1):** {PEy_kg:,.2f} kg CO₂e/year")

    st.markdown("### 3) Duration & grid decarbonisation")
    d1, d2 = st.columns(2)
    with d1:
        years = st.number_input("Project duration (years)", min_value=1, value=10, step=1, key="vm0038_years")
    with d2:
        grid_decarb = st.slider("Annual grid EF reduction (%/yr)", 0.0, 10.0, 2.0, 0.5, key="vm0038_decarb")

    if BEy_kg <= 0 or kwh_year <= 0:
        st.warning("Enter non-zero baseline fuel and project electricity to compute reductions.")
        return

    u_b = baseline_uncert_pct / 100.0
    u_p = project_uncert_pct / 100.0
    combined_u = float((u_b**2 + u_p**2) ** 0.5)

    records = []
    current_ef = float(eff_grid_ef)

    for y in range(1, int(years) + 1):
        year_PEy_kg = useful_kwh * current_ef
        year_BEy_kg = float(BEy_kg)
        year_ER_kg = year_BEy_kg - year_PEy_kg

        bey_unc_kg = year_BEy_kg * u_b
        pey_unc_kg = year_PEy_kg * u_p
        ery_unc_kg = abs(year_ER_kg) * combined_u

        records.append({
            "Year": y,
            "BEy (t)": year_BEy_kg / 1000.0,
            "PEy (t)": year_PEy_kg / 1000.0,
            "ER (t)": year_ER_kg / 1000.0,
            "ER ± (t)": ery_unc_kg / 1000.0,
        })

        current_ef *= (1 - grid_decarb / 100.0)

    df = pd.DataFrame(records)
    total_t = float(df["ER (t)"].sum())
    total_unc_t = float((df["ER ± (t)"] ** 2).sum() ** 0.5)

    st.markdown("### Results")
    st.dataframe(df, use_container_width=True)
    st.success(f"Total ER over {int(years)} years: **{total_t:,.3f} ± {total_unc_t:,.3f} tCO₂e**")

    chart_data = df.melt("Year", var_name="Series", value_name="tCO2e")
    chart = alt.Chart(chart_data).mark_line(point=True).encode(
        x="Year:O", y="tCO2e:Q", color="Series:N", tooltip=["Year", "Series", "tCO2e"]
    )
    st.altair_chart(chart, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download CSV", data=csv_bytes, file_name="vm0038_demo_results.csv", mime="text/csv")

    inputs = {
        "fuel_type": fuel_type,
        "fuel_use_l": fuel_use_l,
        "ef_fuel": ef_fuel,
        "include_wtt": include_wtt,
        "wtt_ef": wtt,
        "baseline_uncert_pct": baseline_uncert_pct,
        "mode": mode,
        "kwh_year": kwh_year,
        "charge_eff_pct": charge_eff,
        "ef_grid": ef_grid,
        "renewable_fraction_pct": renewable_fraction,
        "project_uncert_pct": project_uncert_pct,
        "years": int(years),
        "grid_decarb_pct_per_year": grid_decarb,
    }
    outputs = {
        "BEy_kg": BEy_kg,
        "PEy_kg_year1": PEy_kg,
        "total_ER_t": total_t,
        "total_ER_unc_t": total_unc_t,
        "annual_table": df.to_dict(orient="records"),
    }

    render_save_panel(
        methodology="VM0038 (demo) – EV Charging",
        total_tco2e=total_t,
        inputs=inputs,
        outputs=outputs,
        notes_default=f"VM0038-style demo over {int(years)} years. Total ER {total_t:,.3f} ± {total_unc_t:,.3f} tCO2e.",
    )


# ============================================================
# 2) AM0124 – Hydrogen Electrolysis (demo-style)
# ============================================================
def am0124_hydrogen_app():
    st.subheader("🟦 AM0124 (demo-style) — Hydrogen Electrolysis")

    with st.expander("📘 Overview", expanded=False):
        st.markdown(
            """
Demo-style AM0124 structure:
- Baseline EF depends on baseline tech (simple placeholder logic)
- Project emissions include grid electricity, fossil fuel, transport, and leakage proxy
- Applicability check: grid/captive ratio < 0.1 (as per your MVP rule)
This is a **demonstration** and not a substitute for official methodology implementation.
            """
        )

    c1, c2 = st.columns(2)
    with c1:
        MH2 = st.number_input("Hydrogen produced (t H₂/year)", min_value=0.0, value=1000.0, step=10.0, key="am0124_mh2")
        baseline = st.selectbox("Baseline technology", ["Coal (gasification + SMR)", "Natural Gas (SMR)", "Oil (gasification + SMR)"], key="am0124_base")
        grid_mwh = st.number_input("Grid electricity used (MWh/year)", min_value=0.0, value=500.0, key="am0124_grid")
        captive_mwh = st.number_input("Captive renewable electricity (MWh/year)", min_value=0.0, value=9500.0, key="am0124_cap")
        grid_ef = st.number_input("Grid EF (t CO₂/MWh)", min_value=0.0, value=1.3, key="am0124_grid_ef")

    with c2:
        fossil_gj = st.number_input("On-site fossil fuel use (GJ/year)", min_value=0.0, value=0.0, key="am0124_gj")
        fossil_ef = st.number_input("Fuel EF (t CO₂/GJ)", min_value=0.0, value=0.0, key="am0124_fuel_ef")
        transport_t = st.number_input("Transport emissions (t CO₂e/year)", min_value=0.0, value=0.0, key="am0124_tr")
        leak_pct = st.number_input("Hydrogen leak rate (%)", min_value=0.0, value=5.0, key="am0124_leak")
        gwp_h2 = st.number_input("GWP of H₂ (t CO₂e / t H₂)", min_value=0.0, value=5.8, key="am0124_gwp")
        years = st.number_input("Project duration (years)", min_value=1, value=10, key="am0124_years")

    # Baseline EF (your MVP logic preserved)
    EF_BL = 19.0 if "Coal" in baseline else 9.0

    ratio = (grid_mwh / captive_mwh) if captive_mwh > 0 else np.inf
    if ratio >= 0.1:
        st.warning(f"⚠️ Applicability check failed: grid/captive must be < 0.1. Current ratio = {ratio:.3f}")

    BEy = MH2 * EF_BL
    PE_ec = grid_mwh * grid_ef
    PE_fc = fossil_gj * fossil_ef if (fossil_gj > 0 and fossil_ef > 0) else 0.0
    PE_leak = MH2 * (leak_pct / 100.0) * gwp_h2
    PE_tr = transport_t
    PEy = PE_ec + PE_fc + PE_tr + PE_leak

    ERy = BEy - PEy
    total_ER = ERy * years

    st.markdown("### Results")
    cA, cB, cC, cD, cE = st.columns(5)
    cA.metric("Baseline EF", f"{EF_BL:.2f} tCO₂/tH₂")
    cB.metric("BEy", f"{BEy:,.2f} tCO₂e/yr")
    cC.metric("PEy", f"{PEy:,.2f} tCO₂e/yr")
    cD.metric("ERy", f"{ERy:,.2f} tCO₂e/yr")
    cE.metric("Total ER", f"{total_ER:,.2f} tCO₂e")

    st.markdown("### Project emissions breakdown")
    df = pd.DataFrame({
        "Component": ["Grid electricity", "On-site fossil fuel", "Transport", "Hydrogen leaks"],
        "tCO2e/year": [PE_ec, PE_fc, PE_tr, PE_leak],
    })
    st.dataframe(df, use_container_width=True)

    inputs = {
        "MH2_t_per_year": MH2,
        "baseline_tech": baseline,
        "grid_mwh": grid_mwh,
        "captive_mwh": captive_mwh,
        "grid_ef_t_per_mwh": grid_ef,
        "fossil_gj": fossil_gj,
        "fossil_ef_t_per_gj": fossil_ef,
        "transport_t": transport_t,
        "leak_pct": leak_pct,
        "gwp_h2": gwp_h2,
        "years": int(years),
        "grid_captive_ratio": float(ratio),
    }
    outputs = {
        "EF_BL": EF_BL,
        "BEy": float(BEy),
        "PE_ec": float(PE_ec),
        "PE_fc": float(PE_fc),
        "PE_tr": float(PE_tr),
        "PE_leak": float(PE_leak),
        "PEy": float(PEy),
        "ERy": float(ERy),
        "total_ER": float(total_ER),
    }

    render_save_panel(
        methodology="AM0124 (demo) – Hydrogen Electrolysis",
        total_tco2e=float(total_ER),
        inputs=inputs,
        outputs=outputs,
        notes_default=f"AM0124-style demo over {int(years)} years. Total ER {total_ER:,.2f} tCO2e. Grid/captive ratio={ratio:.3f}.",
    )


# ============================================================
# 3) VMR0007 – Solid Waste Recovery & Recycling (demo-style)
# ============================================================
BASELINE_EF = {"Plastic": 1.3, "Paper": 0.9, "Metal": 1.5, "Glass": 0.4}       # tCO2e/ton (demo)
PROJECT_EF  = {"Plastic": 0.55, "Paper": 0.35, "Metal": 0.20, "Glass": 0.15}   # tCO2e/ton (demo)
TRANSPORT_EF = 0.000102   # tCO2e per tonne-km (demo)
DIESEL_EF = 0.0027        # tCO2e per litre (demo)
ELECTRICITY_EF = 0.0009   # tCO2e per kWh (demo)

def vmr0007_app():
    st.subheader("♻ VMR0007 (demo-style) — Solid Waste Recovery & Recycling")

    with st.expander("📘 Overview", expanded=False):
        st.markdown(
            """
Demo-style structure:
- Baseline emissions: total tons × baseline EF
- Project emissions: recycling emissions + residual disposal + transport + energy
- ER = baseline − project
All factors shown here are **placeholders** for demonstration only.
            """
        )

    c1, c2 = st.columns(2)
    with c1:
        material = st.selectbox("Material", ["Plastic", "Paper", "Metal", "Glass"], key="vmr_mat")
        tons = st.number_input("Total material processed (tons/year)", min_value=0.0, value=100.0, step=0.1, key="vmr_tons")
        contamination = st.slider("Contamination (%)", 0, 40, 10, key="vmr_cont")
    with c2:
        diesel = st.number_input("Diesel used (litres/year)", min_value=0.0, value=500.0, key="vmr_diesel")
        electricity = st.number_input("Electricity (kWh/year)", min_value=0.0, value=30000.0, key="vmr_kwh")
        distance = st.number_input("Transport distance (km)", min_value=0.0, value=15.0, key="vmr_km")

    baseline_emissions = float(tons) * float(BASELINE_EF[material])

    clean_tons = float(tons) * (1 - float(contamination) / 100.0)
    residue_tons = float(tons) - clean_tons

    pe_recycling = clean_tons * float(PROJECT_EF[material])
    pe_residuals = residue_tons * float(BASELINE_EF[material])
    pe_transport = clean_tons * float(distance) * float(TRANSPORT_EF)
    pe_energy = (float(diesel) * float(DIESEL_EF)) + (float(electricity) * float(ELECTRICITY_EF))

    project_emissions = pe_recycling + pe_residuals + pe_transport + pe_energy
    er = baseline_emissions - project_emissions

    st.markdown("### Results")
    a, b, c = st.columns(3)
    a.metric("Baseline (tCO₂e/yr)", f"{baseline_emissions:,.2f}")
    b.metric("Project (tCO₂e/yr)", f"{project_emissions:,.2f}")
    c.metric("ER (tCO₂e/yr)", f"{er:,.2f}")

    st.markdown("### Breakdown")
    st.dataframe(pd.DataFrame({
        "Item": [
            "Clean recyclable tons",
            "Residual tons",
            "Project – recycling",
            "Project – residual disposal",
            "Transport emissions",
            "Energy emissions",
        ],
        "Value": [
            clean_tons,
            residue_tons,
            pe_recycling,
            pe_residuals,
            pe_transport,
            pe_energy,
        ]
    }), use_container_width=True)

    inputs = {
        "material": material,
        "tons": float(tons),
        "contamination_pct": float(contamination),
        "diesel_l": float(diesel),
        "electricity_kwh": float(electricity),
        "distance_km": float(distance),
        "factors": {
            "BASELINE_EF": BASELINE_EF[material],
            "PROJECT_EF": PROJECT_EF[material],
            "TRANSPORT_EF": TRANSPORT_EF,
            "DIESEL_EF": DIESEL_EF,
            "ELECTRICITY_EF": ELECTRICITY_EF,
        }
    }
    outputs = {
        "baseline_tco2e_per_year": baseline_emissions,
        "project_tco2e_per_year": project_emissions,
        "er_tco2e_per_year": er,
        "clean_tons": clean_tons,
        "residual_tons": residue_tons,
    }

    render_save_panel(
        methodology="VMR0007 (demo) – Solid Waste Recovery & Recycling",
        total_tco2e=float(er),  # annual ER in this MVP
        inputs=inputs,
        outputs=outputs,
        notes_default="VMR0007-style demo. Annual ER saved (not lifetime unless you multiply by years externally).",
    )


# ------------------------------------------------------------
# PAGE MAIN
# ------------------------------------------------------------
choice = st.selectbox(
    "Select methodology:",
    [
        "VM0038 (demo) – EV Charging",
        "AM0124 (demo) – Hydrogen Electrolysis",
        "VMR0007 (demo) – Solid Waste Recovery & Recycling",
    ],
)

st.divider()

if choice.startswith("VM0038"):
    vm0038_ev()
elif choice.startswith("AM0124"):
    am0124_hydrogen_app()
else:
    vmr0007_app()

st.divider()
st.caption(
    "Launch note: These are demo-style reference implementations. "
    "They prove structure + data flow + audit-ready saving, not official crediting."
)
