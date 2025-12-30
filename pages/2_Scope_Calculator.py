# pages/2_Scope_Calculator.py
# ------------------------------------------------------------
# Carbon Registry • Scope 1/2/3 Calculator (Cloud-safe, single file)
#
# Goals (v1 "guided but honest"):
# ✅ Free calculations (no project needed)
# ✅ Light guidance to help users derive activity data (tables + simple helpers)
# ✅ Direct entry mode for advanced users
# ✅ Optional: save results to Carbon Registry ledger (SQLite) with MRV traceability
# ✅ Technical rigor guardrails:
#    - EF must be provided to calculate
#    - To SAVE, EF source/reference is required
# ✅ Explicit consultant framing (so users don't confuse this with a full advisory engagement)
#
# Assumptions:
# - Your Registry page creates a `projects` table in the same SQLite DB
#   (data/carbon_registry.db) with columns:
#     project_id, project_code, project_name, status, updated_at
# - This page will create `calc_runs` if missing.
# ------------------------------------------------------------

import streamlit as st
import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd

# ------------------------------------------------------------
# Page header
# ------------------------------------------------------------
st.title("📊 Scope 1 / 2 / 3 Calculator")
st.caption("Guided calculations + optional save to MRV ledger (audit-ready).")

# ------------------------------------------------------------
# DB (SQLite) — Cloud-safe
# ------------------------------------------------------------
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

def ensure_schema() -> None:
    db_exec(
        """
        CREATE TABLE IF NOT EXISTS calc_runs (
            calc_id TEXT PRIMARY KEY,
            project_id TEXT,
            calc_type TEXT NOT NULL,      -- 'scope' | 'methodology'
            calc_name TEXT NOT NULL,
            scope_label TEXT,             -- 'Scope 1'/'Scope 2'/'Scope 3'
            period_start TEXT,
            period_end TEXT,
            baseline_tco2e REAL,
            project_tco2e REAL,
            reduction_tco2e REAL,
            inputs_json TEXT,
            outputs_json TEXT,
            factor_source TEXT,           -- user-provided reference (required when saving)
            status TEXT DEFAULT 'final',
            actor TEXT,
            created_at TEXT NOT NULL
        );
        """
    )

ensure_schema()

def list_projects() -> pd.DataFrame:
    try:
        df = db_query(
            """
            SELECT project_id, project_code, project_name, status
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
        return pd.DataFrame(columns=["project_id", "project_code", "project_name", "status", "label"])

def save_calc_run(
    *,
    project_id: str,
    calc_name: str,
    scope_label: str,
    period_start: Optional[str],
    period_end: Optional[str],
    baseline_tco2e: Optional[float],
    project_tco2e: Optional[float],
    reduction_tco2e: Optional[float],
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
            "scope",
            calc_name,
            scope_label,
            period_start,
            period_end,
            baseline_tco2e,
            project_tco2e,
            reduction_tco2e,
            json.dumps(inputs, ensure_ascii=False),
            json.dumps(outputs, ensure_ascii=False),
            factor_source.strip(),
            status,
            actor,
            ts,
        ),
    )
    return calc_id

def render_save_panel(
    *,
    calc_name: str,
    scope_label: str,
    period_start: Optional[str],
    period_end: Optional[str],
    baseline_tco2e: Optional[float],
    project_tco2e: Optional[float],
    reduction_tco2e: Optional[float],
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
) -> None:
    st.divider()
    with st.expander("💾 Save this result to a project (optional)", expanded=False):
        projs = list_projects()
        if projs.empty:
            st.info("No projects found. Create a project in the Registry page first.")
            return

        active_pid = st.session_state.get("active_project_id")
        options = projs["project_id"].tolist()
        default_idx = options.index(active_pid) if active_pid in options else 0

        pid = st.selectbox(
            "Select project",
            options=options,
            index=default_idx,
            format_func=lambda x: projs.loc[projs.project_id == x, "label"].values[0],
        )

        # Period carried from calc UI, but editable at save time
        c1, c2 = st.columns(2)
        with c1:
            ps = st.text_input("Period start (YYYY-MM-DD)", value=period_start or "")
        with c2:
            pe = st.text_input("Period end (YYYY-MM-DD)", value=period_end or "")

        factor_source = st.text_area(
            "Emission factor source / reference (required to save)",
            placeholder=(
                "Example: 'Factor from official dataset X, version Y, published YYYY-MM-DD' "
                "or 'IPCC / DEFRA / utility disclosure / national inventory table ...'"
            ),
            height=90,
        )
        status = st.selectbox("Status", ["final", "draft"], index=0)

        can_save = (baseline_tco2e is not None) and bool(factor_source.strip())
        if st.button("✅ Save to Ledger", use_container_width=True, disabled=not can_save):
            calc_id = save_calc_run(
                project_id=pid,
                calc_name=calc_name,
                scope_label=scope_label,
                period_start=ps.strip() or None,
                period_end=pe.strip() or None,
                baseline_tco2e=baseline_tco2e,
                project_tco2e=project_tco2e,
                reduction_tco2e=reduction_tco2e,
                inputs=inputs,
                outputs=outputs,
                factor_source=factor_source,
                status=status,
            )
            st.success(f"Saved to ledger ✅  calc_id = {calc_id}")

# ------------------------------------------------------------
# Calculation helpers
# ------------------------------------------------------------
def kg_to_t(kg: float) -> float:
    return kg / 1000.0

def compute_baseline_project_reduction(
    baseline_activity: float, project_activity: float, ef_kgco2e_per_unit: float
) -> Dict[str, float]:
    baseline_kg = baseline_activity * ef_kgco2e_per_unit
    project_kg = project_activity * ef_kgco2e_per_unit
    reduction_kg = baseline_kg - project_kg
    pct = (reduction_kg / baseline_kg * 100.0) if baseline_kg > 0 else 0.0
    return {
        "baseline_tco2e": kg_to_t(baseline_kg),
        "project_tco2e": kg_to_t(project_kg),
        "reduction_tco2e": kg_to_t(reduction_kg),
        "reduction_pct": pct,
    }

def metric_row(baseline_t: float, project_t: float, reduction_t: float, pct: float) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline (tCO₂e)", f"{baseline_t:,.4f}")
    c2.metric("Project (tCO₂e)", f"{project_t:,.4f}")
    c3.metric("Reduction (tCO₂e)", f"{reduction_t:,.4f}")
    c4.metric("Reduction (%)", f"{pct:.2f}%")

def require_positive_baseline(baseline: float) -> bool:
    if baseline <= 0:
        st.error("Baseline activity must be > 0 to compute reductions.")
        return False
    return True

def require_positive_ef(ef: float) -> bool:
    if ef <= 0:
        st.error("Emission factor must be > 0.")
        return False
    return True

def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

# ------------------------------------------------------------
# Light guidance blocks (consultant framing)
# ------------------------------------------------------------
def consultant_notes(scope_label: str) -> None:
    with st.expander("🧭 Consultant notes (read before publishing results)", expanded=False):
        st.markdown(
            f"""
**This calculator helps you structure and compute emissions**, but it does not replace a full MRV advisory engagement.

For **{scope_label}**, results can change materially depending on:
- **Boundary definition** (organizational + operational control, included assets/sites, time period)
- **Baseline definition** (baseline year vs typical year; data gaps; normalization)
- **Emission factor choice** (geography, year, dataset version, market vs location-based for Scope 2)
- **Evidence quality** (bills, invoices, meter exports, maintenance logs)
- **Double counting risks** (especially Scope 2/3 interactions and supplier claims)

If these decisions are unclear, treat outputs as **screening-level** until validated.
            """.strip()
        )

# ------------------------------------------------------------
# Guided activity builders
# ------------------------------------------------------------
def period_inputs(prefix: str) -> Tuple[str, str]:
    c1, c2 = st.columns(2)
    with c1:
        ps = st.text_input(f"{prefix} Period start (YYYY-MM-DD)", value="")
    with c2:
        pe = st.text_input(f"{prefix} Period end (YYYY-MM-DD)", value="")
    return ps.strip(), pe.strip()

def guidance_box(title: str, baseline_meaning: str, project_meaning: str, evidence: List[str]) -> None:
    with st.expander(f"ℹ️ What do baseline/project activity mean here? ({title})", expanded=False):
        st.markdown(f"**Baseline activity:** {baseline_meaning}")
        st.markdown(f"**Project activity:** {project_meaning}")
        st.markdown("**Typical evidence:**")
        for e in evidence:
            st.markdown(f"- {e}")

def df_default(columns: List[str], n_rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame([{c: "" for c in columns} for _ in range(n_rows)])

def sum_numeric_column(df: pd.DataFrame, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return 0.0
    vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return float(vals.sum())

# ------------------------------------------------------------
# Scope scaffolds
# ------------------------------------------------------------
SCOPE1_CATEGORIES = {
    "Diesel combustion (liters)": {"unit": "L"},
    "Petrol/Gasoline combustion (liters)": {"unit": "L"},
    "LPG combustion (kg)": {"unit": "kg"},
    "Natural gas combustion (m³)": {"unit": "m³"},
    "Refrigerant leakage (kg)": {"unit": "kg"},
    "Other (custom)": {"unit": "unit"},
}
SCOPE2_CATEGORIES = {
    "Purchased electricity (kWh)": {"unit": "kWh"},
    "Purchased electricity (MWh)": {"unit": "MWh"},
    "Purchased steam/heat/cooling (kWh)": {"unit": "kWh"},
    "Other (custom)": {"unit": "unit"},
}
SCOPE3_CATEGORIES = {
    "1. Purchased Goods & Services": {"unit": "unit"},
    "2. Capital Goods": {"unit": "unit"},
    "3. Fuel- and Energy-Related Activities (not in S1/S2)": {"unit": "unit"},
    "4. Upstream Transportation & Distribution": {"unit": "unit"},
    "5. Waste Generated in Operations": {"unit": "unit"},
    "6. Business Travel": {"unit": "unit"},
    "7. Employee Commuting": {"unit": "unit"},
    "8. Upstream Leased Assets": {"unit": "unit"},
    "9. Downstream Transportation & Distribution": {"unit": "unit"},
    "10. Processing of Sold Products": {"unit": "unit"},
    "11. Use of Sold Products": {"unit": "unit"},
    "12. End-of-Life Treatment of Sold Products": {"unit": "unit"},
    "13. Downstream Leased Assets": {"unit": "unit"},
    "14. Franchises": {"unit": "unit"},
    "15. Investments": {"unit": "unit"},
}

RESULT_KEYS = {
    "scope1": "scope_calc_last_scope1",
    "scope2": "scope_calc_last_scope2",
    "scope3": "scope_calc_last_scope3",
}

# ------------------------------------------------------------
# Scope 1 UI (guided + direct)
# ------------------------------------------------------------
def calc_scope1() -> None:
    st.header("Scope 1 – Direct emissions (fuels, gases, refrigerants)")
    consultant_notes("Scope 1")

    category = st.selectbox("Category", list(SCOPE1_CATEGORIES.keys()))
    unit = SCOPE1_CATEGORIES[category]["unit"]
    period_start, period_end = period_inputs("Scope 1")

    guidance_box(
        "Scope 1",
        baseline_meaning="Total direct fuel use / refrigerant leakage in the baseline period (often a baseline year or typical year).",
        project_meaning="Total direct fuel use / refrigerant leakage in the project period (same boundary + comparable period).",
        evidence=["Fuel invoices/receipts", "Fleet logbooks / telematics exports", "Generator runtime logs", "Maintenance records for refrigerants"],
    )

    mode = st.radio("Input mode", ["Guided (recommended)", "Direct entry (advanced)"], horizontal=True)

    baseline_activity = 0.0
    project_activity = 0.0

    if mode.startswith("Guided"):
        guided_method = st.selectbox(
            "Guided method",
            ["Fuel from invoices (table)", "Fuel from distance (simple)", "Refrigerant leakage (simple)", "Custom (manual total)"],
        )

        if guided_method == "Fuel from invoices (table)":
            st.write(f"Enter invoice line items. The calculator will sum total {unit}.")
            cols = ["date", "supplier", f"quantity_{unit}", "notes"]
            key_b = "s1_inv_baseline"
            key_p = "s1_inv_project"

            if key_b not in st.session_state:
                st.session_state[key_b] = df_default(cols)
            if key_p not in st.session_state:
                st.session_state[key_p] = df_default(cols)

            st.markdown("**Baseline invoices**")
            dfb = st.data_editor(st.session_state[key_b], key="s1_inv_baseline_editor", use_container_width=True, num_rows="dynamic")
            st.session_state[key_b] = dfb
            baseline_activity = sum_numeric_column(dfb, f"quantity_{unit}")

            st.markdown("**Project invoices**")
            dfp = st.data_editor(st.session_state[key_p], key="s1_inv_project_editor", use_container_width=True, num_rows="dynamic")
            st.session_state[key_p] = dfp
            project_activity = sum_numeric_column(dfp, f"quantity_{unit}")

            st.info(f"Derived baseline activity: **{baseline_activity:,.3f} {unit}** • project activity: **{project_activity:,.3f} {unit}**")

        elif guided_method == "Fuel from distance (simple)":
            st.write("This is a helper to convert distance to fuel quantity.")
            c1, c2 = st.columns(2)
            with c1:
                baseline_km = st.number_input("Baseline distance (km)", min_value=0.0, value=0.0)
                baseline_l_per_100 = st.number_input("Baseline fuel economy (L/100km)", min_value=0.0, value=0.0)
            with c2:
                project_km = st.number_input("Project distance (km)", min_value=0.0, value=0.0)
                project_l_per_100 = st.number_input("Project fuel economy (L/100km)", min_value=0.0, value=0.0)

            # Only meaningful if unit is liters; otherwise we still compute "units" as liters-equivalent
            baseline_activity = baseline_km * (baseline_l_per_100 / 100.0)
            project_activity = project_km * (project_l_per_100 / 100.0)

            st.info(f"Derived baseline fuel: **{baseline_activity:,.3f} L** • project fuel: **{project_activity:,.3f} L**")
            if unit != "L":
                st.warning(f"Your selected category unit is **{unit}** but this method outputs liters. Consider selecting a liters-based category or use Direct entry.")

        elif guided_method == "Refrigerant leakage (simple)":
            st.write("Simple leakage helper. In rigorous inventories, leakage can be estimated from top-ups, recoveries, and system charge.")
            c1, c2 = st.columns(2)
            with c1:
                baseline_topup = st.number_input("Baseline refrigerant top-up (kg)", min_value=0.0, value=0.0)
                baseline_recovered = st.number_input("Baseline recovered (kg) [optional]", min_value=0.0, value=0.0)
            with c2:
                project_topup = st.number_input("Project refrigerant top-up (kg)", min_value=0.0, value=0.0)
                project_recovered = st.number_input("Project recovered (kg) [optional]", min_value=0.0, value=0.0)

            # Simple proxy: leaked ≈ top-ups - recovered (bounded at >= 0)
            baseline_activity = max(0.0, baseline_topup - baseline_recovered)
            project_activity = max(0.0, project_topup - project_recovered)

            st.info(f"Derived baseline leaked: **{baseline_activity:,.3f} kg** • project leaked: **{project_activity:,.3f} kg**")
            if unit != "kg":
                st.warning(f"Your selected category unit is **{unit}** but this method outputs kg. Consider selecting a kg-based category or use Direct entry.")

        else:
            baseline_activity = st.number_input(f"Baseline activity total ({unit})", min_value=0.0, value=0.0)
            project_activity = st.number_input(f"Project activity total ({unit})", min_value=0.0, value=0.0)

    else:
        baseline_activity = st.number_input(f"Baseline activity ({unit} per period)", min_value=0.0, value=0.0)
        project_activity = st.number_input(f"Project activity ({unit} per period)", min_value=0.0, value=0.0)

    st.markdown("#### Emission factor")
    ef_kg = st.number_input(f"EF (kg CO₂e per {unit})", min_value=0.0, value=0.0)
    st.caption("Use an authoritative factor source and cite it when saving to the ledger.")

    # Optional uncertainty (light)
    with st.expander("Optional: uncertainty (screening-level)", expanded=False):
        ua = st.number_input("Activity data uncertainty (%)", min_value=0.0, value=0.0)
        uf = st.number_input("Emission factor uncertainty (%)", min_value=0.0, value=0.0)

    calc_btn = st.button("Calculate Scope 1", use_container_width=True)

    if calc_btn:
        if not require_positive_baseline(baseline_activity) or not require_positive_ef(ef_kg):
            return

        res = compute_baseline_project_reduction(baseline_activity, project_activity, ef_kg)
        st.success("Calculated.")
        metric_row(res["baseline_tco2e"], res["project_tco2e"], res["reduction_tco2e"], res["reduction_pct"])

        # Simple uncertainty propagation (screening): combine relative uncertainties in quadrature
        if ua > 0 or uf > 0:
            rel = ((ua / 100.0) ** 2 + (uf / 100.0) ** 2) ** 0.5
            approx_unc = res["baseline_tco2e"] * rel
            st.caption(f"Screening uncertainty (baseline): ± {approx_unc:,.4f} tCO₂e (combined).")

        inputs = {
            "scope": "Scope 1",
            "category": category,
            "unit": unit,
            "period_start": period_start or None,
            "period_end": period_end or None,
            "input_mode": mode,
            "baseline_activity": baseline_activity,
            "project_activity": project_activity,
            "ef_kgco2e_per_unit": ef_kg,
            "uncertainty_activity_pct": ua,
            "uncertainty_ef_pct": uf,
        }
        outputs = {**res, "scope": "Scope 1", "category": category, "unit": unit}

        # Explain the equation plainly
        with st.expander("Show calculation details", expanded=False):
            st.markdown(
                f"""
**Equation:** Emissions (tCO₂e) = Activity × EF ÷ 1000

- Baseline: {baseline_activity:,.6g} {unit} × {ef_kg:,.6g} kgCO₂e/{unit} ÷ 1000
- Project: {project_activity:,.6g} {unit} × {ef_kg:,.6g} kgCO₂e/{unit} ÷ 1000
                """.strip()
            )

        st.session_state[RESULT_KEYS["scope1"]] = {"inputs": inputs, "outputs": outputs, "period_start": period_start, "period_end": period_end}

    if RESULT_KEYS["scope1"] in st.session_state:
        last = st.session_state[RESULT_KEYS["scope1"]]
        o = last["outputs"]
        render_save_panel(
            calc_name=f"Scope 1 — {o.get('category')}",
            scope_label="Scope 1",
            period_start=last.get("period_start"),
            period_end=last.get("period_end"),
            baseline_tco2e=o.get("baseline_tco2e"),
            project_tco2e=o.get("project_tco2e"),
            reduction_tco2e=o.get("reduction_tco2e"),
            inputs=last["inputs"],
            outputs=o,
        )

# ------------------------------------------------------------
# Scope 2 UI (guided + direct)
# ------------------------------------------------------------
def calc_scope2() -> None:
    st.header("Scope 2 – Purchased energy (electricity, steam, heat, cooling)")
    consultant_notes("Scope 2")

    category = st.selectbox("Category", list(SCOPE2_CATEGORIES.keys()))
    unit = SCOPE2_CATEGORIES[category]["unit"]
    period_start, period_end = period_inputs("Scope 2")

    guidance_box(
        "Scope 2",
        baseline_meaning="Total purchased energy consumed in the baseline period (e.g., grid electricity kWh).",
        project_meaning="Total purchased energy consumed in the project period (same boundary + comparable period).",
        evidence=["Utility bills", "Meter downloads / AMI exports", "Energy management system exports", "On-site generation reports (for context)"],
    )

    mode = st.radio("Input mode", ["Guided (recommended)", "Direct entry (advanced)"], horizontal=True, key="s2_mode")

    baseline_activity = 0.0
    project_activity = 0.0

    if mode.startswith("Guided"):
        guided_method = st.selectbox(
            "Guided method",
            ["Bills / meter readings (table)", "PV displacement helper (simple)", "Custom (manual total)"],
            key="s2_method"
        )

        if guided_method == "Bills / meter readings (table)":
            st.write(f"Enter monthly/period readings; totals will be summed as {unit}.")
            cols = ["period_label", f"consumption_{unit}", "notes"]
            key_b = "s2_tbl_baseline"
            key_p = "s2_tbl_project"

            if key_b not in st.session_state:
                st.session_state[key_b] = df_default(cols)
            if key_p not in st.session_state:
                st.session_state[key_p] = df_default(cols)

            st.markdown("**Baseline bills / meter readings**")
            dfb = st.data_editor(st.session_state[key_b], key="s2_tbl_baseline_editor", use_container_width=True, num_rows="dynamic")
            st.session_state[key_b] = dfb
            baseline_activity = sum_numeric_column(dfb, f"consumption_{unit}")

            st.markdown("**Project bills / meter readings**")
            dfp = st.data_editor(st.session_state[key_p], key="s2_tbl_project_editor", use_container_width=True, num_rows="dynamic")
            st.session_state[key_p] = dfp
            project_activity = sum_numeric_column(dfp, f"consumption_{unit}")

            st.info(f"Derived baseline consumption: **{baseline_activity:,.3f} {unit}** • project consumption: **{project_activity:,.3f} {unit}**")

        elif guided_method == "PV displacement helper (simple)":
            st.write("Quick helper: project grid consumption = baseline grid consumption − PV generation used on-site.")
            baseline_grid = st.number_input(f"Baseline grid consumption ({unit})", min_value=0.0, value=0.0)
            pv_used = st.number_input(f"PV used on-site (same period, {unit})", min_value=0.0, value=0.0)
            baseline_activity = baseline_grid
            project_activity = max(0.0, baseline_grid - pv_used)
            st.info(f"Derived project grid consumption: **{project_activity:,.3f} {unit}**")

        else:
            baseline_activity = st.number_input(f"Baseline activity total ({unit})", min_value=0.0, value=0.0)
            project_activity = st.number_input(f"Project activity total ({unit})", min_value=0.0, value=0.0)

    else:
        baseline_activity = st.number_input(f"Baseline activity ({unit} per period)", min_value=0.0, value=0.0)
        project_activity = st.number_input(f"Project activity ({unit} per period)", min_value=0.0, value=0.0)

    st.markdown("#### Factor basis (for reporting rigor)")
    factor_basis = st.radio("Factor basis", ["Location-based", "Market-based", "Other/Custom"], horizontal=True)
    ef_kg = st.number_input(f"EF (kg CO₂e per {unit})", min_value=0.0, value=0.0)

    c3, c4 = st.columns(2)
    with c3:
        supplier = st.text_input("Electricity supplier / utility (optional)")
    with c4:
        grid_region = st.text_input("Grid region (optional)")

    with st.expander("Optional: uncertainty (screening-level)", expanded=False):
        ua = st.number_input("Activity data uncertainty (%)", min_value=0.0, value=0.0, key="s2_ua")
        uf = st.number_input("Emission factor uncertainty (%)", min_value=0.0, value=0.0, key="s2_uf")

    calc_btn = st.button("Calculate Scope 2", use_container_width=True)

    if calc_btn:
        if not require_positive_baseline(baseline_activity) or not require_positive_ef(ef_kg):
            return

        res = compute_baseline_project_reduction(baseline_activity, project_activity, ef_kg)
        st.success("Calculated.")
        metric_row(res["baseline_tco2e"], res["project_tco2e"], res["reduction_tco2e"], res["reduction_pct"])

        if ua > 0 or uf > 0:
            rel = ((ua / 100.0) ** 2 + (uf / 100.0) ** 2) ** 0.5
            approx_unc = res["baseline_tco2e"] * rel
            st.caption(f"Screening uncertainty (baseline): ± {approx_unc:,.4f} tCO₂e (combined).")

        inputs = {
            "scope": "Scope 2",
            "category": category,
            "unit": unit,
            "period_start": period_start or None,
            "period_end": period_end or None,
            "input_mode": mode,
            "baseline_activity": baseline_activity,
            "project_activity": project_activity,
            "factor_basis": factor_basis,
            "ef_kgco2e_per_unit": ef_kg,
            "supplier": supplier,
            "grid_region": grid_region,
            "uncertainty_activity_pct": ua,
            "uncertainty_ef_pct": uf,
        }
        outputs = {**res, "scope": "Scope 2", "category": category, "unit": unit, "factor_basis": factor_basis}

        with st.expander("Show calculation details", expanded=False):
            st.markdown(
                f"""
**Equation:** Emissions (tCO₂e) = Activity × EF ÷ 1000

- Baseline: {baseline_activity:,.6g} {unit} × {ef_kg:,.6g} kgCO₂e/{unit} ÷ 1000
- Project: {project_activity:,.6g} {unit} × {ef_kg:,.6g} kgCO₂e/{unit} ÷ 1000

**Factor basis selected:** {factor_basis}
                """.strip()
            )

        st.session_state[RESULT_KEYS["scope2"]] = {"inputs": inputs, "outputs": outputs, "period_start": period_start, "period_end": period_end}

    if RESULT_KEYS["scope2"] in st.session_state:
        last = st.session_state[RESULT_KEYS["scope2"]]
        o = last["outputs"]
        render_save_panel(
            calc_name=f"Scope 2 — {o.get('category')} ({o.get('factor_basis')})",
            scope_label="Scope 2",
            period_start=last.get("period_start"),
            period_end=last.get("period_end"),
            baseline_tco2e=o.get("baseline_tco2e"),
            project_tco2e=o.get("project_tco2e"),
            reduction_tco2e=o.get("reduction_tco2e"),
            inputs=last["inputs"],
            outputs=o,
        )

# ------------------------------------------------------------
# Scope 3 UI (guided + direct, light templates)
# ------------------------------------------------------------
def calc_scope3() -> None:
    st.header("Scope 3 – Value chain emissions (GHG Protocol categories)")
    consultant_notes("Scope 3")

    category = st.selectbox("Scope 3 category", list(SCOPE3_CATEGORIES.keys()))
    default_unit = SCOPE3_CATEGORIES[category]["unit"]
    period_start, period_end = period_inputs("Scope 3")

    guidance_box(
        "Scope 3",
        baseline_meaning="Total value-chain activity in the baseline period (depends on category and method: spend, distance, mass, supplier data).",
        project_meaning="Total value-chain activity in the project period (same boundary + comparable period).",
        evidence=["Supplier invoices / procurement extracts", "Travel booking exports", "Freight waybills", "Waste manifests", "Commuting surveys / HR data"],
    )

    mode = st.radio("Input mode", ["Guided (recommended)", "Direct entry (advanced)"], horizontal=True, key="s3_mode")

    baseline_activity = 0.0
    project_activity = 0.0
    unit = st.text_input(
        "Unit (be specific)",
        value=default_unit,
        placeholder="e.g., passenger-km, ton-km, kg, km, nights, spend-ZAR, units",
        key="s3_unit"
    )

    if mode.startswith("Guided"):
        method = st.selectbox(
            "Guided method",
            ["Spend-based (table)", "Distance-based (table)", "Mass-based (table)", "Custom (manual total)"],
            key="s3_method"
        )

        if method == "Spend-based (table)":
            st.write("Enter spend lines (currency/period). You must use an EF consistent with your spend unit.")
            cols = ["supplier/category", f"spend_{unit}", "notes"]
            key_b = "s3_spend_baseline"
            key_p = "s3_spend_project"
            if key_b not in st.session_state:
                st.session_state[key_b] = df_default(cols)
            if key_p not in st.session_state:
                st.session_state[key_p] = df_default(cols)

            st.markdown("**Baseline spend lines**")
            dfb = st.data_editor(st.session_state[key_b], key="s3_spend_baseline_editor", use_container_width=True, num_rows="dynamic")
            st.session_state[key_b] = dfb
            baseline_activity = sum_numeric_column(dfb, f"spend_{unit}")

            st.markdown("**Project spend lines**")
            dfp = st.data_editor(st.session_state[key_p], key="s3_spend_project_editor", use_container_width=True, num_rows="dynamic")
            st.session_state[key_p] = dfp
            project_activity = sum_numeric_column(dfp, f"spend_{unit}")

            st.info(f"Derived baseline spend: **{baseline_activity:,.3f} {unit}** • project spend: **{project_activity:,.3f} {unit}**")

        elif method == "Distance-based (table)":
            st.write("Enter distance lines (e.g., km, passenger-km). Ensure your EF matches your chosen unit.")
            cols = ["route/activity", f"distance_{unit}", "notes"]
            key_b = "s3_dist_baseline"
            key_p = "s3_dist_project"
            if key_b not in st.session_state:
                st.session_state[key_b] = df_default(cols)
            if key_p not in st.session_state:
                st.session_state[key_p] = df_default(cols)

            st.markdown("**Baseline distance lines**")
            dfb = st.data_editor(st.session_state[key_b], key="s3_dist_baseline_editor", use_container_width=True, num_rows="dynamic")
            st.session_state[key_b] = dfb
            baseline_activity = sum_numeric_column(dfb, f"distance_{unit}")

            st.markdown("**Project distance lines**")
            dfp = st.data_editor(st.session_state[key_p], key="s3_dist_project_editor", use_container_width=True, num_rows="dynamic")
            st.session_state[key_p] = dfp
            project_activity = sum_numeric_column(dfp, f"distance_{unit}")

            st.info(f"Derived baseline distance: **{baseline_activity:,.3f} {unit}** • project distance: **{project_activity:,.3f} {unit}**")

        elif method == "Mass-based (table)":
            st.write("Enter mass lines (e.g., kg, tonnes). Ensure EF matches your mass unit.")
            cols = ["material/waste type", f"mass_{unit}", "notes"]
            key_b = "s3_mass_baseline"
            key_p = "s3_mass_project"
            if key_b not in st.session_state:
                st.session_state[key_b] = df_default(cols)
            if key_p not in st.session_state:
                st.session_state[key_p] = df_default(cols)

            st.markdown("**Baseline mass lines**")
            dfb = st.data_editor(st.session_state[key_b], key="s3_mass_baseline_editor", use_container_width=True, num_rows="dynamic")
            st.session_state[key_b] = dfb
            baseline_activity = sum_numeric_column(dfb, f"mass_{unit}")

            st.markdown("**Project mass lines**")
            dfp = st.data_editor(st.session_state[key_p], key="s3_mass_project_editor", use_container_width=True, num_rows="dynamic")
            st.session_state[key_p] = dfp
            project_activity = sum_numeric_column(dfp, f"mass_{unit}")

            st.info(f"Derived baseline mass: **{baseline_activity:,.3f} {unit}** • project mass: **{project_activity:,.3f} {unit}**")

        else:
            baseline_activity = st.number_input(f"Baseline activity total ({unit})", min_value=0.0, value=0.0, key="s3_base_custom")
            project_activity = st.number_input(f"Project activity total ({unit})", min_value=0.0, value=0.0, key="s3_proj_custom")

    else:
        c1, c2 = st.columns(2)
        with c1:
            baseline_activity = st.number_input(f"Baseline activity ({unit} per period)", min_value=0.0, value=0.0, key="s3_base")
        with c2:
            project_activity = st.number_input(f"Project activity ({unit} per period)", min_value=0.0, value=0.0, key="s3_proj")

    st.markdown("#### Emission factor")
    ef_kg = st.number_input(f"EF (kg CO₂e per {unit})", min_value=0.0, value=0.0, key="s3_ef")

    c5, c6 = st.columns(2)
    with c5:
        activity_method = st.selectbox(
            "Activity data method (metadata)",
            ["", "Spend-based", "Distance-based", "Mass-based", "Supplier-specific", "Hybrid"],
            key="s3_meta_method",
        )
    with c6:
        boundary_note = st.text_input("Boundary note (optional)", placeholder="e.g., upstream logistics only", key="s3_boundary")

    with st.expander("Optional: uncertainty (screening-level)", expanded=False):
        ua = st.number_input("Activity data uncertainty (%)", min_value=0.0, value=0.0, key="s3_ua")
        uf = st.number_input("Emission factor uncertainty (%)", min_value=0.0, value=0.0, key="s3_uf")

    calc_btn = st.button("Calculate Scope 3", use_container_width=True)

    if calc_btn:
        if not require_positive_baseline(baseline_activity) or not require_positive_ef(ef_kg):
            return

        res = compute_baseline_project_reduction(baseline_activity, project_activity, ef_kg)
        st.success("Calculated.")
        metric_row(res["baseline_tco2e"], res["project_tco2e"], res["reduction_tco2e"], res["reduction_pct"])

        if ua > 0 or uf > 0:
            rel = ((ua / 100.0) ** 2 + (uf / 100.0) ** 2) ** 0.5
            approx_unc = res["baseline_tco2e"] * rel
            st.caption(f"Screening uncertainty (baseline): ± {approx_unc:,.4f} tCO₂e (combined).")

        inputs = {
            "scope": "Scope 3",
            "category": category,
            "unit": unit,
            "period_start": period_start or None,
            "period_end": period_end or None,
            "input_mode": mode,
            "baseline_activity": baseline_activity,
            "project_activity": project_activity,
            "ef_kgco2e_per_unit": ef_kg,
            "activity_method_meta": activity_method,
            "boundary_note": boundary_note,
            "uncertainty_activity_pct": ua,
            "uncertainty_ef_pct": uf,
        }
        outputs = {**res, "scope": "Scope 3", "category": category, "unit": unit}

        with st.expander("Show calculation details", expanded=False):
            st.markdown(
                f"""
**Equation:** Emissions (tCO₂e) = Activity × EF ÷ 1000

- Baseline: {baseline_activity:,.6g} {unit} × {ef_kg:,.6g} kgCO₂e/{unit} ÷ 1000
- Project: {project_activity:,.6g} {unit} × {ef_kg:,.6g} kgCO₂e/{unit} ÷ 1000
                """.strip()
            )

        st.session_state[RESULT_KEYS["scope3"]] = {"inputs": inputs, "outputs": outputs, "period_start": period_start, "period_end": period_end}

    if RESULT_KEYS["scope3"] in st.session_state:
        last = st.session_state[RESULT_KEYS["scope3"]]
        o = last["outputs"]
        render_save_panel(
            calc_name=f"Scope 3 — {o.get('category')}",
            scope_label="Scope 3",
            period_start=last.get("period_start"),
            period_end=last.get("period_end"),
            baseline_tco2e=o.get("baseline_tco2e"),
            project_tco2e=o.get("project_tco2e"),
            reduction_tco2e=o.get("reduction_tco2e"),
            inputs=last["inputs"],
            outputs=o,
        )

# ------------------------------------------------------------
# MAIN NAV
# ------------------------------------------------------------
st.divider()
choice = st.radio(
    "Select scope:",
    ["Scope 1 – Direct", "Scope 2 – Purchased Energy", "Scope 3 – Value Chain"],
    horizontal=True,
)

if choice.startswith("Scope 1"):
    calc_scope1()
elif choice.startswith("Scope 2"):
    calc_scope2()
else:
    calc_scope3()

st.divider()
st.caption(
    "Rigor note: You must input the emission factor used. "
    "When saving to the ledger, provide the factor source/reference for auditability. "
    "Guided inputs help structure data, but do not replace full MRV consulting."
)


