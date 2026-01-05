# pages/2_Scope_Calculator.py
# ------------------------------------------------------------
# Carbon Registry • Scope 1/2/3 Calculator (Cloud-safe, single file)
#
# v1.2 "guided but honest" upgrades:
# ✅ Page config + shared CSS
# ✅ Save -> calc_runs + audit_logs entry (registry consistency)
# ✅ Baseline=0 handled (no forced error; % becomes N/A)
# ✅ Notes + better run naming
# ✅ Minor unit safety warnings
#
# IMPORTANT:
# - This tool does NOT ship official EF values. Users must input EF and cite a source.
# - Guidance here is structural and educational; verification depends on your standard/methodology.
# ------------------------------------------------------------

import streamlit as st
import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd

from utils.load_css import load_css

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
from utils.ui import setup_page, render_hero

setup_page(page_title="Carbon Registry • Scopes", page_icon="📊", layout="wide")

render_hero(
    title="📊 Scope 1 / 2 / 3 Calculator",
    subtitle_html="Baseline estimates across scopes with transparent factors + assumptions.",
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
            factor_source TEXT,           -- required when saving
            status TEXT DEFAULT 'final',
            actor TEXT,
            created_at TEXT NOT NULL
        );
        """
    )

    # If registry page already created this, no harm.
    db_exec(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            audit_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            actor TEXT,
            action TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT,
            project_id TEXT,
            before_json TEXT,
            after_json TEXT,
            meta_json TEXT
        );
        """
    )

ensure_schema()

# ------------------------------------------------------------
# AUDIT
# ------------------------------------------------------------
def audit_log(
    action: str,
    entity_type: str,
    entity_id: Optional[str] = None,
    project_id: Optional[str] = None,
    before: Optional[Dict[str, Any]] = None,
    after: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    actor = st.session_state.get("actor_name", "unknown")
    db_exec(
        """
        INSERT INTO audit_logs (audit_id, timestamp, actor, action, entity_type, entity_id, project_id, before_json, after_json, meta_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            now_iso(),
            actor,
            action,
            entity_type,
            entity_id,
            project_id,
            json.dumps(before, ensure_ascii=False) if before else None,
            json.dumps(after, ensure_ascii=False) if after else None,
            json.dumps(meta, ensure_ascii=False) if meta else None,
        ),
    )

# ------------------------------------------------------------
# Projects + save helpers
# ------------------------------------------------------------
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

    # Audit log entry
    audit_log(
        action="CREATE",
        entity_type="calc_run",
        entity_id=calc_id,
        project_id=project_id,
        after={
            "calc_name": calc_name,
            "scope_label": scope_label,
            "period_start": period_start,
            "period_end": period_end,
            "baseline_tco2e": baseline_tco2e,
            "project_tco2e": project_tco2e,
            "reduction_tco2e": reduction_tco2e,
            "factor_source": factor_source.strip(),
            "status": status,
        },
        meta={"calc_type": "scope"},
    )

    return calc_id

def render_save_panel(
    *,
    calc_name_default: str,
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

        c1, c2 = st.columns(2)
        with c1:
            ps = st.text_input("Period start (YYYY-MM-DD)", value=period_start or "")
        with c2:
            pe = st.text_input("Period end (YYYY-MM-DD)", value=period_end or "")

        calc_name = st.text_input("Run name", value=calc_name_default)

        notes = st.text_area("Notes (optional)", height=80, placeholder="Any caveats, boundary notes, missing data, estimation method...")

        factor_source = st.text_area(
            "Emission factor source / reference (required to save)",
            placeholder=(
                "Dataset/standard name, version, year, geography/boundary. "
                "If Scope 2, also specify market vs location basis. "
                "Include URL or document reference where possible."
            ),
            height=90,
        )
        status = st.selectbox("Status", ["final", "draft"], index=0)

        # Attach notes in inputs for traceability
        inputs_to_save = dict(inputs)
        inputs_to_save["run_notes"] = notes.strip() or None

        can_save = (baseline_tco2e is not None) and bool(factor_source.strip()) and bool(calc_name.strip())
        if st.button("✅ Save to Ledger", use_container_width=True, disabled=not can_save):
            calc_id = save_calc_run(
                project_id=pid,
                calc_name=calc_name.strip(),
                scope_label=scope_label,
                period_start=ps.strip() or None,
                period_end=pe.strip() or None,
                baseline_tco2e=baseline_tco2e,
                project_tco2e=project_tco2e,
                reduction_tco2e=reduction_tco2e,
                inputs=inputs_to_save,
                outputs=outputs,
                factor_source=factor_source,
                status=status,
            )
            st.success(f"Saved to ledger ✅  calc_id = {calc_id}")

# ------------------------------------------------------------
# Core math helpers
# ------------------------------------------------------------
def kg_to_t(kg: float) -> float:
    return kg / 1000.0

def compute_baseline_project_reduction(
    baseline_activity: float, project_activity: float, ef_kgco2e_per_unit: float
) -> Dict[str, Optional[float]]:
    baseline_kg = baseline_activity * ef_kgco2e_per_unit
    project_kg = project_activity * ef_kgco2e_per_unit
    reduction_kg = baseline_kg - project_kg

    reduction_pct = None
    if baseline_kg > 0:
        reduction_pct = (reduction_kg / baseline_kg * 100.0)

    return {
        "baseline_tco2e": kg_to_t(baseline_kg),
        "project_tco2e": kg_to_t(project_kg),
        "reduction_tco2e": kg_to_t(reduction_kg),
        "reduction_pct": reduction_pct,
    }

def metric_row(baseline_t: float, project_t: float, reduction_t: float, pct: Optional[float]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline (tCO₂e)", f"{baseline_t:,.4f}")
    c2.metric("Project (tCO₂e)", f"{project_t:,.4f}")
    c3.metric("Reduction (tCO₂e)", f"{reduction_t:,.4f}")
    c4.metric("Reduction (%)", "N/A" if pct is None else f"{pct:.2f}%")

def require_nonnegative_activity(x: float, label: str) -> bool:
    if x < 0:
        st.error(f"{label} must be ≥ 0.")
        return False
    return True

def require_positive_ef(ef: float) -> bool:
    if ef <= 0:
        st.error("Emission factor must be > 0.")
        return False
    return True

# ------------------------------------------------------------
# Uncertainty helpers (screening-level propagation)
# ------------------------------------------------------------
UNCERTAINTY_BANDS = {
    "High quality (±2%)": 2.0,
    "Good (±5%)": 5.0,
    "Medium (±10%)": 10.0,
    "Low (±20%)": 20.0,
    "Very low (±30%)": 30.0,
    "Custom": None,
}

def combined_rel_uncertainty(*rel_uncertainties: float) -> float:
    return (sum((u ** 2 for u in rel_uncertainties))) ** 0.5

def apply_uncertainty(value: float, rel_u: float) -> Tuple[float, float, float]:
    delta = abs(value) * rel_u
    return value - delta, value, value + delta

def uncertainty_panel(scope_key: str) -> Dict[str, float]:
    with st.expander("📉 Uncertainty (screening-level, optional)", expanded=False):
        st.caption("Use this to express data quality for internal screening and MRV readiness.")

        c1, c2, c3 = st.columns(3)
        with c1:
            b_band = st.selectbox("Baseline activity data quality", list(UNCERTAINTY_BANDS.keys()), index=2, key=f"{scope_key}_b_band")
        with c2:
            p_band = st.selectbox("Project activity data quality", list(UNCERTAINTY_BANDS.keys()), index=2, key=f"{scope_key}_p_band")
        with c3:
            ef_band = st.selectbox("Emission factor data quality", list(UNCERTAINTY_BANDS.keys()), index=2, key=f"{scope_key}_ef_band")

        def band_to_pct(band: str, default_key: str) -> float:
            if band != "Custom":
                return float(UNCERTAINTY_BANDS[band])
            return float(st.number_input(f"{default_key} uncertainty (%)", min_value=0.0, value=10.0, key=f"{scope_key}_{default_key}_custom"))

        b_u = band_to_pct(b_band, "baseline_activity")
        p_u = band_to_pct(p_band, "project_activity")
        ef_u = band_to_pct(ef_band, "ef")

        b_rel = b_u / 100.0
        p_rel = p_u / 100.0
        ef_rel = ef_u / 100.0

        baseline_rel = combined_rel_uncertainty(b_rel, ef_rel)
        project_rel = combined_rel_uncertainty(p_rel, ef_rel)

        st.caption("Reduction uncertainty computed after calculation (baseline/project combined).")

        return {
            "baseline_activity_u_pct": b_u,
            "project_activity_u_pct": p_u,
            "ef_u_pct": ef_u,
            "baseline_rel_u": baseline_rel,
            "project_rel_u": project_rel,
        }

def render_uncertainty_results(baseline_t: float, project_t: float, reduction_t: float, u_meta: Dict[str, float]) -> Dict[str, Any]:
    baseline_rel = u_meta.get("baseline_rel_u", 0.0)
    project_rel = u_meta.get("project_rel_u", 0.0)

    b_lo, b_nom, b_hi = apply_uncertainty(baseline_t, baseline_rel)
    p_lo, p_nom, p_hi = apply_uncertainty(project_t, project_rel)

    b_abs = abs(b_hi - b_nom)
    p_abs = abs(p_hi - p_nom)

    red_abs = (b_abs**2 + p_abs**2) ** 0.5
    red_rel = (red_abs / abs(reduction_t)) if abs(reduction_t) > 1e-12 else None

    st.markdown("##### Uncertainty summary (screening-level)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline ± (tCO₂e)", f"{b_nom:,.4f} ± {b_abs:,.4f}")
    c2.metric("Project ± (tCO₂e)", f"{p_nom:,.4f} ± {p_abs:,.4f}")
    c3.metric("Reduction ± (tCO₂e)", f"{reduction_t:,.4f} ± {red_abs:,.4f}")
    if red_rel is None:
        st.caption("Note: Reduction relative uncertainty not shown because reduction is ~0.")

    return {
        "baseline_rel_u": baseline_rel,
        "project_rel_u": project_rel,
        "baseline_abs_u_tco2e": b_abs,
        "project_abs_u_tco2e": p_abs,
        "reduction_abs_u_tco2e": red_abs,
        "reduction_rel_u": red_rel,
    }

# ------------------------------------------------------------
# EF guidance + metadata (direction, not numbers)
# ------------------------------------------------------------
EF_SOURCES_GENERAL = [
    "IPCC Guidelines / national inventory factors (country-specific)",
    "National utility disclosures / grid operator factors (Scope 2)",
    "Government conversion factor publications (e.g., UK Gov GHG Conversion Factors)",
    "US EPA (eGRID for electricity; other published factors)",
    "GHG Protocol guidance (factor selection principles, market vs location-based)",
    "Supplier-specific data (EPDs, LCAs) for Scope 3 where available",
]

def ef_guidance_panel(scope_label: str, unit: str) -> Dict[str, Any]:
    with st.expander("🧾 Emission factor guidance + metadata (recommended)", expanded=True):
        st.markdown(
            f"""
**You must input an emission factor (EF)** in units of **kgCO₂e per {unit}**.

This tool does not provide official factors. Use a credible dataset and record:
- **Source name**
- **Year / version**
- **Geography / boundary**
- **Any basis choice** (Scope 2 location vs market)
            """.strip()
        )

        st.markdown("**Common reputable sources (examples):**")
        for s in EF_SOURCES_GENERAL:
            st.markdown(f"- {s}")

        c1, c2 = st.columns(2)
        with c1:
            ef_source_name = st.text_input("EF source name (dataset/standard/utility)", value="")
            ef_year = st.text_input("EF year / version", value="")
        with c2:
            ef_geography = st.text_input("EF geography / boundary (e.g., country, grid region)", value="")
            ef_notes = st.text_input("EF notes (optional)", value="")

        st.caption("Sanity checks are heuristics only (do not override authoritative sources).")
        sanity_on = st.checkbox("Enable basic sanity warnings", value=True)

        return {
            "ef_scope_label": scope_label,
            "ef_unit": f"kgCO2e per {unit}",
            "ef_source_name": ef_source_name.strip() or None,
            "ef_year_version": ef_year.strip() or None,
            "ef_geography_boundary": ef_geography.strip() or None,
            "ef_notes": ef_notes.strip() or None,
            "ef_sanity_warnings_enabled": sanity_on,
        }

def ef_sanity_warnings(unit: str, ef_kg_per_unit: float) -> None:
    if ef_kg_per_unit <= 0:
        return
    if ef_kg_per_unit > 1e4:
        st.warning("EF is extremely large. Check units (kg vs t, kWh vs MWh, liters vs gallons, etc.).")
    if 0 < ef_kg_per_unit < 1e-6:
        st.warning("EF is extremely small. Check units (kg vs g) and activity unit consistency.")

# ------------------------------------------------------------
# Light guidance blocks
# ------------------------------------------------------------
def consultant_notes(scope_label: str) -> None:
    with st.expander("🧭 Consultant notes (read before publishing results)", expanded=False):
        st.markdown(
            f"""
For **{scope_label}**, results can change materially depending on:
- Boundary definition
- Baseline definition
- EF choice (geography/year/version; Scope 2 market vs location)
- Evidence quality
- Double-counting risks

Treat outputs as **screening-level** until validated.
            """.strip()
        )

# ------------------------------------------------------------
# Builders
# ------------------------------------------------------------
def period_inputs(prefix: str) -> Tuple[str, str]:
    c1, c2 = st.columns(2)
    with c1:
        ps = st.text_input(f"{prefix} Period start (YYYY-MM-DD)", value="")
    with c2:
        pe = st.text_input(f"{prefix} Period end (YYYY-MM-DD)", value="")
    return ps.strip(), pe.strip()

def guidance_box(title: str, baseline_meaning: str, project_meaning: str, evidence: List[str]) -> None:
    with st.expander(f"ℹ️ Baseline vs project activity meaning ({title})", expanded=False):
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
# Calculation core
# ------------------------------------------------------------
def compute_and_render(scope_label: str, category: str, unit: str,
                       period_start: str, period_end: str,
                       baseline_activity: float, project_activity: float, ef_kg: float,
                       inputs_extra: Dict[str, Any]) -> None:

    if not require_nonnegative_activity(baseline_activity, "Baseline activity"):
        return
    if not require_nonnegative_activity(project_activity, "Project activity"):
        return
    if not require_positive_ef(ef_kg):
        return

    res = compute_baseline_project_reduction(baseline_activity, project_activity, ef_kg)
    st.success("Calculated.")
    metric_row(res["baseline_tco2e"], res["project_tco2e"], res["reduction_tco2e"], res["reduction_pct"])

    u_meta = inputs_extra.get("uncertainty", {})
    u_results = render_uncertainty_results(res["baseline_tco2e"], res["project_tco2e"], res["reduction_tco2e"], u_meta)

    inputs = {
        "scope": scope_label,
        "category": category,
        "unit": unit,
        "period_start": period_start or None,
        "period_end": period_end or None,
        "baseline_activity": baseline_activity,
        "project_activity": project_activity,
        "ef_kgco2e_per_unit": ef_kg,
        **inputs_extra,
    }
    outputs = {**res, "scope": scope_label, "category": category, "unit": unit, "uncertainty_results": u_results}

    with st.expander("Show calculation details", expanded=False):
        st.markdown(
            f"""
**Equation:** Emissions (tCO₂e) = Activity × EF ÷ 1000

- Baseline: {baseline_activity:,.6g} {unit} × {ef_kg:,.6g} kgCO₂e/{unit} ÷ 1000
- Project: {project_activity:,.6g} {unit} × {ef_kg:,.6g} kgCO₂e/{unit} ÷ 1000
            """.strip()
        )
        if res["reduction_pct"] is None:
            st.caption("Reduction % is N/A because baseline emissions are 0.")

    key = RESULT_KEYS["scope1"] if scope_label == "Scope 1" else RESULT_KEYS["scope2"] if scope_label == "Scope 2" else RESULT_KEYS["scope3"]
    st.session_state[key] = {"inputs": inputs, "outputs": outputs, "period_start": period_start, "period_end": period_end}

# ------------------------------------------------------------
# Scope 1 / 2 / 3 pages (your structure kept)
# ------------------------------------------------------------
def calc_scope1() -> None:
    st.header("Scope 1 – Direct emissions (fuels, gases, refrigerants)")
    consultant_notes("Scope 1")

    category = st.selectbox("Category", list(SCOPE1_CATEGORIES.keys()))
    unit = SCOPE1_CATEGORIES[category]["unit"]
    period_start, period_end = period_inputs("Scope 1")

    guidance_box(
        "Scope 1",
        baseline_meaning="Total direct fuel use / refrigerant leakage in the baseline period (often baseline year or typical year).",
        project_meaning="Total direct fuel use / refrigerant leakage in the project period (same boundary + comparable period).",
        evidence=["Fuel invoices/receipts", "Fleet logbooks / telematics exports", "Generator runtime logs", "Maintenance records for refrigerants"],
    )

    mode = st.radio("Input mode", ["Guided (recommended)", "Direct entry (advanced)"], horizontal=True, key="s1_mode")

    baseline_activity = 0.0
    project_activity = 0.0
    guided_method = None

    if mode.startswith("Guided"):
        guided_method = st.selectbox(
            "Guided method",
            ["Fuel from invoices (table)", "Fuel from distance (simple)", "Refrigerant leakage (simple)", "Custom (manual total)"],
            key="s1_method"
        )

        if guided_method == "Fuel from invoices (table)":
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

            st.info(f"Derived baseline: **{baseline_activity:,.3f} {unit}** • project: **{project_activity:,.3f} {unit}**")

        elif guided_method == "Fuel from distance (simple)":
            st.write("Helper: litres = km × (L/100km) ÷ 100")
            baseline_km = st.number_input("Baseline distance (km)", min_value=0.0, value=0.0)
            baseline_l_per_100 = st.number_input("Baseline fuel economy (L/100km)", min_value=0.0, value=0.0)
            project_km = st.number_input("Project distance (km)", min_value=0.0, value=0.0)
            project_l_per_100 = st.number_input("Project fuel economy (L/100km)", min_value=0.0, value=0.0)

            baseline_activity = baseline_km * (baseline_l_per_100 / 100.0)
            project_activity = project_km * (project_l_per_100 / 100.0)

            st.info(f"Derived baseline fuel: **{baseline_activity:,.3f} L** • project fuel: **{project_activity:,.3f} L**")
            if unit != "L":
                st.error("This method outputs liters. Choose a liters-based category (diesel/petrol) or use Direct entry.")
                return

        elif guided_method == "Refrigerant leakage (simple)":
            baseline_topup = st.number_input("Baseline refrigerant top-up (kg)", min_value=0.0, value=0.0)
            baseline_recovered = st.number_input("Baseline recovered (kg) [optional]", min_value=0.0, value=0.0)
            project_topup = st.number_input("Project refrigerant top-up (kg)", min_value=0.0, value=0.0)
            project_recovered = st.number_input("Project recovered (kg) [optional]", min_value=0.0, value=0.0)

            baseline_activity = max(0.0, baseline_topup - baseline_recovered)
            project_activity = max(0.0, project_topup - project_recovered)

            st.info(f"Derived baseline leaked: **{baseline_activity:,.3f} kg** • project leaked: **{project_activity:,.3f} kg**")
            if unit != "kg":
                st.error("This method outputs kg. Choose a kg-based category (refrigerants/LPG) or use Direct entry.")
                return
        else:
            baseline_activity = st.number_input(f"Baseline activity total ({unit})", min_value=0.0, value=0.0, key="s1_base_custom")
            project_activity = st.number_input(f"Project activity total ({unit})", min_value=0.0, value=0.0, key="s1_proj_custom")
    else:
        baseline_activity = st.number_input(f"Baseline activity ({unit} per period)", min_value=0.0, value=0.0, key="s1_base")
        project_activity = st.number_input(f"Project activity ({unit} per period)", min_value=0.0, value=0.0, key="s1_proj")

    ef_meta = ef_guidance_panel("Scope 1", unit)
    u_meta = uncertainty_panel("s1")

    st.markdown("#### Emission factor")
    ef_kg = st.number_input(f"EF (kg CO₂e per {unit})", min_value=0.0, value=0.0, key="s1_ef")
    if ef_meta.get("ef_sanity_warnings_enabled"):
        ef_sanity_warnings(unit, ef_kg)

    if st.button("Calculate Scope 1", use_container_width=True, key="s1_calc"):
        compute_and_render(
            "Scope 1", category, unit, period_start, period_end,
            baseline_activity, project_activity, ef_kg,
            {
                "input_mode": mode,
                "guided_method": guided_method,
                "ef_metadata": ef_meta,
                "uncertainty": u_meta,
            }
        )

    if RESULT_KEYS["scope1"] in st.session_state:
        last = st.session_state[RESULT_KEYS["scope1"]]
        o = last["outputs"]
        calc_name_default = f"Scope 1 — {o.get('category')} — {last.get('period_start','') or 'period'}"
        render_save_panel(
            calc_name_default=calc_name_default,
            scope_label="Scope 1",
            period_start=last.get("period_start"),
            period_end=last.get("period_end"),
            baseline_tco2e=o.get("baseline_tco2e"),
            project_tco2e=o.get("project_tco2e"),
            reduction_tco2e=o.get("reduction_tco2e"),
            inputs=last["inputs"],
            outputs=o,
        )

def calc_scope2() -> None:
    st.header("Scope 2 – Purchased energy (electricity, steam, heat, cooling)")
    consultant_notes("Scope 2")

    category = st.selectbox("Category", list(SCOPE2_CATEGORIES.keys()), key="s2_cat")
    unit = SCOPE2_CATEGORIES[category]["unit"]
    period_start, period_end = period_inputs("Scope 2")

    guidance_box(
        "Scope 2",
        baseline_meaning="Total purchased energy consumed in the baseline period (e.g., grid electricity).",
        project_meaning="Total purchased energy consumed in the project period (same boundary + comparable period).",
        evidence=["Utility bills", "Meter downloads", "Energy management exports"],
    )

    mode = st.radio("Input mode", ["Guided (recommended)", "Direct entry (advanced)"], horizontal=True, key="s2_mode")
    baseline_activity = 0.0
    project_activity = 0.0
    guided_method = None

    if mode.startswith("Guided"):
        guided_method = st.selectbox("Guided method", ["Bills / meter readings (table)", "PV displacement helper (simple)", "Custom (manual total)"], key="s2_method")

        if guided_method == "Bills / meter readings (table)":
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

            st.info(f"Derived baseline: **{baseline_activity:,.3f} {unit}** • project: **{project_activity:,.3f} {unit}**")

        elif guided_method == "PV displacement helper (simple)":
            baseline_grid = st.number_input(f"Baseline grid consumption ({unit})", min_value=0.0, value=0.0, key="s2_base_grid")
            pv_used = st.number_input(f"PV used on-site (same period, {unit})", min_value=0.0, value=0.0, key="s2_pv_used")
            baseline_activity = baseline_grid
            project_activity = max(0.0, baseline_grid - pv_used)
            st.info(f"Derived project grid consumption: **{project_activity:,.3f} {unit}**")
        else:
            baseline_activity = st.number_input(f"Baseline activity total ({unit})", min_value=0.0, value=0.0, key="s2_base_custom")
            project_activity = st.number_input(f"Project activity total ({unit})", min_value=0.0, value=0.0, key="s2_proj_custom")
    else:
        baseline_activity = st.number_input(f"Baseline activity ({unit} per period)", min_value=0.0, value=0.0, key="s2_base")
        project_activity = st.number_input(f"Project activity ({unit} per period)", min_value=0.0, value=0.0, key="s2_proj")

    st.markdown("#### Factor basis (Scope 2 reporting)")
    factor_basis = st.radio("Factor basis", ["Location-based", "Market-based", "Other/Custom"], horizontal=True, key="s2_basis")

    ef_meta = ef_guidance_panel("Scope 2", unit)
    ef_meta = {**ef_meta, "factor_basis": factor_basis}

    u_meta = uncertainty_panel("s2")

    st.markdown("#### Emission factor")
    ef_kg = st.number_input(f"EF (kg CO₂e per {unit})", min_value=0.0, value=0.0, key="s2_ef")
    if ef_meta.get("ef_sanity_warnings_enabled"):
        ef_sanity_warnings(unit, ef_kg)

    c3, c4 = st.columns(2)
    with c3:
        supplier = st.text_input("Electricity supplier / utility (optional)", key="s2_supplier")
    with c4:
        grid_region = st.text_input("Grid region (optional)", key="s2_region")

    if st.button("Calculate Scope 2", use_container_width=True, key="s2_calc"):
        compute_and_render(
            "Scope 2", category, unit, period_start, period_end,
            baseline_activity, project_activity, ef_kg,
            {
                "input_mode": mode,
                "guided_method": guided_method,
                "factor_basis": factor_basis,
                "supplier": supplier,
                "grid_region": grid_region,
                "ef_metadata": ef_meta,
                "uncertainty": u_meta,
            }
        )

    if RESULT_KEYS["scope2"] in st.session_state:
        last = st.session_state[RESULT_KEYS["scope2"]]
        o = last["outputs"]
        calc_name_default = f"Scope 2 — {o.get('category')} — {o.get('factor_basis','')} — {last.get('period_start','') or 'period'}"
        render_save_panel(
            calc_name_default=calc_name_default,
            scope_label="Scope 2",
            period_start=last.get("period_start"),
            period_end=last.get("period_end"),
            baseline_tco2e=o.get("baseline_tco2e"),
            project_tco2e=o.get("project_tco2e"),
            reduction_tco2e=o.get("reduction_tco2e"),
            inputs=last["inputs"],
            outputs=o,
        )

def calc_scope3() -> None:
    st.header("Scope 3 – Value chain emissions (GHG Protocol categories)")
    consultant_notes("Scope 3")

    category = st.selectbox("Scope 3 category", list(SCOPE3_CATEGORIES.keys()), key="s3_cat")
    default_unit = SCOPE3_CATEGORIES[category]["unit"]
    period_start, period_end = period_inputs("Scope 3")

    guidance_box(
        "Scope 3",
        baseline_meaning="Total value-chain activity in the baseline period (depends on category + method).",
        project_meaning="Total value-chain activity in the project period (comparable period).",
        evidence=["Procurement extracts", "Travel booking exports", "Freight waybills", "Waste manifests", "Commuting surveys"],
    )

    unit = st.text_input("Unit (be specific)", value=default_unit, key="s3_unit")

    mode = st.radio("Input mode", ["Guided (recommended)", "Direct entry (advanced)"], horizontal=True, key="s3_mode")
    baseline_activity = 0.0
    project_activity = 0.0
    guided_method = None

    if mode.startswith("Guided"):
        guided_method = st.selectbox("Guided method", ["Spend-based (table)", "Distance-based (table)", "Mass-based (table)", "Custom (manual total)"], key="s3_method")

        if guided_method == "Spend-based (table)":
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

        elif guided_method == "Distance-based (table)":
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

        elif guided_method == "Mass-based (table)":
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
        else:
            baseline_activity = st.number_input(f"Baseline activity total ({unit})", min_value=0.0, value=0.0, key="s3_base_custom")
            project_activity = st.number_input(f"Project activity total ({unit})", min_value=0.0, value=0.0, key="s3_proj_custom")
    else:
        baseline_activity = st.number_input(f"Baseline activity ({unit} per period)", min_value=0.0, value=0.0, key="s3_base")
        project_activity = st.number_input(f"Project activity ({unit} per period)", min_value=0.0, value=0.0, key="s3_proj")

    ef_meta = ef_guidance_panel("Scope 3", unit)
    u_meta = uncertainty_panel("s3")

    st.markdown("#### Emission factor")
    ef_kg = st.number_input(f"EF (kg CO₂e per {unit})", min_value=0.0, value=0.0, key="s3_ef")
    if ef_meta.get("ef_sanity_warnings_enabled"):
        ef_sanity_warnings(unit, ef_kg)

    c5, c6 = st.columns(2)
    with c5:
        activity_method = st.selectbox("Activity data method (metadata)", ["", "Spend-based", "Distance-based", "Mass-based", "Supplier-specific", "Hybrid"], key="s3_meta_method")
    with c6:
        boundary_note = st.text_input("Boundary note (optional)", key="s3_boundary")

    if st.button("Calculate Scope 3", use_container_width=True, key="s3_calc"):
        compute_and_render(
            "Scope 3", category, unit, period_start, period_end,
            baseline_activity, project_activity, ef_kg,
            {
                "input_mode": mode,
                "guided_method": guided_method,
                "activity_method_meta": activity_method,
                "boundary_note": boundary_note,
                "ef_metadata": ef_meta,
                "uncertainty": u_meta,
            }
        )

    if RESULT_KEYS["scope3"] in st.session_state:
        last = st.session_state[RESULT_KEYS["scope3"]]
        o = last["outputs"]
        calc_name_default = f"Scope 3 — {o.get('category')} — {last.get('period_start','') or 'period'}"
        render_save_panel(
            calc_name_default=calc_name_default,
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
    "Rigor note: This tool requires user-supplied emission factors. "
    "When saving to the ledger, EF source/reference is required. "
    "Uncertainty outputs are screening-level and do not represent verification."
)

