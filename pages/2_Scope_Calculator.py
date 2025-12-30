# pages/2_Scope_Calculator.py
# ------------------------------------------------------------
# Carbon Registry • Scope 1/2/3 Calculator (Cloud-safe, single file)
#
# ✅ Free calculations (no project needed)
# ✅ Optional: save results to Carbon Registry ledger (SQLite)
# ✅ Stores full inputs/outputs as JSON for auditability
# ✅ Rigor guardrails:
#    - You must provide an emission factor (EF) to calculate.
#    - To SAVE, you must also provide the EF source/reference text.
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
    # Generic saved calculator outputs (scope + methodologies)
    db_exec(
        """
        CREATE TABLE IF NOT EXISTS calc_runs (
            calc_id TEXT PRIMARY KEY,
            project_id TEXT,
            calc_type TEXT NOT NULL,      -- 'scope' | 'methodology'
            calc_name TEXT NOT NULL,
            scope_label TEXT,             -- e.g., 'Scope 2'
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
    """
    Reads projects from the registry DB.
    If the projects table doesn't exist yet (Registry page not run), returns empty.
    """
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
            period_start = st.text_input("Period start (YYYY-MM-DD) [optional]", value="")
        with c2:
            period_end = st.text_input("Period end (YYYY-MM-DD) [optional]", value="")

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
                period_start=period_start.strip() or None,
                period_end=period_end.strip() or None,
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


# ------------------------------------------------------------
# Scope category scaffolds (no “official” defaults baked in)
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
# Scope 1
# ------------------------------------------------------------
def calc_scope1() -> None:
    st.header("Scope 1 – Direct emissions (fuels, gases, refrigerants)")

    category = st.selectbox("Category", list(SCOPE1_CATEGORIES.keys()))
    unit = SCOPE1_CATEGORIES[category]["unit"]

    c1, c2 = st.columns(2)
    with c1:
        baseline = st.number_input(f"Baseline activity ({unit} per period)", min_value=0.0, value=0.0)
    with c2:
        project = st.number_input(f"Project activity ({unit} per period)", min_value=0.0, value=0.0)

    st.markdown("#### Emission factor")
    ef_kg = st.number_input(f"EF (kg CO₂e per {unit})", min_value=0.0, value=0.0)
    st.caption("Use an authoritative factor source and cite it when saving to the ledger.")

    calc_btn = st.button("Calculate Scope 1", use_container_width=True)

    if calc_btn:
        if not require_positive_baseline(baseline) or not require_positive_ef(ef_kg):
            return

        res = compute_baseline_project_reduction(baseline, project, ef_kg)
        st.success("Calculated.")
        metric_row(res["baseline_tco2e"], res["project_tco2e"], res["reduction_tco2e"], res["reduction_pct"])

        inputs = {
            "scope": "Scope 1",
            "category": category,
            "unit": unit,
            "baseline_activity": baseline,
            "project_activity": project,
            "ef_kgco2e_per_unit": ef_kg,
        }
        outputs = {**res, "scope": "Scope 1", "category": category, "unit": unit}

        st.session_state[RESULT_KEYS["scope1"]] = {"inputs": inputs, "outputs": outputs}

    if RESULT_KEYS["scope1"] in st.session_state:
        last = st.session_state[RESULT_KEYS["scope1"]]
        o = last["outputs"]
        render_save_panel(
            calc_name=f"Scope 1 — {o.get('category')}",
            scope_label="Scope 1",
            baseline_tco2e=o.get("baseline_tco2e"),
            project_tco2e=o.get("project_tco2e"),
            reduction_tco2e=o.get("reduction_tco2e"),
            inputs=last["inputs"],
            outputs=o,
        )


# ------------------------------------------------------------
# Scope 2
# ------------------------------------------------------------
def calc_scope2() -> None:
    st.header("Scope 2 – Purchased energy (electricity, steam, heat, cooling)")

    category = st.selectbox("Category", list(SCOPE2_CATEGORIES.keys()))
    unit = SCOPE2_CATEGORIES[category]["unit"]

    c1, c2 = st.columns(2)
    with c1:
        baseline = st.number_input(f"Baseline activity ({unit} per period)", min_value=0.0, value=0.0)
    with c2:
        project = st.number_input(f"Project activity ({unit} per period)", min_value=0.0, value=0.0)

    st.markdown("#### Factor basis (for reporting rigor)")
    factor_basis = st.radio("Factor basis", ["Location-based", "Market-based", "Other/Custom"], horizontal=True)
    ef_kg = st.number_input(f"EF (kg CO₂e per {unit})", min_value=0.0, value=0.0)

    c3, c4 = st.columns(2)
    with c3:
        supplier = st.text_input("Electricity supplier / utility (optional)")
    with c4:
        grid_region = st.text_input("Grid region (optional)")

    calc_btn = st.button("Calculate Scope 2", use_container_width=True)

    if calc_btn:
        if not require_positive_baseline(baseline) or not require_positive_ef(ef_kg):
            return

        res = compute_baseline_project_reduction(baseline, project, ef_kg)
        st.success("Calculated.")
        metric_row(res["baseline_tco2e"], res["project_tco2e"], res["reduction_tco2e"], res["reduction_pct"])

        inputs = {
            "scope": "Scope 2",
            "category": category,
            "unit": unit,
            "baseline_activity": baseline,
            "project_activity": project,
            "ef_kgco2e_per_unit": ef_kg,
            "factor_basis": factor_basis,
            "supplier": supplier,
            "grid_region": grid_region,
        }
        outputs = {
            **res,
            "scope": "Scope 2",
            "category": category,
            "unit": unit,
            "factor_basis": factor_basis,
            "supplier": supplier,
            "grid_region": grid_region,
        }

        st.session_state[RESULT_KEYS["scope2"]] = {"inputs": inputs, "outputs": outputs}

    if RESULT_KEYS["scope2"] in st.session_state:
        last = st.session_state[RESULT_KEYS["scope2"]]
        o = last["outputs"]
        render_save_panel(
            calc_name=f"Scope 2 — {o.get('category')} ({o.get('factor_basis')})",
            scope_label="Scope 2",
            baseline_tco2e=o.get("baseline_tco2e"),
            project_tco2e=o.get("project_tco2e"),
            reduction_tco2e=o.get("reduction_tco2e"),
            inputs=last["inputs"],
            outputs=o,
        )


# ------------------------------------------------------------
# Scope 3
# ------------------------------------------------------------
def calc_scope3() -> None:
    st.header("Scope 3 – Value chain emissions (GHG Protocol categories)")

    category = st.selectbox("Scope 3 category", list(SCOPE3_CATEGORIES.keys()))
    default_unit = SCOPE3_CATEGORIES[category]["unit"]

    unit = st.text_input(
        "Unit (be specific)",
        value=default_unit,
        placeholder="e.g., passenger-km, ton-km, kg, km, nights, spend-ZAR, units",
    )

    c1, c2 = st.columns(2)
    with c1:
        baseline = st.number_input(f"Baseline activity ({unit} per period)", min_value=0.0, value=0.0)
    with c2:
        project = st.number_input(f"Project activity ({unit} per period)", min_value=0.0, value=0.0)

    st.markdown("#### Emission factor")
    ef_kg = st.number_input(f"EF (kg CO₂e per {unit})", min_value=0.0, value=0.0)

    st.caption("Optional metadata for traceability (saved with the run):")
    c3, c4 = st.columns(2)
    with c3:
        activity_method = st.selectbox(
            "Activity data method (optional)",
            ["", "Spend-based", "Distance-based", "Mass-based", "Supplier-specific", "Hybrid"],
        )
    with c4:
        boundary_note = st.text_input("Boundary note (optional)", placeholder="e.g., includes upstream logistics only")

    calc_btn = st.button("Calculate Scope 3", use_container_width=True)

    if calc_btn:
        if not require_positive_baseline(baseline) or not require_positive_ef(ef_kg):
            return

        res = compute_baseline_project_reduction(baseline, project, ef_kg)
        st.success("Calculated.")
        metric_row(res["baseline_tco2e"], res["project_tco2e"], res["reduction_tco2e"], res["reduction_pct"])

        inputs = {
            "scope": "Scope 3",
            "category": category,
            "unit": unit,
            "baseline_activity": baseline,
            "project_activity": project,
            "ef_kgco2e_per_unit": ef_kg,
            "activity_method": activity_method,
            "boundary_note": boundary_note,
        }
        outputs = {
            **res,
            "scope": "Scope 3",
            "category": category,
            "unit": unit,
            "activity_method": activity_method,
            "boundary_note": boundary_note,
        }

        st.session_state[RESULT_KEYS["scope3"]] = {"inputs": inputs, "outputs": outputs}

    if RESULT_KEYS["scope3"] in st.session_state:
        last = st.session_state[RESULT_KEYS["scope3"]]
        o = last["outputs"]
        render_save_panel(
            calc_name=f"Scope 3 — {o.get('category')}",
            scope_label="Scope 3",
            baseline_tco2e=o.get("baseline_tco2e"),
            project_tco2e=o.get("project_tco2e"),
            reduction_tco2e=o.get("reduction_tco2e"),
            inputs=last["inputs"],
            outputs=o,
        )


# ------------------------------------------------------------
# MAIN NAV (Radio)
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
    "When saving to the ledger, provide the factor source/reference for auditability."
)

