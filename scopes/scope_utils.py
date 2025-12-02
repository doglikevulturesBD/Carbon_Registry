# scopes/scope_utils.py
import json
from datetime import datetime
from pathlib import Path
import streamlit as st

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SCOPE_RECORDS_FILE = DATA_DIR / "scope_records.json"

# -------------------------------------------------------------------
# Emission factor dictionaries (kg CO2e per unit)
# (You can later swap these with a proper EF database or JSON.)
# -------------------------------------------------------------------

SCOPE1_EF = {
    "Stationary – Diesel (litres)": 2.68,
    "Stationary – Natural gas (m³)": 2.03,
    "Stationary – LPG (kg)": 3.00,   # approx; update as needed
    "Mobile – Fleet (diesel, litres)": 2.68,
    "Mobile – Fleet (petrol, litres)": 2.31,
    "Fugitive – R410A (kg)": 2088.0,  # GWP x 1 kg
    "Fugitive – R134a (kg)": 1430.0,
}

SCOPE2_EF = {
    "South Africa Grid (kWh)": 0.92,
    "Global Grid Avg (kWh)": 0.48,
    "Solar PV (kWh)": 0.05,
    "Wind (kWh)": 0.011,
    "Hydro (kWh)": 0.024,
}

# Scope 3 activity-based EF (kg CO2e per activity unit)
SCOPE3_ACTIVITY_EF = {
    "Business travel – Car (km)": 0.21,
    "Business travel – Bus (km)": 0.09,
    "Business travel – Rail (km)": 0.041,
    "Business travel – Air (short-haul, km)": 0.28,
    "Business travel – Air (long-haul, km)": 0.18,
    "Employee commuting – Car (km)": 0.21,
    "Employee commuting – Bus (km)": 0.09,
    "Employee commuting – Rail (km)": 0.041,
    "Upstream freight – Truck (tonne-km)": 0.1,
    "Upstream freight – Ship (tonne-km)": 0.015,
    "Downstream distribution – Truck (tonne-km)": 0.1,
    "Waste – Landfilled (tonnes)": 100.0,  # placeholder
    "Waste – Recycled (tonnes)": 20.0,     # placeholder
}

# Scope 3 spend-based EF (kg CO2e per currency unit, e.g. per ZAR)
SCOPE3_SPEND_EF = {
    "Purchased goods & services": 0.3,
    "Capital goods": 0.4,
    "Fuel & energy related activities": 0.25,
    "Upstream transport & distribution": 0.2,
    "Waste generated in operations": 0.15,
    "Business travel (spend)": 0.35,
    "Employee commuting (benefits)": 0.2,
    "Upstream leased assets": 0.3,
    "Downstream transport": 0.2,
    "Use of sold products": 0.5,
    "End-of-life treatment of sold products": 0.45,
    "Investments": 0.6,
}


def parse_ef(label: str, dictionary: dict, custom_value: float | None) -> float | None:
    """
    Return emission factor in kg CO2e per unit.
    If custom_value is provided, it overrides.
    Otherwise gets EF from dictionary by label.
    """
    if custom_value is not None and custom_value > 0:
        return custom_value
    return dictionary.get(label)


def compute_emissions(baseline_activity: float, project_activity: float, ef: float):
    """
    Compute baseline, project and reduction in tonnes CO2e
    given activity data and emission factor in kg CO2e per unit.
    """
    baseline_kg = baseline_activity * ef
    project_kg = project_activity * ef
    reduction_kg = baseline_kg - project_kg

    if baseline_kg <= 0:
        reduction_pct = 0.0
    else:
        reduction_pct = (reduction_kg / baseline_kg) * 100.0

    return (
        baseline_kg / 1000.0,
        project_kg / 1000.0,
        reduction_kg / 1000.0,
        reduction_pct,
    )


def show_results_table(baseline_t, project_t, reduction_t, reduction_pct):
    """Display a clean results block in Streamlit."""
    st.success("✅ Emission Calculation Complete")

    st.markdown(
        f"""
        | Metric | Value |
        |--------|-------|
        | **Baseline Emissions** | `{baseline_t:.3f}` tCO₂e |
        | **Project Emissions**  | `{project_t:.3f}` tCO₂e |
        | **Reduction**          | `{reduction_t:.3f}` tCO₂e |
        | **Reduction %**        | `{reduction_pct:.1f}` % |
        """,
        unsafe_allow_html=False,
    )


def save_scope_record(
    scope: str,
    category: str,
    method: str,
    baseline_activity: float,
    project_activity: float,
    ef: float,
    baseline_t: float,
    project_t: float,
    reduction_t: float,
):
    """Save a record of the calculation to JSON for registry integration."""
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scope": scope,
        "category": category,
        "method": method,
        "baseline_activity": baseline_activity,
        "project_activity": project_activity,
        "emission_factor_kg_per_unit": ef,
        "baseline_tonnes": baseline_t,
        "project_tonnes": project_t,
        "reduction_tonnes": reduction_t,
    }

    existing = []
    if SCOPE_RECORDS_FILE.exists():
        try:
            existing = json.loads(SCOPE_RECORDS_FILE.read_text())
        except json.JSONDecodeError:
            existing = []

    existing.append(record)
    SCOPE_RECORDS_FILE.write_text(json.dumps(existing, indent=2))
