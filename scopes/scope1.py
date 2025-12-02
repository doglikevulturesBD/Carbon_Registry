# scopes/scope1.py
import streamlit as st
from datetime import date

from registry.database import SessionLocal
from registry.crud import list_projects, add_emission

from .scope_utils import (
    SCOPE1_EF,
    parse_ef,
    compute_emissions,
    show_results_table,
)


def calc_scope1():

    st.header("Scope 1 – Direct Emissions (Fuel, Gas, Refrigerants)")

    # ---- Project Selection ----
    with SessionLocal() as session:
        projects = list_projects(session)

    if not projects:
        st.warning("Create a project in the Carbon Registry before saving emissions.")
        save_enabled = False
        project_id = None
    else:
        project_map = {f"{p.id}: {p.name}": p.id for p in projects}
        selected_label = st.selectbox("Save to project:", list(project_map.keys()))
        project_id = project_map[selected_label]
        save_enabled = True

    # ---- Category ----
    category = st.selectbox("Emission Source Category:", list(SCOPE1_EF.keys()))
    unit_hint = category.split("(")[-1].replace(")", "")

    # ---- Inputs ----
    col1, col2 = st.columns(2)
    with col1:
        baseline = st.number_input(f"Baseline activity ({unit_hint}/year):", min_value=0.0)
    with col2:
        project = st.number_input(f"Project activity ({unit_hint}/year):", min_value=0.0)

    # ---- Emission factor ----
    ef_choice = st.selectbox(
        "Emission factor (kg CO₂e per unit):",
        [f"Default – {SCOPE1_EF[category]}"] + ["Custom"]
    )

    custom_ef = None
    if ef_choice == "Custom":
        custom_ef = st.number_input("Custom EF (kg CO₂e per unit):", min_value=0.0)

    # ---- Calculate ----
    if st.button("Calculate Scope 1 Emissions"):
        if baseline <= 0:
            st.error("Baseline must be > 0.")
            return

        ef = parse_ef(category, SCOPE1_EF, custom_ef)
        if ef is None or ef <= 0:
            st.error("Invalid emission factor.")
            return

        baseline_t, project_t, reduction_t, reduction_pct = compute_emissions(
            baseline, project, ef
        )

        show_results_table(baseline_t, project_t, reduction_t, reduction_pct)

        st.session_state["scope1_results"] = {
            "baseline_t": baseline_t,
            "project_t": project_t,
            "reduction_t": reduction_t,
            "category": category
        }

    # ---- Save to Registry ----
    if save_enabled and "scope1_results" in


