# scopes/scope3.py
import streamlit as st
from datetime import date

from registry.database import SessionLocal
from registry.crud import list_projects, add_emission

from .scope_utils import (
    SCOPE3_ACTIVITY_EF,
    SCOPE3_SPEND_EF,
    parse_ef,
    compute_emissions,
    show_results_table,
)


def calc_scope3():

    st.header("Scope 3 – Value Chain Emissions")

    with SessionLocal() as session:
        projects = list_projects(session)

    if not projects:
        st.warning("Create a project before saving emissions.")
        save_enabled = False
        project_id = None
    else:
        project_map = {f"{p.id}: {p.name}": p.id for p in projects}
        selected_label = st.selectbox("Save to project:", list(project_map.keys()))
        project_id = project_map[selected_label]
        save_enabled = True

    method = st.radio("Method:", ["Activity-based", "Spend-based"])

    if method == "Activity-based":
        _scope3_activity(project_id, save_enabled)
    else:
        _scope3_spend(project_id, save_enabled)


def _scope3_activity(project_id, save_enabled):

    category = st.selectbox("Activity Category:", list(SCOPE3_ACTIVITY_EF.keys()))
    unit = category.split("(")[-1].replace(")", "")

    col1, col2 = st.columns(2)
    with col1:
        baseline = st.number_input(f"Baseline ({unit}/yr):", min_value=0.0)
    with col2:
        project = st.number_input(f"Project ({unit}/yr):", min_value=0.0)

    ef_choice = st.selectbox(
        "EF (kg CO₂e/unit):",
        [f"Default – {SCOPE3_ACTIVITY_EF[category]}"] + ["Custom"]
    )

    custom_ef = None
    if ef_choice == "Custom":
        custom_ef = st.number_input("Custom EF:", min_value=0.0)

    if st.button("Calculate Scope 3 (activity)"):
        if baseline <= 0:
            st.error("Baseline must be >0")
            return

        ef = parse_ef(category, SCOPE3_ACTIVITY_EF, custom_ef)
        if ef is None or ef <= 0:
            st.error("Invalid EF.")
            return

        baseline_t, project_t, reduction_t, reduction_pct = compute_emissions(
            baseline, project, ef
        )

        show_results_table(baseline_t, project_t, reduction_t, reduction_pct)

        st.session_state["scope3_results"] = {
            "category": category,
            "baseline_t": baseline_t,
            "project_t": project_t,
            "reduction_t": reduction_t,
            "method": "activity",
        }

    # Save
    if save_enabled and "scope3_results" in st.session_state:
        results = st.session_state["scope3_results"]

        opt = st.selectbox(
            "Save which value?",
            [
                f"Project Emissions: {results['project_t']:.3f} tCO₂e",
                f"Emission Reduction: {results['reduction_t']:.3f} tCO₂e",
            ]
        )
        qty = float(opt.split(":")[1].replace("tCO₂e",""))

        if st.button("Save to Registry"):
            with SessionLocal() as session:
                add_emission(
                    session,
                    project_id=project_id,
                    date=date.today(),
                    quantity_tCO2e=qty,
                    activity_type=f"Scope 3 – {results['category']}",
                    notes="Saved via Scope 3 Calculator (Activity)",
                )
            st.success("Saved!")


def _scope3_spend(project_id, save_enabled):

    category = st.selectbox("Spend Category:", list(SCOPE3_SPEND_EF.keys()))

    col1, col2 = st.columns(2)
    with col1:
        baseline = st.number_input("Baseline Spend (ZAR/yr):", min_value=0.0)
    with col2:
        project = st.number_input("Project Spend (ZAR/yr):", min_value=0.0)

    ef_choice = st.selectbox(
        "EF (kg CO₂e / ZAR):",
        [f"Default – {SCOPE3_SPEND_EF[category]}"] + ["Custom"]
    )

    custom_ef = None
    if ef_choice == "Custom":
        custom_ef = st.number_input("Custom EF:", min_value=0.0)

    if st.button("Calculate Scope 3 (spend)"):
        if baseline <= 0:
            st.error("Baseline must be >0")
            return

        ef = parse_ef(category, SCOPE3_SPEND_EF, custom_ef)
        if ef is None:
            st.error("Invalid EF.")
            return

        baseline_t, project_t, reduction_t, reduction_pct = compute_emissions(
            baseline, project, ef
        )

        show_results_table(baseline_t, project_t, reduction_t, reduction_pct)

        st.session_state["scope3_results"] = {
            "category": category,
            "baseline_t": baseline_t,
            "project_t": project_t,
            "reduction_t": reduction_t,
            "method": "spend",
        }

    if save_enabled and "scope3_results" in st.session_state:
        results = st.session_state["scope3_results"]

        opt = st.selectbox(
            "Save which value?",
            [
                f"Project Emissions: {results['project_t']:.3f} tCO₂e",
                f"Emission Reduction: {results['reduction_t']:.3f} tCO₂e",
            ]
        )
        qty = float(opt.split(":")[1].replace("tCO₂e",""))

        if st.button("Save Scope 3 Emission"):
            with SessionLocal() as session:
                add_emission(
                    session,
                    project_id=project_id,
                    date=date.today(),
                    quantity_tCO2e=qty,
                    activity_type=f"Scope 3 – {results['category']}",
                    notes="Saved via Scope 3 Calculator (Spend)",
                )
            st.success("Saved to Carbon Registry!")
