# scopes/scope2.py
import streamlit as st
from datetime import date

from registry.database import SessionLocal
from registry.crud import list_projects, add_emission

from .scope_utils import (
    SCOPE2_EF,
    parse_ef,
    compute_emissions,
    show_results_table,
)


def calc_scope2():

    st.header("Scope 2 – Purchased Electricity")

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

    source = st.selectbox("Energy Source:", list(SCOPE2_EF.keys()))

    col1, col2 = st.columns(2)
    with col1:
        baseline_kwh = st.number_input("Baseline (kWh/year):", min_value=0.0)
    with col2:
        project_kwh = st.number_input("Project (kWh/year):", min_value=0.0)

    ef_choice = st.selectbox(
        "EF (kg CO₂e per kWh):",
        [f"Default – {SCOPE2_EF[source]}"] + ["Custom"]
    )

    custom_ef = None
    if ef_choice == "Custom":
        custom_ef = st.number_input("Custom EF:", min_value=0.0)

    if st.button("Calculate Scope 2 Emissions"):
        if baseline_kwh <= 0:
            st.error("Baseline must be >0.")
            return

        ef = parse_ef(source, SCOPE2_EF, custom_ef)
        if ef is None or ef <= 0:
            st.error("Invalid EF.")
            return

        baseline_t, project_t, reduction_t, reduction_pct = compute_emissions(
            baseline_kwh, project_kwh, ef
        )

        show_results_table(baseline_t, project_t, reduction_t, reduction_pct)

        st.session_state["scope2_results"] = {
            "category": source,
            "baseline_t": baseline_t,
            "project_t": project_t,
            "reduction_t": reduction_t,
        }

    if save_enabled and "scope2_results" in st.session_state:
        results = st.session_state["scope2_results"]

        st.markdown("### 💾 Save to Registry")

        opt = st.selectbox(
            "Save which value?",
            [
                f"Project Emissions: {results['project_t']:.3f} tCO₂e",
                f"Emission Reduction: {results['reduction_t']:.3f} tCO₂e",
            ]
        )

        qty = float(opt.split(":")[1].replace("tCO₂e",""))

        if st.button("Save Emission"):
            with SessionLocal() as session:
                add_emission(
                    session,
                    project_id=project_id,
                    date=date.today(),
                    quantity_tCO2e=qty,
                    activity_type=f"Scope 2 – {results['category']}",
                    notes="Saved via Scope 2 Calculator",
                )
            st.success("Saved to Carbon Registry.")
