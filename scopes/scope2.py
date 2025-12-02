# scopes/scope2.py
import streamlit as st
from .scope_utils import (
    SCOPE2_EF,
    parse_ef,
    compute_emissions,
    show_results_table,
    save_scope_record,
)


def calc_scope2():
    st.subheader("Scope 2 – Indirect Emissions from Purchased Electricity, Heat & Steam")

    st.markdown(
        """
        Scope 2 emissions are **indirect** GHG emissions from the generation of purchased energy.
        This module calculates emissions from:
        - Grid electricity
        - Renewable electricity (market-based factors)
        - Other purchased energy carriers
        """,
        unsafe_allow_html=False,
    )

    source = st.selectbox("Energy source:", list(SCOPE2_EF.keys()))

    col1, col2 = st.columns(2)
    with col1:
        baseline_kwh = st.number_input(
            "Baseline consumption (kWh per year):", min_value=0.0, step=0.01
        )
    with col2:
        project_kwh = st.number_input(
            "Project consumption (kWh per year):", min_value=0.0, step=0.01
        )

    st.caption(
        "Tip: You can model a fuel switch or on-site renewables by changing the energy mix "
        "or using different emission factors for baseline vs project in the methodologies module."
    )

    ef_choice = st.selectbox(
        "Emission factor (kg CO₂e per kWh):",
        [f"Default – {SCOPE2_EF[source]}"] + ["Custom"],
    )

    custom_ef = None
    if ef_choice == "Custom":
        custom_ef = st.number_input(
            "Custom emission factor (kg CO₂e per kWh):",
            min_value=0.0,
            step=0.0001,
        )

    if st.button("Calculate Scope 2 emissions"):
        if baseline_kwh <= 0:
            st.error("Baseline consumption must be greater than zero.")
            return
        if project_kwh < 0:
            st.error("Project consumption cannot be negative.")
            return
        if project_kwh > baseline_kwh:
            st.warning(
                "⚠️ Project consumption is higher than baseline. "
                "This indicates an increase rather than a reduction."
            )

        ef = parse_ef(source, SCOPE2_EF, custom_ef)

        if ef is None or ef <= 0:
            st.error("Valid emission factor is required.")
            return

        baseline_t, project_t, reduction_t, reduction_pct = compute_emissions(
            baseline_kwh, project_kwh, ef
        )

        show_results_table(baseline_t, project_t, reduction_t, reduction_pct)

        save_scope_record(
            scope="Scope 2",
            category=source,
            method="activity-based",
            baseline_activity=baseline_kwh,
            project_activity=project_kwh,
            ef=ef,
            baseline_t=baseline_t,
            project_t=project_t,
            reduction_t=reduction_t,
        )

