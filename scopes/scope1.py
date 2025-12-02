# scopes/scope1.py
import streamlit as st
from .scope_utils import (
    SCOPE1_EF,
    parse_ef,
    compute_emissions,
    show_results_table,
    save_scope_record,
)


def calc_scope1():
    st.subheader("Scope 1 – Direct GHG Emissions")

    st.markdown(
        """
        Scope 1 emissions are **direct** greenhouse gas emissions from owned or controlled sources.
        This module covers:
        - Stationary combustion (e.g. boilers, generators)
        - Mobile combustion (vehicle fleet)
        - Fugitive emissions (e.g. refrigerant leaks)
        """,
        unsafe_allow_html=False,
    )

    category = st.selectbox(
        "Emission source category:",
        list(SCOPE1_EF.keys()),
    )

    unit_hint = category.split("(")[-1].rstrip(")")

    col1, col2 = st.columns(2)
    with col1:
        baseline = st.number_input(
            f"Baseline activity ({unit_hint} per year):",
            min_value=0.0,
            step=0.01,
        )
    with col2:
        project = st.number_input(
            f"Project activity ({unit_hint} per year):",
            min_value=0.0,
            step=0.01,
        )

    st.caption(
        "Baseline = current or historical annual value. "
        "Project = annual value under the low-carbon project scenario."
    )

    ef_choice = st.selectbox(
        "Emission factor (kg CO₂e per unit):",
        [f"Default – {SCOPE1_EF[category]}"] + ["Custom"],
    )

    custom_ef = None
    if ef_choice == "Custom":
        custom_ef = st.number_input(
            "Custom emission factor (kg CO₂e per unit):", min_value=0.0, step=0.0001
        )

    if st.button("Calculate Scope 1 emissions"):
        # Validation
        if baseline <= 0:
            st.error("Baseline activity must be greater than zero.")
            return
        if project < 0:
            st.error("Project activity cannot be negative.")
            return
        if project > baseline:
            st.warning(
                "⚠️ Project activity is higher than baseline. "
                "This indicates an increase rather than a reduction."
            )

        ef = parse_ef(category, SCOPE1_EF, custom_ef)

        if ef is None or ef <= 0:
            st.error("Valid emission factor is required.")
            return

        baseline_t, project_t, reduction_t, reduction_pct = compute_emissions(
            baseline, project, ef
        )

        show_results_table(baseline_t, project_t, reduction_t, reduction_pct)

        # Auto-save for registry integration
        method = "activity-based"
        save_scope_record(
            scope="Scope 1",
            category=category,
            method=method,
            baseline_activity=baseline,
            project_activity=project,
            ef=ef,
            baseline_t=baseline_t,
            project_t=project_t,
            reduction_t=reduction_t,
        )

