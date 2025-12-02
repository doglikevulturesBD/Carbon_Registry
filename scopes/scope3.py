# scopes/scope3.py
import streamlit as st
from .scope_utils import (
    SCOPE3_ACTIVITY_EF,
    SCOPE3_SPEND_EF,
    parse_ef,
    compute_emissions,
    show_results_table,
    save_scope_record,
)


def calc_scope3():
    st.subheader("Scope 3 – Other Indirect Emissions (Value Chain)")

    st.markdown(
        """
        Scope 3 emissions include all other indirect emissions in the **value chain** of the reporting organisation.
        This module supports:
        - **Activity-based** calculations (e.g. km, tonne-km, tonnes)
        - **Spend-based** calculations (e.g. Rands spent × kg CO₂e/R)
        
        Categories are aligned with the **GHG Protocol Scope 3 Standard**.
        """,
        unsafe_allow_html=False,
    )

    method = st.radio(
        "Choose calculation method:",
        ["Activity-based (km, tonne-km, tonnes, etc.)", "Spend-based (currency)"],
    )

    if method.startswith("Activity"):
        _calc_scope3_activity()
    else:
        _calc_scope3_spend()


def _calc_scope3_activity():
    st.markdown("### Activity-based Scope 3")

    category = st.selectbox(
        "Activity category:",
        list(SCOPE3_ACTIVITY_EF.keys()),
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

    ef_choice = st.selectbox(
        "Emission factor (kg CO₂e per unit):",
        [f"Default – {SCOPE3_ACTIVITY_EF[category]}"] + ["Custom"],
    )

    custom_ef = None
    if ef_choice == "Custom":
        custom_ef = st.number_input(
            "Custom emission factor (kg CO₂e per unit):",
            min_value=0.0,
            step=0.0001,
        )

    if st.button("Calculate Scope 3 (activity-based)"):
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

        ef = parse_ef(category, SCOPE3_ACTIVITY_EF, custom_ef)
        if ef is None or ef <= 0:
            st.error("Valid emission factor is required.")
            return

        baseline_t, project_t, reduction_t, reduction_pct = compute_emissions(
            baseline, project, ef
        )

        show_results_table(baseline_t, project_t, reduction_t, reduction_pct)

        save_scope_record(
            scope="Scope 3",
            category=category,
            method="activity-based",
            baseline_activity=baseline,
            project_activity=project,
            ef=ef,
            baseline_t=baseline_t,
            project_t=project_t,
            reduction_t=reduction_t,
        )


def _calc_scope3_spend():
    st.markdown("### Spend-based Scope 3")

    category = st.selectbox(
        "Spend category (GHG Protocol):",
        list(SCOPE3_SPEND_EF.keys()),
    )

    col1, col2 = st.columns(2)
    with col1:
        baseline_spend = st.number_input(
            "Baseline spend (e.g. ZAR per year):",
            min_value=0.0,
            step=0.01,
        )
    with col2:
        project_spend = st.number_input(
            "Project spend (e.g. ZAR per year):",
            min_value=0.0,
            step=0.01,
        )

    ef_choice = st.selectbox(
        "Emission factor (kg CO₂e per currency unit):",
        [f"Default – {SCOPE3_SPEND_EF[category]}"] + ["Custom"],
    )

    custom_ef = None
    if ef_choice == "Custom":
        custom_ef = st.number_input(
            "Custom emission factor (kg CO₂e per currency unit):",
            min_value=0.0,
            step=0.0001,
        )

    if st.button("Calculate Scope 3 (spend-based)"):
        if baseline_spend <= 0:
            st.error("Baseline spend must be greater than zero.")
            return
        if project_spend < 0:
            st.error("Project spend cannot be negative.")
            return
        if project_spend > baseline_spend:
            st.warning(
                "⚠️ Project spend is higher than baseline. "
                "This may still be valid (e.g. greener but more expensive services) – "
                "interpret reductions carefully."
            )

        ef = parse_ef(category, SCOPE3_SPEND_EF, custom_ef)
        if ef is None or ef <= 0:
            st.error("Valid emission factor is required.")
            return

        baseline_t, project_t, reduction_t, reduction_pct = compute_emissions(
            baseline_spend, project_spend, ef
        )

        show_results_table(baseline_t, project_t, reduction_t, reduction_pct)

        save_scope_record(
            scope="Scope 3",
            category=category,
            method="spend-based",
            baseline_activity=baseline_spend,
            project_activity=project_spend,
            ef=ef,
            baseline_t=baseline_t,
            project_t=project_t,
            reduction_t=reduction_t,
        )

