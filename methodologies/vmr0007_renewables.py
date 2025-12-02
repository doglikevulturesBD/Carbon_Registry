import streamlit as st
import pandas as pd
import altair as alt
from datetime import date

from registry.database import SessionLocal
from registry.crud import list_projects, add_emission


# ----------------------------------------------------
# DEFAULT EMISSION FACTORS (EXAMPLE ranges)
# Users can override with custom values
# ----------------------------------------------------

BASELINE_EF = {
    "Plastic": {"Landfill": 1.3, "Incineration": 2.6},
    "Paper":   {"Landfill": 1.0, "Incineration": 1.8},
    "Metal":   {"Landfill": 1.8, "Incineration": 2.0},
    "Glass":   {"Landfill": 0.5, "Incineration": 0.7},
    "Other":   {"Landfill": 0.8, "Incineration": 1.1},
}

VIRGIN_AVOIDED_EF = {
    "Plastic": 1.1,
    "Paper": 0.6,
    "Metal": 2.5,
    "Glass": 0.3,
    "Other": 0.5,
}

# Recycling electricity EF in kg CO2e/kWh (user can override)
DEFAULT_GRID_EF = 0.9



# ----------------------------------------------------
# STREAMLIT MODULE
# ----------------------------------------------------

def vmr0007_solid_waste():

    st.title("♻️ VMR0007 – Solid Waste Recovery & Recycling Calculator")
    st.caption("Full multi-material, multi-year Verra-aligned recycling methodology")

    # ----------------------------------------------------
    # METHODOLOGY OVERVIEW
    # ----------------------------------------------------
    with st.expander("📘 Methodology Overview (VMR0007)", expanded=False):
        st.markdown(
            """
            **VMR0007 quantifies emission reductions from solid waste recovery & recycling:**

            ### 1. Baseline emissions (BE)
            Emissions that would have occurred if material was **landfilled** or **incinerated**.

            ### 2. Avoided virgin material emissions (AE)
            Emissions avoided by producing **recycled material** instead of virgin material.

            ### 3. Project emissions (PE)
            Emissions from:
            - Collection & transport  
            - Sorting & conditioning  
            - Recycling facility energy use  
            - Residual waste disposal  

            ### 4. Emission reduction (ER)

            \[
            ER = BE + AE - PE
            \]

            With:
            - Optional uncertainty analysis  
            - Multi-year calculation  
            - Material-specific emission factors  
            - Grid decarbonisation  
            - Leakage/residual waste factors  
            """
        )

    # ----------------------------------------------------
    # SELECT PROJECT FROM REGISTRY
    # ----------------------------------------------------
    with SessionLocal() as session:
        projects = list_projects(session)

    if not projects:
        st.error("Create a project before using VMR0007.")
        return

    mapping = {f"{p.id}: {p.name}": p.id for p in projects}
    chosen = st.selectbox("Save results to project:", list(mapping.keys()))
    project_id = mapping[chosen]

    st.divider()

    # ----------------------------------------------------
    # MULTI-MATERIAL INPUT TABLE
    # ----------------------------------------------------
    st.subheader("1. Material Recovery Inputs")

    st.markdown("Enter recovery amounts for each material (tons/year):")

    default_df = pd.DataFrame({
        "Material": ["Plastic", "Paper", "Metal", "Glass", "Other"],
        "Tons/year": [0, 0, 0, 0, 0],
        "Contamination %": [5, 5, 5, 5, 5],     # impacts net recycled mass
    })

    data = st.data_editor(default_df, num_rows="dynamic")

    st.divider()

    # ----------------------------------------------------
    # BASELINE SCENARIO + EFs
    # ----------------------------------------------------
    st.subheader("2. Baseline Scenario & Emission Factors")

    baseline_option = st.radio("Baseline scenario:", ["Landfill", "Incineration"], horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        custom_base = st.checkbox("Use custom baseline EF values", value=False)
    with col2:
        custom_virgin = st.checkbox("Use custom avoided virgin material EF", value=False)

    # Editable EF tables
    if custom_base:
        st.markdown("### Baseline emission factors (tCO₂e/ton)")
        baseline_ef_df = st.data_editor(
            pd.DataFrame({
                "Material": list(BASELINE_EF.keys()),
                "Baseline EF (tCO₂e/t)": [BASELINE_EF[m][baseline_option] for m in BASELINE_EF]
            }),
            key="baseline_ef_editor"
        )
    if custom_virgin:
        st.markdown("### Avoided virgin production EF (tCO₂e/ton)")
        virgin_df = st.data_editor(
            pd.DataFrame({
                "Material": list(VIRGIN_AVOIDED_EF.keys()),
                "AE EF (tCO₂e/t)": [VIRGIN_AVOIDED_EF[m] for m in VIRGIN_AVOIDED_EF]
            }),
            key="virgin_ef_editor"
        )

    st.divider()

    # ----------------------------------------------------
    # PROJECT EMISSIONS
    # ----------------------------------------------------
    st.subheader("3. Project Emissions (PE)")

    col_pe1, col_pe2 = st.columns(2)
    with col_pe1:
        transport_em = st.number_input("Collection & transport emissions (tCO₂e/year):", min_value=0.0, value=0.0)
        residual_fraction = st.slider("Residual waste fraction (%):", 0, 30, 10)
    with col_pe2:
        facility_kwh = st.number_input("Recycling facility electricity use (kWh/year):", min_value=0.0)
        grid_ef = st.number_input("Grid EF (kg CO₂e/kWh):", min_value=0.0, value=DEFAULT_GRID_EF)

    facility_em_t = facility_kwh * grid_ef / 1000.0

    st.info(f"Facility emissions: **{facility_em_t:.3f} tCO₂e/year**")

    st.divider()

    # ----------------------------------------------------
    # MULTI-YEAR SETTINGS
    # ----------------------------------------------------
    st.subheader("4. Project Lifetime & Grid Decarbonisation")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        years = st.number_input("Project duration (years):", min_value=1, value=10)
    with col_t2:
        grid_reduction = st.slider("Annual grid EF reduction (%):", 0.0, 10.0, 2.0)

    # ----------------------------------------------------
    # UNCERTAINTY
    # ----------------------------------------------------
    unc_base = st.slider("Baseline uncertainty (%):", 0.0, 20.0, 5.0)
    unc_ae = st.slider("Virgin material avoided uncertainty (%):", 0.0, 20.0, 5.0)
    unc_pe = st.slider("Project emissions uncertainty (%):", 0.0, 20.0, 5.0)

    st.divider()

    # ----------------------------------------------------
    # CALCULATE RESULTS
    # ----------------------------------------------------
    if st.button("Calculate VMR0007 Emission Reductions", type="primary"):

        rows = []

        # Loop per material
        for _, row in data.iterrows():
            mat = row["Material"]
            tons = row["Tons/year"]
            cont = row["Contamination %"] / 100.0

            clean_tons = tons * (1 - cont)

            # Baseline EF
            if custom_base:
                baseline_val = float(
                    baseline_ef_df.loc[baseline_ef_df["Material"] == mat, "Baseline EF (tCO₂e/t)"].values[0]
                )
            else:
                baseline_val = BASELINE_EF[mat][baseline_option]

            # Avoided virgin EF
            if custom_virgin:
                virgin_val = float(
                    virgin_df.loc[virgin_df["Material"] == mat, "AE EF (tCO₂e/t)"].values[0]
                )
            else:
                virgin_val = VIRGIN_AVOIDED_EF[mat]

            BE = clean_tons * baseline_val
            AE = clean_tons * virgin_val

            rows.append([mat, tons, clean_tons, BE, AE])

        df_materials = pd.DataFrame(
            rows, columns=["Material", "Tons/year", "Clean tons", "BE (t)", "AE (t)"]
        )

        # Residual waste emissions
        residual_em_t = df_materials["Clean tons"].sum() * (residual_fraction / 100) * 0.8  # assumed landfill EF

        # Project emissions total
        PE_total = transport_em + facility_em_t + residual_em_t

        # Baseline + AE
        BE_total = df_materials["BE (t)"].sum()
        AE_total = df_materials["AE (t)"].sum()

        # Year 1
        ER_y1 = BE_total + AE_total - PE_total

        # Multi-year
        results = []
        cur_grid_ef = grid_ef

        for y in range(1, int(years) + 1):
            # Adjust facility emissions each year by decarbonising grid
            facility_yr = facility_kwh * (cur_grid_ef / 1000)
            PE_yr = transport_em + facility_yr + residual_em_t

            ER = BE_total + AE_total - PE_yr

            results.append([y, BE_total, AE_total, PE_yr, ER])

            cur_grid_ef *= (1 - grid_reduction / 100)

        df_years = pd.DataFrame(
            results, columns=["Year", "BE (t)", "AE (t)", "PE (t)", "ER (t)"]
        )

        total_ER = df_years["ER (t)"].sum()

        # Uncertainty propagation (simplified RSS)
        unc_total_pct = ((unc_base/100)**2 + (unc_ae/100)**2 + (unc_pe/100)**2) ** 0.5
        total_ER_unc = total_ER * unc_total_pct

        st.success(
            f"**Total Emission Reductions: {total_ER:.3f} ± {total_ER_unc:.3f} tCO₂e** "
            f"over {years} years"
        )

        st.subheader("Annual Emission Results")
        st.dataframe(df_years, use_container_width=True)

        # Chart
        chart = (
            alt.Chart(df_years)
            .mark_line(point=True)
            .encode(
                x="Year:O",
                y="ER (t):Q",
                tooltip=["Year", "ER (t)"],
            )
        )
        st.altair_chart(chart, use_container_width=True)

        # Download CSV
        st.download_button(
            "💾 Download Annual Results (CSV)",
            df_years.to_csv(index=False).encode("utf-8"),
            "vmr0007_results.csv",
            "text/csv",
        )

        st.divider()

        # Save to registry
        st.subheader("Save to Carbon Registry")

        if st.button("Save total reduction to Registry"):
            with SessionLocal() as session:
                add_emission(
                    session,
                    project_id=project_id,
                    date=date.today(),
                    quantity_tCO2e=total_ER,
                    activity_type="VMR0007 – Solid Waste Recycling",
                    notes=f"Multi-material recycling over {years} years using VMR0007 method.",
                )
            st.success("Saved into the Carbon Registry.")



