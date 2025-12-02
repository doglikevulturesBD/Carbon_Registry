import streamlit as st
from datetime import date
import pandas as pd
import altair as alt

from registry.database import SessionLocal
from registry.crud import list_projects, add_emission


# ---------------------------
# Default emission factors (kg CO₂e / litre)
# ---------------------------
FUEL_EF = {
    "Petrol": 2.31,
    "Diesel": 2.68,
    "LPG": 1.51,
    "Other": None,
}

# Well-to-tank (upstream) emissions (kg CO₂e / litre)
WTT_EF = {
    "Petrol": 0.52,
    "Diesel": 0.58,
    "LPG": 0.21,
}

# Approximate lower heating value (MJ / litre)
FUEL_ENERGY_MJ = {
    "Petrol": 34.2,
    "Diesel": 38.6,
    "LPG": 26.8,
    "Other": 0.0,
}

# Renewable electricity EF (assumed 0 for CO₂e)
RENEWABLE_EF = 0.0


def vm0038_ev():
    st.title("⚡ VM0038 – EV Charging System Methodology")
    st.caption("Baseline: Internal combustion vehicles | Project: Electric vehicles with grid/renewable electricity")

    # -------------------------------------------------
    # 0. Methodology summary (collapsible)
    # -------------------------------------------------
    with st.expander("📘 Methodology overview (VM0038-style)", expanded=False):
        st.markdown(
            """
            This tool implements a **VM0038-style EV charging methodology**:

            - **Baseline emissions (BEy)**: fuel consumption of internal combustion vehicles  
              \\( BE_y = FC_{fuel,y} \\times EF_{fuel} \\)  
              with optional *well-to-tank (WTT)* upstream emissions.

            - **Project emissions (PEy)**: electricity used to charge EVs  
              \\( PE_y = E_{EV,y} \\times EF_{grid,y} \\),  
              adjusted for:
                - charging efficiency losses
                - renewable fraction in electricity supply
                - grid decarbonisation over time.

            - **Net emission reduction per year**:  
              \\( ER_y = BE_y - PE_y \\)

            - **Total emission reductions**:  
              \\( ER_{total} = \\sum_y ER_y \\)

            This implementation also adds:
            - Multiple charger aggregation from charging sessions
            - Fuel energy equivalence (MJ/year)
            - Uncertainty ranges for BEy, PEy and ER_y
            - Annual table, charts, CSV download
            - Direct saving of total reductions into the **Carbon Registry**.
            """
        )

    # -------------------------------------------------
    # 1. Project selection for saving
    # -------------------------------------------------
    with SessionLocal() as session:
        projects = list_projects(session)

    if not projects:
        st.error("Create a project in the Carbon Registry before using this methodology.")
        return

    project_map = {f"{p.id}: {p.name}": p.id for p in projects}
    selected_project_label = st.selectbox("Save VM0038 results to project:", list(project_map.keys()))
    project_id = project_map[selected_project_label]

    st.divider()

    # -------------------------------------------------
    # 2. Baseline – fuel avoided
    # -------------------------------------------------
    st.subheader("1. Baseline – Fuel Consumption Avoided (Internal Combustion Vehicles)")

    col1, col2 = st.columns(2)
    with col1:
        fuel_type = st.selectbox("Baseline fuel type:", list(FUEL_EF.keys()))
        fuel_use_l = st.number_input("Fuel consumption avoided (litres/year):", min_value=0.0, step=0.01)
    with col2:
        default_fuel_ef = FUEL_EF[fuel_type] if FUEL_EF[fuel_type] is not None else 0.0
        ef_fuel = st.number_input("Fuel EF (kg CO₂e/litre):", value=float(default_fuel_ef), min_value=0.0, step=0.0001)
        include_wtt = st.checkbox("Include Well-to-Tank (upstream) emissions", value=True)

    wtt_ef = WTT_EF.get(fuel_type, 0.0) if include_wtt else 0.0

    # Fuel energy equivalence
    energy_mj_year = fuel_use_l * FUEL_ENERGY_MJ.get(fuel_type, 0.0)

    col_energy1, col_energy2 = st.columns(2)
    with col_energy1:
        st.info(f"Fuel energy equivalent: **{energy_mj_year:,.1f} MJ/year**")
    with col_energy2:
        st.caption("Approximate lower heating value used; for diagnostics only, not in CO₂e calculation.")

    # Uncertainty on baseline
    baseline_uncert_pct = st.slider("Baseline emission uncertainty (%):", 0.0, 20.0, 5.0, step=0.5)

    BEy_kg = fuel_use_l * (ef_fuel + wtt_ef)

    st.write(f"**Baseline emissions (BEy): {BEy_kg:,.2f} kg CO₂e/year**")

    st.divider()

    # -------------------------------------------------
    # 3. Project emissions – electricity for EV charging
    # -------------------------------------------------
    st.subheader("2. Project – EV Charging Electricity Use")

    mode = st.radio(
        "How do you want to define project electricity use?",
        ["Direct annual kWh", "From charger fleet parameters"],
        horizontal=True,
    )

    if mode == "Direct annual kWh":
        col1, col2 = st.columns(2)
        with col1:
            kwh_year = st.number_input("EV charging electricity (kWh/year):", min_value=0.0, step=0.01)
        with col2:
            charge_eff = st.slider("Charging system efficiency (%):", 70, 100, 90)
    else:
        st.markdown("**Charger fleet parameters**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            n_chargers = st.number_input("Number of chargers:", min_value=1, value=4, step=1)
        with c2:
            sessions_per_day = st.number_input("Sessions per charger per day:", min_value=0.0, value=4.0, step=0.5)
        with c3:
            kwh_per_session = st.number_input("kWh delivered per session:", min_value=0.0, value=20.0, step=0.5)
        with c4:
            operating_days = st.number_input("Operating days per year:", min_value=0, value=300, step=1)

        kwh_year = n_chargers * sessions_per_day * kwh_per_session * operating_days
        st.info(f"Derived annual EV charging electricity: **{kwh_year:,.1f} kWh/year**")
        charge_eff = st.slider("Charging system efficiency (%):", 70, 100, 90)

    col_grid1, col_grid2 = st.columns(2)
    with col_grid1:
        ef_grid = st.number_input("Grid EF (kg CO₂e/kWh):", min_value=0.0, value=0.9, step=0.0001)
    with col_grid2:
        renewable_fraction = st.slider("Renewable supply fraction (% of total electricity):", 0, 100, 0, step=5)

    project_uncert_pct = st.slider("Project emission uncertainty (%):", 0.0, 20.0, 5.0, step=0.5)

    # Effective EF with renewables
    eff_grid_ef = ef_grid * (1 - renewable_fraction / 100.0) + RENEWABLE_EF * (renewable_fraction / 100.0)

    # Adjust for charging losses – convert from kWh drawn to "useful" kWh
    useful_kwh = kwh_year * (charge_eff / 100.0)

    PEy_kg = useful_kwh * eff_grid_ef
    st.write(f"**Project emissions (PEy, year 1): {PEy_kg:,.2f} kg CO₂e/year**")

    st.divider()

    # -------------------------------------------------
    # 4. Project duration, grid decarbonisation & annual series
    # -------------------------------------------------
    st.subheader("3. Project Duration, Grid Decarbonisation & Annual Emissions")

    col_dur1, col_dur2 = st.columns(2)
    with col_dur1:
        years = st.number_input("Project duration (years):", min_value=1, value=10, step=1)
    with col_dur2:
        grid_decarb = st.slider("Annual grid EF reduction (%/year):", 0.0, 10.0, 2.0, step=0.5)

    if BEy_kg <= 0 or kwh_year <= 0:
        st.warning("Enter non-zero baseline fuel and project electricity data to compute reductions.")
        return

    records = []
    current_ef = eff_grid_ef

    # Combine baseline & project uncertainties (approximate root-sum-square)
    u_b = baseline_uncert_pct / 100.0
    u_p = project_uncert_pct / 100.0
    combined_u = (u_b**2 + u_p**2) ** 0.5

    for y in range(1, int(years) + 1):
        year_PEy_kg = useful_kwh * current_ef
        year_BEy_kg = BEy_kg  # baseline assumed constant unless updated by user
        year_ER_kg = year_BEy_kg - year_PEy_kg

        # Uncertainty estimates
        bey_unc_kg = year_BEy_kg * u_b
        pey_unc_kg = year_PEy_kg * u_p
        ery_unc_kg = abs(year_ER_kg) * combined_u

        records.append(
            {
                "Year": y,
                "BEy (kg CO₂e)": year_BEy_kg,
                "BEy ± (kg)": bey_unc_kg,
                "PEy (kg CO₂e)": year_PEy_kg,
                "PEy ± (kg)": pey_unc_kg,
                "Net Reduction (kg)": year_ER_kg,
                "Net Reduction ± (kg)": ery_unc_kg,
                "BEy (t)": year_BEy_kg / 1000.0,
                "PEy (t)": year_PEy_kg / 1000.0,
                "Net Reduction (t)": year_ER_kg / 1000.0,
                "Net Reduction ± (t)": ery_unc_kg / 1000.0,
            }
        )

        # Grid decarbonisation for next year
        current_ef *= (1 - grid_decarb / 100.0)

    df = pd.DataFrame(records)

    total_reduction_kg = df["Net Reduction (kg)"].sum()
    total_reduction_t = total_reduction_kg / 1000.0

    total_reduction_unc_kg = (df["Net Reduction ± (kg)"] ** 2).sum() ** 0.5
    total_reduction_unc_t = total_reduction_unc_kg / 1000.0

    st.markdown("### Annual emission results")
    st.dataframe(df[["Year", "BEy (t)", "PEy (t)", "Net Reduction (t)", "Net Reduction ± (t)"]],
                 use_container_width=True)

    st.success(
        f"**Total emission reductions over {int(years)} years:** "
        f"{total_reduction_t:,.3f} ± {total_reduction_unc_t:,.3f} t CO₂e"
    )

    # -------------------------------------------------
    # 5. Charts
    # -------------------------------------------------
    st.markdown("### Emission trajectories")

    chart_data = df[["Year", "BEy (t)", "PEy (t)", "Net Reduction (t)"]].melt(
        id_vars="Year",
        var_name="Series",
        value_name="tCO2e",
    )

    chart = (
        alt.Chart(chart_data)
        .mark_line(point=True)
        .encode(
            x="Year:O",
            y="tCO2e:Q",
            color="Series:N",
            tooltip=["Year", "Series", "tCO2e"],
        )
    )

    st.altair_chart(chart, use_container_width=True)

    # -------------------------------------------------
    # 6. Download CSV
    # -------------------------------------------------
    st.markdown("### Download annual results")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download VM0038 annual results (CSV)",
        data=csv_bytes,
        file_name="vm0038_ev_annual_results.csv",
        mime="text/csv",
    )

    st.divider()

    # -------------------------------------------------
    # 7. Save to Carbon Registry
    # -------------------------------------------------
    st.subheader("4. Save Total Reduction to Carbon Registry")

    st.caption(
        "This will create a single emission record with the **total lifetime reduction** in tonnes CO₂e. "
        "You can also model per-year records manually using the CSV."
    )

    if st.button("💾 Save total reduction as emission record"):
        with SessionLocal() as session:
            add_emission(
                session,
                project_id=project_id,
                date=date.today(),
                quantity_tCO2e=total_reduction_t,
                activity_type="VM0038 – EV Charging (lifetime)",
                notes=(
                    f"VM0038-like EV charging methodology over {int(years)} years. "
                    f"Total reduction {total_reduction_t:,.3f} ± {total_reduction_unc_t:,.3f} tCO₂e."
                ),
            )
        st.success("Total reduction saved into the Carbon Registry.")

    # -------------------------------------------------
    # 8. Monitoring & QA/QC guidance
    # -------------------------------------------------
    with st.expander("🔍 Monitoring, data & QA/QC hints"):
        st.markdown(
            """
            **Suggested monitoring parameters:**
            - Baseline fuel:
              - Fuel purchase records (litres)
              - Vehicle kilometres travelled (if applicable)
              - Fuel type breakdown (petrol/diesel)
            - Project electricity:
              - Metered kWh per charger / per site
              - Charging sessions and duration
              - EV fleet size and utilisation

            **QA/QC suggestions:**
            - Cross-check fuel data with invoices and logbooks
            - Check meter calibration certificates for chargers
            - Reconcile station-level kWh with utility bills
            - Document all emission factors with source (e.g. IPCC, national inventory, utility EF)

            **Uncertainty:**
            - The uncertainty sliders approximate combined uncertainty using root-sum-square of baseline and project.
            - For formal crediting, a dedicated uncertainty analysis following Verra/VM0038 guidance is recommended.
            """
        )

