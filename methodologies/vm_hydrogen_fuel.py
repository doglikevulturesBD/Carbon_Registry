import streamlit as st
import pandas as pd
import numpy as np


# -----------------------------------------------
# REGISTRY SAVING (CSV-based MVP)
# -----------------------------------------------
def save_to_registry(result: dict):
    path = "registry_data.csv"

    try:
        df = pd.read_csv(path)
    except:
        df = pd.DataFrame(columns=result.keys())

    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    df.to_csv(path, index=False)


# -----------------------------------------------
# STREAMLIT HYDROGEN CALCULATOR (AM0124, MVP)
# -----------------------------------------------
def am0124_hydrogen_app():

    st.title("🟦 AM0124 – Hydrogen Electrolysis Emissions Calculator (MVP)")
    st.caption("Educational & demonstration version aligned with AM0124 baseline and project emissions.")

    st.markdown("## 1️⃣ Project Inputs")

    col1, col2 = st.columns(2)

    # --------------------- Project Quantities ---------------------
    with col1:
        MH2 = st.number_input(
            "Hydrogen Produced (t H₂ / year)",
            min_value=0.0, value=1000.0, step=10.0
        )

        baseline = st.selectbox(
            "Baseline Technology",
            ["Coal (gasification + SMR)", "Natural Gas (SMR)", "Oil (gasification + SMR)"]
        )

        grid_mwh = st.number_input(
            "Grid Electricity Used (MWh / year)", min_value=0.0, value=500.0
        )

        captive_mwh = st.number_input(
            "Captive Renewable Electricity (MWh / year)", min_value=0.0, value=9500.0
        )

        grid_ef = st.number_input(
            "Grid Emission Factor (t CO₂ / MWh)", min_value=0.0, value=1.3
        )

    with col2:
        fossil_gj = st.number_input(
            "On-site Fossil Fuel Use (GJ / year)",
            min_value=0.0, value=0.0
        )

        fossil_ef = st.number_input(
            "CO₂ Factor of Fuel (t CO₂ / GJ)",
            min_value=0.0, value=0.0,
            help="E.g., Diesel ≈ 0.074 tCO₂/GJ"
        )

        transport_t = st.number_input(
            "Transport Emissions (t CO₂e / year)",
            min_value=0.0, value=0.0,
            help="Can be 0 if pipeline is powered by captive plant"
        )

        leak_pct = st.number_input(
            "Hydrogen Leak Rate (%)",
            min_value=0.0, value=5.0
        )

        gwp_h2 = st.number_input(
            "GWP of Hydrogen (t CO₂e / t H₂)",
            min_value=0.0, value=5.8
        )

        years = st.number_input(
            "Project Duration (years)",
            min_value=1, value=10
        )

    st.divider()

    # ==============================================================
    # --------------------- CALCULATIONS ---------------------------
    # ==============================================================

    # Baseline EF
    if "Coal" in baseline:
        EF_BL = 19.0
    else:
        EF_BL = 9.0

    # Applicability ratio
    ratio = (grid_mwh / captive_mwh) if captive_mwh > 0 else np.inf

    if ratio >= 0.1:
        st.warning(
            f"⚠️ AM0124 compliance check failed: grid/captive ratio must be < 0.1.\nCurrent ratio = {ratio:.3f}"
        )

    # Baseline emissions
    BEy = MH2 * EF_BL

    # Project emissions
    PE_ec = grid_mwh * grid_ef
    PE_fc = fossil_gj * fossil_ef if (fossil_gj > 0 and fossil_ef > 0) else 0.0
    PE_leak = MH2 * (leak_pct / 100) * gwp_h2
    PE_tr = transport_t
    PEy = PE_ec + PE_fc + PE_tr + PE_leak

    # Net ER
    ERy = BEy - PEy
    total_ER = ERy * years

    # ==============================================================
    # ------------------------- RESULTS ----------------------------
    # ==============================================================

    st.markdown("## 2️⃣ Results")

    st.metric("Baseline EF (t CO₂ / t H₂)", f"{EF_BL:.2f}")
    st.metric("Baseline Emissions (t CO₂e / year)", f"{BEy:,.2f}")
    st.metric("Project Emissions (t CO₂e / year)", f"{PEy:,.2f}")
    st.metric("Net ERᵧ (t CO₂e / year)", f"{ERy:,.2f}")
    st.metric("Total ER over Project Life (t CO₂e)", f"{total_ER:,.2f}")

    st.divider()

    st.markdown("### 🔍 Breakdown of Project Emissions")

    df = pd.DataFrame({
        "Component": [
            "Grid Electricity",
            "On-site Fossil Fuel",
            "Transport",
            "Hydrogen Leaks"
        ],
        "Emissions (t CO₂e/year)": [
            PE_ec,
            PE_fc,
            PE_tr,
            PE_leak
        ]
    })

    st.dataframe(df, use_container_width=True)

    st.divider()

    # ==============================================================
    # ------------------- SAVE INTO REGISTRY -----------------------
    # ==============================================================

    st.markdown("## 3️⃣ Save Result to Local Registry")

    project_name = st.text_input("Project Name")
    operator = st.text_input("Operator")
    year = st.number_input("Reporting Year", min_value=2020, value=2025)

    if st.button("💾 Save to Registry"):
        if project_name == "":
            st.error("Project name is required.")
        else:
            save_to_registry({
                "Project": project_name,
                "Operator": operator,
                "Year": year,
                "Methodology": "AM0124 Hydrogen",
                "Hydrogen (t/yr)": MH2,
                "Baseline EF": EF_BL,
                "Baseline Emissions": BEy,
                "Project Emissions": PEy,
                "Net ER (annual)": ERy,
                "Total ER (lifetime)": total_ER,
                "Grid/Captive Ratio": ratio
            })
            st.success("Saved successfully to registry!")


