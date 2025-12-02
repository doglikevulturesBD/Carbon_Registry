import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# EMISSION FACTOR TABLES (MVP VERSION)
# ---------------------------------------------------------

BASELINE_EF = {
    "Plastic": 1.3,     # landfill/incineration average
    "Paper": 0.9,
    "Metal": 1.5,
    "Glass": 0.4,
}

PROJECT_EF = {
    "Plastic": 0.55,    # recycling benefit + avoided virgin
    "Paper": 0.35,
    "Metal": 0.20,
    "Glass": 0.15,
}

TRANSPORT_EF = 0.000102  # tCO2e per tonne-km (generic default)
DIESEL_EF = 0.0027       # tCO2e per litre diesel
ELECTRICITY_EF = 0.0009  # tCO2e per kWh (MVP default)


# ---------------------------------------------------------
# CALCULATION FUNCTIONS
# ---------------------------------------------------------

def calculate_baseline(material, tons):
    ef = BASELINE_EF[material]
    return tons * ef


def calculate_project_emissions(material, tons_recycled, contamination, diesel_litres, kwh, transport_km):
    clean_tons = tons_recycled * (1 - contamination / 100)
    residue_tons = tons_recycled - clean_tons

    pe_recycling = clean_tons * PROJECT_EF[material]
    pe_residuals = residue_tons * BASELINE_EF[material]  # assume landfilled
    pe_transport = clean_tons * transport_km * TRANSPORT_EF
    pe_energy = (diesel_litres * DIESEL_EF) + (kwh * ELECTRICITY_EF)

    total_pe = pe_recycling + pe_residuals + pe_transport + pe_energy

    return {
        "clean_tons": clean_tons,
        "residual_tons": residue_tons,
        "pe_recycling": pe_recycling,
        "pe_residuals": pe_residuals,
        "pe_transport": pe_transport,
        "pe_energy": pe_energy,
        "total_pe": total_pe,
    }


def calculate_er(baseline, project):
    return baseline - project


# ---------------------------------------------------------
# REGISTRY SAVING FUNCTION
# ---------------------------------------------------------

def save_to_registry(result):
    csv_path = "registry_data.csv"

    try:
        df = pd.read_csv(csv_path)
    except:
        df = pd.DataFrame(columns=result.keys())

    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    df.to_csv(csv_path, index=False)


# ---------------------------------------------------------
# MAIN STREAMLIT APP FUNCTION
# ---------------------------------------------------------

def vmr0007_app():

    st.title("♻ VMR0007 – Solid Waste Recovery & Recycling (MVP)")
    st.caption("Education-focused version. Calculates ERs for recycling vs baseline disposal.")

    st.markdown("### 1️⃣ Material Selection & Input Data")
    
    col1, col2 = st.columns(2)

    with col1:
        material = st.selectbox(
            "Material Type",
            ["Plastic", "Paper", "Metal", "Glass"]
        )

        tons = st.number_input(
            "Total Material Processed (tons/year)",
            min_value=0.0, step=0.1, value=100.0
        )

        contamination = st.slider(
            "Contamination Rate (%)",
            min_value=0, max_value=40, value=10
        )

    with col2:
        diesel = st.number_input(
            "Diesel Used (litres/year)", min_value=0.0, value=500.0
        )
        electricity = st.number_input(
            "Electricity Use (kWh/year)", min_value=0.0, value=30000.0
        )
        distance = st.number_input(
            "Transport Distance (km)", min_value=0.0, value=15.0
        )

    st.divider()

    # ---------------------------------------------------------
    # CALCULATIONS
    # ---------------------------------------------------------

    baseline_emissions = calculate_baseline(material, tons)
    project_data = calculate_project_emissions(material, tons, contamination, diesel, electricity, distance)
    project_emissions = project_data["total_pe"]
    er = calculate_er(baseline_emissions, project_emissions)

    # ---------------------------------------------------------
    # RESULTS DISPLAY
    # ---------------------------------------------------------

    st.markdown("## 2️⃣ Results")

    st.metric("Baseline Emissions (tCO₂e/yr)", f"{baseline_emissions:,.2f}")
    st.metric("Project Emissions (tCO₂e/yr)", f"{project_emissions:,.2f}")
    st.metric("Net Emission Reductions (tCO₂e/yr)", f"{er:,.2f}")

    st.markdown("### Breakdown")
    st.write(pd.DataFrame({
        "Parameter": [
            "Clean Recyclable Tons", "Residual Tons", 
            "Project – Recycling", "Project – Residual Disposal",
            "Transport Emissions", "Energy Emissions"
        ],
        "Value (tCO₂e or tons)": [
            f"{project_data['clean_tons']:.2f}",
            f"{project_data['residual_tons']:.2f}",
            f"{project_data['pe_recycling']:.2f}",
            f"{project_data['pe_residuals']:.2f}",
            f"{project_data['pe_transport']:.2f}",
            f"{project_data['pe_energy']:.2f}",
        ]
    }))

    st.divider()

    # ---------------------------------------------------------
    # SAVE TO REGISTRY
    # ---------------------------------------------------------

    st.markdown("### 3️⃣ Save Result to Registry")

    project_name = st.text_input("Project Name")
    operator = st.text_input("Operator Name")
    year = st.number_input("Reporting Year", min_value=2020, value=2025)

    if st.button("💾 Save to Registry"):
        if project_name.strip() == "":
            st.error("Project name required.")
        else:
            save_to_registry({
                "Project": project_name,
                "Operator": operator,
                "Year": year,
                "Material": material,
                "Total Tons": tons,
                "Contamination %": contamination,
                "Baseline Emissions": baseline_emissions,
                "Project Emissions": project_emissions,
                "Emission Reductions": er
            })

            st.success("Saved to local registry!")

