# pages/3_Methodologies.py
# ------------------------------------------------------------
# Carbon Registry • Methodology Demos (single-file launcher)
# VM0038 (EV charging) • AM0124 (Hydrogen electrolysis) • VMR0007 (Waste recycling)
#
# Goal: "launch-proof" demo without import/package issues.
# Later you can split back into methodologies/ modules.
# ------------------------------------------------------------

import streamlit as st
from datetime import date
import pandas as pd
import numpy as np

# Altair is optional. If you don't have it installed, set USE_ALTAIR = False.
USE_ALTAIR = True
try:
    import altair as alt
except Exception:
    USE_ALTAIR = False


# ============================================================
# Shared utilities
# ============================================================

def save_to_registry_csv(result: dict, path: str = "registry_data.csv"):
    """Very lightweight demo 'registry' save into a local CSV file."""
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame(columns=list(result.keys()))

    # ensure columns include all keys
    for k in result.keys():
        if k not in df.columns:
            df[k] = None

    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    df.to_csv(path, index=False)
    return path


def section_note(text: str):
    st.caption(text)


# ============================================================
# VM0038 – EV Charging (demo)
# ============================================================

def vm0038_ev_demo():
    st.header("⚡ VM0038 – EV Charging System (Demo)")
    st.caption("Baseline: ICE fuel | Project: EV charging electricity (grid/renewable). Demo math only.")

    # Default emission factors (kg CO2e / litre) — demo placeholders
    FUEL_EF = {"Petrol": 2.31, "Diesel": 2.68, "LPG": 1.51, "Other": None}
    WTT_EF = {"Petrol": 0.52, "Diesel": 0.58, "LPG": 0.21}  # upstream WTT
    FUEL_ENERGY_MJ = {"Petrol": 34.2, "Diesel": 38.6, "LPG": 26.8, "Other": 0.0}
    RENEWABLE_EF = 0.0  # demo assumption

    with st.expander("📘 Methodology overview (VM0038-style)", expanded=False):
        st.markdown(
            r"""
- **Baseline emissions (BEy)**: fuel avoided  
  \( BE_y = FC_{fuel,y} \times (EF_{fuel} + EF_{WTT}) \)

- **Project emissions (PEy)**: charging electricity  
  \( PE_y = E_{EV,y} \times EF_{grid,y} \times (1 - r_{renew}) \)

- **Net reduction**: \( ER_y = BE_y - PE_y \)

This is a **demo implementation** for education and discussion (not an official Verra calculator).
"""
        )

    st.subheader("1) Baseline – Fuel avoided")
    c1, c2 = st.columns(2)
    with c1:
        fuel_type = st.selectbox("Fuel type", list(FUEL_EF.keys()), key="vm0038_fuel")
        fuel_use_l = st.number_input("Fuel avoided (litres/year)", min_value=0.0, step=0.01, value=10000.0, key="vm0038_l")
    with c2:
        default_fuel_ef = FUEL_EF[fuel_type] if FUEL_EF[fuel_type] is not None else 0.0
        ef_fuel = st.number_input("Fuel EF (kg CO₂e/litre)", value=float(default_fuel_ef), min_value=0.0, step=0.0001, key="vm0038_ef")
        include_wtt = st.checkbox("Include WTT (upstream)", value=True, key="vm0038_wtt")

    wtt_ef = WTT_EF.get(fuel_type, 0.0) if include_wtt else 0.0
    BEy_kg = fuel_use_l * (ef_fuel + wtt_ef)

    # Energy equivalence (diagnostic)
    energy_mj_year = fuel_use_l * FUEL_ENERGY_MJ.get(fuel_type, 0.0)
    st.info(f"Fuel energy equivalent (diagnostic): **{energy_mj_year:,.1f} MJ/year**")

    baseline_uncert_pct = st.slider("Baseline uncertainty (%)", 0.0, 20.0, 5.0, step=0.5, key="vm0038_u_b")

    st.write(f"**Baseline emissions (BEy): {BEy_kg:,.2f} kg CO₂e/year**")

    st.divider()
    st.subheader("2) Project – EV charging electricity")

    mode = st.radio(
        "Define project electricity use:",
        ["Direct annual kWh", "From charger fleet parameters"],
        horizontal=True,
        key="vm0038_mode",
    )

    if mode == "Direct annual kWh":
        c1, c2 = st.columns(2)
        with c1:
            kwh_year = st.number_input("EV charging electricity (kWh/year)", min_value=0.0, step=0.01, value=250000.0, key="vm0038_kwh")
        with c2:
            charge_eff = st.slider("Charging efficiency (%)", 70, 100, 90, key="vm0038_eff")
    else:
        st.markdown("**Charger fleet parameters**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            n_chargers = st.number_input("Number of chargers", min_value=1, value=4, step=1, key="vm0038_n")
        with c2:
            sessions_per_day = st.number_input("Sessions/charger/day", min_value=0.0, value=4.0, step=0.5, key="vm0038_sess")
        with c3:
            kwh_per_session = st.number_input("kWh/session", min_value=0.0, value=20.0, step=0.5, key="vm0038_kwhps")
        with c4:
            operating_days = st.number_input("Operating days/year", min_value=0, value=300, step=1, key="vm0038_days")

        kwh_year = n_chargers * sessions_per_day * kwh_per_session * operating_days
        st.info(f"Derived annual charging electricity: **{kwh_year:,.1f} kWh/year**")
        charge_eff = st.slider("Charging efficiency (%)", 70, 100, 90, key="vm0038_eff2")

    c1, c2 = st.columns(2)
    with c1:
        ef_grid = st.number_input("Grid EF (kg CO₂e/kWh)", min_value=0.0, value=0.9, step=0.0001, key="vm0038_grid")
    with c2:
        renewable_fraction = st.slider("Renewable fraction (%)", 0, 100, 0, step=5, key="vm0038_ren")

    project_uncert_pct = st.slider("Project uncertainty (%)", 0.0, 20.0, 5.0, step=0.5, key="vm0038_u_p")

    eff_grid_ef = ef_grid * (1 - renewable_fraction / 100.0) + RENEWABLE_EF * (renewable_fraction / 100.0)
    useful_kwh = kwh_year * (charge_eff / 100.0)
    PEy_kg = useful_kwh * eff_grid_ef

    st.write(f"**Project emissions (PEy, year 1): {PEy_kg:,.2f} kg CO₂e/year**")

    st.divider()
    st.subheader("3) Duration & grid decarbonisation")

    c1, c2 = st.columns(2)
    with c1:
        years = st.number_input("Project duration (years)", min_value=1, value=10, step=1, key="vm0038_years")
    with c2:
        grid_decarb = st.slider("Annual grid EF reduction (%/year)", 0.0, 10.0, 2.0, step=0.5, key="vm0038_decarb")

    if BEy_kg <= 0 or kwh_year <= 0:
        st.warning("Enter non-zero baseline fuel and project electricity data to compute reductions.")
        return

    records = []
    current_ef = eff_grid_ef

    u_b = baseline_uncert_pct / 100.0
    u_p = project_uncert_pct / 100.0
    combined_u = (u_b**2 + u_p**2) ** 0.5

    for y in range(1, int(years) + 1):
        year_BEy_kg = BEy_kg  # constant baseline in this demo
        year_PEy_kg = useful_kwh * current_ef
        year_ER_kg = year_BEy_kg - year_PEy_kg

        bey_unc_kg = year_BEy_kg * u_b
        pey_unc_kg = year_PEy_kg * u_p
        ery_unc_kg = abs(year_ER_kg) * combined_u

        records.append(
            {
                "Year": y,
                "BEy (t)": year_BEy_kg / 1000.0,
                "PEy (t)": year_PEy_kg / 1000.0,
                "Net Reduction (t)": year_ER_kg / 1000.0,
                "Net Reduction ± (t)": ery_unc_kg / 1000.0,
            }
        )
        current_ef *= (1 - grid_decarb / 100.0)

    df = pd.DataFrame(records)
    total_reduction_t = float(df["Net Reduction (t)"].sum())
    total_unc_t = float((df["Net Reduction ± (t)"] ** 2).sum() ** 0.5)

    st.markdown("### Annual results")
    st.dataframe(df, use_container_width=True)

    st.success(f"**Total reductions over {int(years)} years:** {total_reduction_t:,.3f} ± {total_unc_t:,.3f} t CO₂e")

    if USE_ALTAIR:
        st.markdown("### Chart")
        chart_data = df.melt(id_vars="Year", var_name="Series", value_name="tCO2e")
        chart = (
            alt.Chart(chart_data)
            .mark_line(point=True)
            .encode(x="Year:O", y="tCO2e:Q", color="Series:N", tooltip=["Year", "Series", "tCO2e"])
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Altair not installed; chart disabled. (pip install altair)")

    st.markdown("### Download CSV")
    st.download_button(
        "💾 Download VM0038 annual results (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="vm0038_ev_annual_results.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("4) Save summary to local demo registry (CSV)")

    project_name = st.text_input("Project name", key="vm0038_pname")
    operator = st.text_input("Operator", key="vm0038_op")
    reporting_year = st.number_input("Reporting year", min_value=2020, value=date.today().year, step=1, key="vm0038_ry")

    if st.button("💾 Save to demo registry CSV", key="vm0038_save"):
        if not project_name.strip():
            st.error("Project name is required.")
        else:
            path = save_to_registry_csv(
                {
                    "Project": project_name,
                    "Operator": operator,
                    "Year": int(reporting_year),
                    "Methodology": "VM0038 – EV Charging (demo)",
                    "Baseline Fuel (L/yr)": float(fuel_use_l),
                    "Fuel Type": fuel_type,
                    "BEy (tCO2e/yr)": float(BEy_kg / 1000.0),
                    "Electricity (kWh/yr)": float(kwh_year),
                    "Grid EF (kg/kWh)": float(ef_grid),
                    "Renewable %": int(renewable_fraction),
                    "PEy (tCO2e/yr, y1)": float(PEy_kg / 1000.0),
                    "Years": int(years),
                    "Total ER (tCO2e)": float(total_reduction_t),
                    "Total ER ± (tCO2e)": float(total_unc_t),
                }
            )
            st.success(f"Saved to {path} ✅")


# ============================================================
# AM0124 – Hydrogen electrolysis (demo)
# ============================================================

def am0124_hydrogen_demo():
    st.header("🟦 AM0124 – Hydrogen Electrolysis (Demo)")
    st.caption("Educational demo aligned to a simplified AM0124-style structure.")

    with st.expander("📘 Methodology overview (AM0124-style)", expanded=False):
        st.markdown(
            r"""
- Baseline emissions: \( BE_y = M_{H2} \times EF_{BL} \)
- Project emissions: grid + fuel + transport + leakage proxy  
- Net reductions: \( ER_y = BE_y - PE_y \)

This is a **demo** calculator: you must replace defaults with cited factors for any real MRV use.
"""
        )

    st.markdown("## 1️⃣ Inputs")
    col1, col2 = st.columns(2)

    with col1:
        MH2 = st.number_input("Hydrogen produced (t H₂ / year)", min_value=0.0, value=1000.0, step=10.0, key="am_mh2")
        baseline = st.selectbox(
            "Baseline technology",
            ["Coal (gasification + SMR)", "Natural Gas (SMR)", "Oil (gasification + SMR)"],
            key="am_base",
        )
        grid_mwh = st.number_input("Grid electricity used (MWh / year)", min_value=0.0, value=500.0, key="am_grid")
        captive_mwh = st.number_input("Captive renewable electricity (MWh / year)", min_value=0.0, value=9500.0, key="am_cap")
        grid_ef = st.number_input("Grid emission factor (t CO₂ / MWh)", min_value=0.0, value=1.3, key="am_grid_ef")

    with col2:
        fossil_gj = st.number_input("On-site fossil fuel use (GJ / year)", min_value=0.0, value=0.0, key="am_fossil_gj")
        fossil_ef = st.number_input("CO₂ factor of fuel (t CO₂ / GJ)", min_value=0.0, value=0.0, key="am_fossil_ef")
        transport_t = st.number_input("Transport emissions (t CO₂e / year)", min_value=0.0, value=0.0, key="am_tr")
        leak_pct = st.number_input("Hydrogen leak rate (%)", min_value=0.0, value=5.0, key="am_leak")
        gwp_h2 = st.number_input("GWP of H₂ (t CO₂e / t H₂)", min_value=0.0, value=5.8, key="am_gwp")
        years = st.number_input("Project duration (years)", min_value=1, value=10, key="am_years")

    st.divider()

    # Baseline EF (demo)
    EF_BL = 19.0 if "Coal" in baseline else 9.0

    # Applicability ratio check (demo rule from your code)
    ratio = (grid_mwh / captive_mwh) if captive_mwh > 0 else np.inf
    if ratio >= 0.1:
        st.warning(f"⚠️ Demo compliance check: grid/captive ratio should be < 0.1. Current ratio = {ratio:.3f}")

    # Calculations
    BEy = MH2 * EF_BL
    PE_ec = grid_mwh * grid_ef
    PE_fc = fossil_gj * fossil_ef if (fossil_gj > 0 and fossil_ef > 0) else 0.0
    PE_leak = MH2 * (leak_pct / 100.0) * gwp_h2
    PE_tr = transport_t
    PEy = PE_ec + PE_fc + PE_tr + PE_leak

    ERy = BEy - PEy
    total_ER = ERy * years

    st.markdown("## 2️⃣ Results")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Baseline EF", f"{EF_BL:.2f} tCO₂/tH₂")
    c2.metric("BEy", f"{BEy:,.2f} tCO₂e/yr")
    c3.metric("PEy", f"{PEy:,.2f} tCO₂e/yr")
    c4.metric("ERy", f"{ERy:,.2f} tCO₂e/yr")
    c5.metric("Total ER", f"{total_ER:,.2f} tCO₂e")

    st.markdown("### Breakdown of project emissions")
    df = pd.DataFrame(
        {
            "Component": ["Grid electricity", "On-site fossil fuel", "Transport", "Hydrogen leaks"],
            "Emissions (tCO₂e/yr)": [PE_ec, PE_fc, PE_tr, PE_leak],
        }
    )
    st.dataframe(df, use_container_width=True)

    st.divider()
    st.markdown("## 3️⃣ Save summary to local demo registry (CSV)")

    project_name = st.text_input("Project name", key="am_pname")
    operator = st.text_input("Operator", key="am_op")
    reporting_year = st.number_input("Reporting year", min_value=2020, value=date.today().year, step=1, key="am_ry")

    if st.button("💾 Save to demo registry CSV", key="am_save"):
        if not project_name.strip():
            st.error("Project name is required.")
        else:
            path = save_to_registry_csv(
                {
                    "Project": project_name,
                    "Operator": operator,
                    "Year": int(reporting_year),
                    "Methodology": "AM0124 – Hydrogen (demo)",
                    "H2 (t/yr)": float(MH2),
                    "Baseline Tech": baseline,
                    "Baseline EF (tCO2/tH2)": float(EF_BL),
                    "BEy (tCO2e/yr)": float(BEy),
                    "Grid (MWh/yr)": float(grid_mwh),
                    "Captive (MWh/yr)": float(captive_mwh),
                    "Grid EF (t/MWh)": float(grid_ef),
                    "Fossil (GJ/yr)": float(fossil_gj),
                    "Fossil EF (t/GJ)": float(fossil_ef),
                    "Transport (t/yr)": float(transport_t),
                    "Leak %": float(leak_pct),
                    "GWP H2": float(gwp_h2),
                    "PEy (tCO2e/yr)": float(PEy),
                    "ERy (tCO2e/yr)": float(ERy),
                    "Years": int(years),
                    "Total ER (tCO2e)": float(total_ER),
                    "Grid/Captive Ratio": float(ratio) if np.isfinite(ratio) else None,
                }
            )
            st.success(f"Saved to {path} ✅")


# ============================================================
# VMR0007 – Waste recycling (demo)
# ============================================================

def vmr0007_waste_demo():
    st.header("♻ VMR0007 – Solid Waste Recovery & Recycling (Demo)")
    st.caption("Education-focused demo: baseline disposal vs project recycling + residue + transport + energy.")

    # Demo factor tables (placeholders)
    BASELINE_EF = {"Plastic": 1.3, "Paper": 0.9, "Metal": 1.5, "Glass": 0.4}
    PROJECT_EF = {"Plastic": 0.55, "Paper": 0.35, "Metal": 0.20, "Glass": 0.15}
    TRANSPORT_EF = 0.000102  # tCO2e per tonne-km (demo)
    DIESEL_EF = 0.0027       # tCO2e per litre diesel (demo)
    ELECTRICITY_EF = 0.0009  # tCO2e per kWh (demo)

    def calculate_baseline(material, tons):
        return tons * BASELINE_EF[material]

    def calculate_project_emissions(material, tons_recycled, contamination, diesel_litres, kwh, transport_km):
        clean_tons = tons_recycled * (1 - contamination / 100.0)
        residue_tons = tons_recycled - clean_tons

        pe_recycling = clean_tons * PROJECT_EF[material]
        pe_residuals = residue_tons * BASELINE_EF[material]  # assume residues disposed
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

    st.markdown("### 1️⃣ Inputs")
    c1, c2 = st.columns(2)
    with c1:
        material = st.selectbox("Material type", ["Plastic", "Paper", "Metal", "Glass"], key="w_mat")
        tons = st.number_input("Total material processed (tons/year)", min_value=0.0, step=0.1, value=100.0, key="w_tons")
        contamination = st.slider("Contamination rate (%)", min_value=0, max_value=40, value=10, key="w_cont")
    with c2:
        diesel = st.number_input("Diesel used (litres/year)", min_value=0.0, value=500.0, key="w_diesel")
        electricity = st.number_input("Electricity use (kWh/year)", min_value=0.0, value=30000.0, key="w_kwh")
        distance = st.number_input("Transport distance (km)", min_value=0.0, value=15.0, key="w_km")

    st.divider()

    baseline_emissions = calculate_baseline(material, tons)
    project_data = calculate_project_emissions(material, tons, contamination, diesel, electricity, distance)
    project_emissions = project_data["total_pe"]
    er = baseline_emissions - project_emissions

    st.markdown("## 2️⃣ Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline (tCO₂e/yr)", f"{baseline_emissions:,.2f}")
    c2.metric("Project (tCO₂e/yr)", f"{project_emissions:,.2f}")
    c3.metric("Net ER (tCO₂e/yr)", f"{er:,.2f}")

    st.markdown("### Breakdown")
    st.dataframe(
        pd.DataFrame(
            {
                "Parameter": [
                    "Clean recyclable tons",
                    "Residual tons",
                    "Project – Recycling",
                    "Project – Residual disposal",
                    "Transport emissions",
                    "Energy emissions",
                ],
                "Value": [
                    f"{project_data['clean_tons']:.2f}",
                    f"{project_data['residual_tons']:.2f}",
                    f"{project_data['pe_recycling']:.2f}",
                    f"{project_data['pe_residuals']:.2f}",
                    f"{project_data['pe_transport']:.2f}",
                    f"{project_data['pe_energy']:.2f}",
                ],
            }
        ),
        use_container_width=True,
    )

    st.divider()
    st.markdown("## 3️⃣ Save summary to local demo registry (CSV)")

    project_name = st.text_input("Project name", key="w_pname")
    operator = st.text_input("Operator", key="w_op")
    reporting_year = st.number_input("Reporting year", min_value=2020, value=date.today().year, step=1, key="w_ry")

    if st.button("💾 Save to demo registry CSV", key="w_save"):
        if not project_name.strip():
            st.error("Project name is required.")
        else:
            path = save_to_registry_csv(
                {
                    "Project": project_name,
                    "Operator": operator,
                    "Year": int(reporting_year),
                    "Methodology": "VMR0007 – Waste (demo)",
                    "Material": material,
                    "Total Tons": float(tons),
                    "Contamination %": int(contamination),
                    "Baseline (tCO2e/yr)": float(baseline_emissions),
                    "Project (tCO2e/yr)": float(project_emissions),
                    "ER (tCO2e/yr)": float(er),
                    "Diesel (L/yr)": float(diesel),
                    "Electricity (kWh/yr)": float(electricity),
                    "Distance (km)": float(distance),
                }
            )
            st.success(f"Saved to {path} ✅")


# ============================================================
# Launcher
# ============================================================

st.title("📘 Methodology Demos (Single File)")
section_note("This page is intentionally single-file for a stable demo/launch. Split into modules later.")

choice = st.selectbox(
    "Select a demo methodology:",
    [
        "VM0038 – EV Charging (demo)",
        "AM0124 – Hydrogen Electrolysis (demo)",
        "VMR0007 – Waste Recovery & Recycling (demo)",
    ],
)

st.divider()

if choice.startswith("VM0038"):
    vm0038_ev_demo()
elif choice.startswith("AM0124"):
    am0124_hydrogen_demo()
else:
    vmr0007_waste_demo()

st.divider()
st.caption(
    "Important: These are demonstration calculators for discussion and prototyping. "
    "Defaults are placeholders unless you supply cited emission factors."
)
