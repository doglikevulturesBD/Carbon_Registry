# registry/dashboard.py
import pandas as pd
import streamlit as st
import altair as alt

from .database import SessionLocal
from . import crud


def render_dashboard():
    st.markdown("### 📊 Registry Dashboard")

    with SessionLocal() as session:
        total_projects, total_emissions, last30 = crud.get_kpis(session)
        bu_totals = crud.get_totals_by_business_unit(session)
        recent = crud.get_recent_emissions(session)

    # KPI cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Projects", total_projects)
    col2.metric("Total Emissions (tCO₂e)", f"{total_emissions:.2f}")
    col3.metric("Last 30 days (tCO₂e)", f"{last30:.2f}")

    st.markdown("---")

    # Business unit totals
    st.markdown("#### Emissions by Business Unit")
    if bu_totals:
        df_bu = pd.DataFrame(bu_totals, columns=["Business Unit", "tCO₂e"])
        chart = (
            alt.Chart(df_bu)
            .mark_bar()
            .encode(
                x="Business Unit",
                y="tCO₂e",
                tooltip=["Business Unit", "tCO₂e"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(df_bu, use_container_width=True)
    else:
        st.info("No emissions recorded yet.")

    # Recent emissions
    st.markdown("#### Recent Emissions (latest 10)")
    if recent:
        df_recent = pd.DataFrame(
            recent,
            columns=["ID", "Date", "Project", "Qty (tCO₂e)", "Activity"],
        )
        st.dataframe(df_recent, use_container_width=True)
    else:
        st.info("No recent emissions.")

