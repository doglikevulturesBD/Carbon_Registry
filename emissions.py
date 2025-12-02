# registry/emissions.py
import streamlit as st
from datetime import date
import pandas as pd

from .database import SessionLocal
from . import crud


def render_emissions():
    st.markdown("### 🔥 Emissions per project")

    with SessionLocal() as session:
        projects = crud.list_projects(session)

    if not projects:
        st.info("Create a project first before adding emissions.")
        return

    project_map = {f"{p.id}: {p.name}": p.id for p in projects}
    selected_label = st.selectbox("Select project:", list(project_map.keys()))
    project_id = project_map[selected_label]

    with SessionLocal() as session:
        emissions = crud.list_emissions_for_project(session, project_id)

    st.markdown("#### Add new emission record")
    with st.form("emission_form"):
        col1, col2 = st.columns(2)
        with col1:
            edate = st.date_input("Emission date", value=date.today())
            qty = st.number_input("Quantity (tCO₂e)", min_value=0.0, step=0.01)
        with col2:
            activity = st.text_input("Activity type (e.g. diesel combustion, travel, etc.)")
            notes = st.text_input("Notes", value="")

        add_clicked = st.form_submit_button("➕ Add emission")

    if add_clicked:
        if qty < 0:
            st.error("Quantity cannot be negative.")
        else:
            with SessionLocal() as session:
                proj = crud.get_project(session, project_id)
                if proj:
                    if proj.start_date and edate < proj.start_date:
                        st.error("Emission date is before project start date.")
                    elif proj.end_date and edate > proj.end_date:
                        st.error("Emission date is after project end date.")
                    else:
                        crud.add_emission(
                            session,
                            project_id=project_id,
                            date=edate,
                            quantity_tCO2e=qty,
                            activity_type=activity.strip(),
                            notes=notes.strip(),
                        )
                        st.success("Emission added.")
                        st.experimental_rerun()
                else:
                    st.error("Project not found.")

    st.markdown("#### Existing emissions")
    if emissions:
        df = pd.DataFrame(
            [
                {
                    "ID": e.id,
                    "Date": e.date,
                    "Qty (tCO₂e)": e.quantity_tCO2e,
                    "Activity": e.activity_type,
                    "Notes": e.notes,
                }
                for e in emissions
            ]
        )
        st.dataframe(df, use_container_width=True)

        emission_ids = [e.id for e in emissions]
        delete_id = st.selectbox(
            "Select emission ID to delete:",
            options=["None"] + emission_ids,
        )
        if delete_id != "None":
            if st.button("🗑️ Delete selected emission"):
                with SessionLocal() as session:
                    ok = crud.delete_emission(session, int(delete_id))
                if ok:
                    st.success("Emission deleted.")
                    st.experimental_rerun()
                else:
                    st.error("Emission not found.")
    else:
        st.info("No emissions recorded yet for this project.")

