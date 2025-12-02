# registry/projects.py
import streamlit as st
from datetime import date

from .database import SessionLocal
from .models import BUSINESS_UNITS
from . import crud


def render_projects():
    st.markdown("### 📂 Projects")

    # Load project list
    with SessionLocal() as session:
        projects = crud.list_projects(session)

    project_options = ["➕ New project"] + [
        f"{p.id}: {p.name}" for p in projects
    ]

    selected_label = st.selectbox(
        "Select project to view / edit:",
        project_options,
    )

    selected_id = None
    if selected_label != "➕ New project":
        selected_id = int(selected_label.split(":")[0])

    # Fetch selected project if any
    name = ""
    location = ""
    bu = list(BUSINESS_UNITS.keys())[0]
    sub_div = BUSINESS_UNITS[bu][0]
    start = date.today()
    end = date.today()
    description = ""

    if selected_id is not None:
        with SessionLocal() as session:
            proj = crud.get_project(session, selected_id)
            if proj:
                name = proj.name or ""
                location = proj.location or ""
                bu = proj.business_unit or bu
                sub_div_list = BUSINESS_UNITS.get(bu, [])
                sub_div = proj.sub_division or (sub_div_list[0] if sub_div_list else "")
                start = proj.start_date or start
                end = proj.end_date or end
                description = proj.description or ""

    with st.form("project_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Project name", value=name)
            location = st.text_input("Location", value=location)
            bu = st.selectbox("Business Unit", options=list(BUSINESS_UNITS.keys()), index=list(BUSINESS_UNITS.keys()).index(bu))
            sub_div = st.selectbox(
                "Sub-division",
                options=BUSINESS_UNITS.get(bu, []),
                index=BUSINESS_UNITS.get(bu, []).index(sub_div) if sub_div in BUSINESS_UNITS.get(bu, []) else 0,
            )
        with col2:
            start = st.date_input("Start date", value=start)
            end = st.date_input("End date", value=end)
            description = st.text_area("Description", value=description, height=120)

        col_save, col_delete = st.columns([2, 1])
        with col_save:
            save_clicked = st.form_submit_button("💾 Save project")
        with col_delete:
            delete_clicked = st.form_submit_button("🗑️ Delete project", disabled=(selected_id is None))

    # Handle form actions
    if save_clicked:
        if not name.strip():
            st.error("Project name is required.")
        elif end < start:
            st.error("End date cannot be before start date.")
        else:
            with SessionLocal() as session:
                existing = crud.find_project_by_name(session, name.strip())
                if existing and existing.id != selected_id:
                    st.error(f"A project named '{name}' already exists (ID {existing.id}).")
                else:
                    if selected_id is None:
                        crud.create_project(
                            session,
                            name.strip(),
                            location.strip(),
                            bu,
                            sub_div,
                            start,
                            end,
                            description.strip(),
                        )
                        st.success("Project created.")
                    else:
                        updated = crud.update_project(
                            session,
                            selected_id,
                            name.strip(),
                            location.strip(),
                            bu,
                            sub_div,
                            start,
                            end,
                            description.strip(),
                        )
                        if updated:
                            st.success("Project updated.")
                        else:
                            st.error("Project not found. It may have been deleted.")

            st.experimental_rerun()

    if delete_clicked and selected_id is not None:
        with SessionLocal() as session:
            ok = crud.delete_project(session, selected_id)
        if ok:
            st.success("Project deleted.")
        else:
            st.error("Project not found.")
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("#### All projects")

    if projects:
        import pandas as pd
        df = pd.DataFrame(
            [
                {
                    "ID": p.id,
                    "Name": p.name,
                    "Location": p.location,
                    "Business Unit": p.business_unit,
                    "Sub-division": p.sub_division,
                    "Start": p.start_date,
                    "End": p.end_date,
                    "Description": p.description,
                }
                for p in projects
            ]
        )
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No projects created yet.")

