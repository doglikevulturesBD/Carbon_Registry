# registry/audit_logs.py
import pandas as pd
import streamlit as st

from .database import SessionLocal
from . import crud


def render_audit_logs():
    st.markdown("### 📝 Audit logs")

    with SessionLocal() as session:
        logs = crud.get_all_logs(session)

    if not logs:
        st.info("No audit logs recorded yet.")
        return

    df = pd.DataFrame(
        [
            {
                "Timestamp": log.timestamp,
                "Project ID": log.project_id,
                "Action": log.action,
                "Note": log.note,
            }
            for log in logs
        ]
    )

    st.dataframe(df, use_container_width=True)

