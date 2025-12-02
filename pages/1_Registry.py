# pages/1_⚖️_Carbon_Registry.py
import streamlit as st

from registry.dashboard import render_dashboard
from registry.projects import render_projects
from registry.emissions import render_emissions
from registry.audit_logs import render_audit_logs
from registry.export_tools import render_export_tools

# If your main app already calls load_css(), you don't need it here.
# If not, you can uncomment:
# from utils.load_css import load_css

def main():
    st.title("⚖️ Carbon Registry")

    st.caption(
        "Projects, emissions, audit logs, and exports – all in one MRV workbench."
    )

    col_nav, col_main = st.columns([1, 3])

    with col_nav:
        section = st.radio(
            "Sections",
            [
                "📊 Dashboard",
                "📂 Projects",
                "🔥 Emissions",
                "📝 Audit Logs",
                "⬇️ Export",
            ],
        )

    with col_main:
        if section.startswith("📊"):
            render_dashboard()
        elif section.startswith("📂"):
            render_projects()
        elif section.startswith("🔥"):
            render_emissions()
        elif section.startswith("📝"):
            render_audit_logs()
        elif section.startswith("⬇️"):
            render_export_tools()

if __name__ == "__main__":
    main()

