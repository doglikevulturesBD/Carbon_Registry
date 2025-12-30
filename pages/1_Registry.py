# pages/1_⚖️_Carbon_Registry.py
import streamlit as st

from registry.dashboard import render_dashboard
from registry.projects import render_projects
from registry.emissions import render_emissions
from registry.audit_logs import render_audit_logs
from registry.export_tools import render_export_tools

# -----------------------------
# Configurable navigation map
# -----------------------------
SECTIONS = {
    "dashboard": {"label": "📊 Dashboard", "render": render_dashboard},
    "projects": {"label": "📂 Projects", "render": render_projects},
    "emissions": {"label": "🔥 Emissions", "render": render_emissions},
    "audit": {"label": "📝 Audit Logs", "render": render_audit_logs},
    "export": {"label": "⬇️ Export", "render": render_export_tools},
}

DEFAULT_SECTION_KEY = "dashboard"


def _get_query_tab() -> str | None:
    """Read 'tab' from query params (for deep links)."""
    try:
        qp = st.query_params  # Streamlit 1.30+ (works in newer versions)
        tab = qp.get("tab", None)
        return tab
    except Exception:
        return None


def _set_query_tab(tab_key: str) -> None:
    """Write 'tab' to query params (so the selected tab is shareable)."""
    try:
        st.query_params["tab"] = tab_key
    except Exception:
        # Older Streamlit versions may not support this API
        pass


def error_boundary(render_fn, section_key: str):
    """Prevent one broken module from taking down the whole page."""
    try:
        render_fn()
    except Exception as e:
        st.error(f"Something went wrong while loading **{SECTIONS[section_key]['label']}**.")
        # In early dev, you want the full stack trace:
        st.exception(e)


def main():
    st.title("⚖️ Carbon Registry")
    st.caption("Projects, emissions, audit logs, and exports – all in one MRV workbench.")

    # -----------------------------
    # State init + deep-link support
    # -----------------------------
    if "registry_section" not in st.session_state:
        # If URL has ?tab=..., use it; else default.
        qp_tab = _get_query_tab()
        st.session_state.registry_section = qp_tab if qp_tab in SECTIONS else DEFAULT_SECTION_KEY

    # Layout
    col_nav, col_main = st.columns([1, 3], gap="large")

    with col_nav:
        st.markdown("### Sections")

        labels = [SECTIONS[k]["label"] for k in SECTIONS]
        keys = list(SECTIONS.keys())

        # Pre-select current state
        current_key = st.session_state.registry_section
        current_index = keys.index(current_key) if current_key in keys else 0

        selected_label = st.radio(
            label="",
            options=labels,
            index=current_index,
            key="registry_section_radio",
        )

        # Map label back to key
        selected_key = keys[labels.index(selected_label)]

        # Persist selection + update URL
        if selected_key != st.session_state.registry_section:
            st.session_state.registry_section = selected_key
            _set_query_tab(selected_key)

        st.divider()

        # Optional: small quality-of-life extras (no CSS needed)
        with st.expander("Help"):
            st.write("Use the left menu to navigate across modules.")
            st.write("Tip: You can share a link to a section using the URL query parameter `?tab=...`.")

    with col_main:
        section_key = st.session_state.registry_section
        st.subheader(SECTIONS[section_key]["label"])

        # Render selected module safely
        error_boundary(SECTIONS[section_key]["render"], section_key)


if __name__ == "__main__":
    main()

