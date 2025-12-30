# pages/1_Registry.py
import streamlit as st
from typing import Optional, Callable, Dict

from registry.dashboard import render_dashboard
from registry.projects import render_projects
from registry.emissions import render_emissions
from registry.audit_logs import render_audit_logs
from registry.export_tools import render_export_tools

# -----------------------------
# Debug toggle (safe defaults)
# -----------------------------
def _get_debug_flag() -> bool:
    # Supports secrets.toml: DEBUG = true
    try:
        return bool(st.secrets.get("DEBUG", False))
    except Exception:
        return False

DEBUG = _get_debug_flag()

# -----------------------------
# Navigation configuration
# -----------------------------
SECTIONS: Dict[str, Dict[str, object]] = {
    "dashboard": {"label": "📊 Dashboard", "render": render_dashboard},
    "projects": {"label": "📂 Projects", "render": render_projects},
    "emissions": {"label": "🔥 Emissions", "render": render_emissions},
    "audit": {"label": "📝 Audit Logs", "render": render_audit_logs},
    "export": {"label": "⬇️ Export", "render": render_export_tools},
}

# Explicit order (don’t rely on dict order)
SECTION_ORDER = ["dashboard", "projects", "emissions", "audit", "export"]
DEFAULT_SECTION_KEY = "dashboard"


# -----------------------------
# Query param helpers (robust across Streamlit versions)
# -----------------------------
def _normalize_query_value(val) -> Optional[str]:
    """Normalize query param values that may be str | list[str] | None."""
    if val is None:
        return None
    if isinstance(val, list):
        return val[0] if val else None
    return str(val)

def _get_query_tab() -> Optional[str]:
    """Read 'tab' from query params (supports new + legacy APIs)."""
    # New API
    try:
        qp = st.query_params
        tab = _normalize_query_value(qp.get("tab", None))
        return tab
    except Exception:
        pass

    # Legacy API fallback
    try:
        qp = st.experimental_get_query_params()
        tab = _normalize_query_value(qp.get("tab", None))
        return tab
    except Exception:
        return None

def _set_query_tab(tab_key: str) -> None:
    """Write 'tab' to query params (supports new + legacy APIs)."""
    # New API
    try:
        st.query_params["tab"] = tab_key
        return
    except Exception:
        pass

    # Legacy fallback
    try:
        st.experimental_set_query_params(tab=tab_key)
    except Exception:
        pass


# -----------------------------
# Error boundary
# -----------------------------
def error_boundary(render_fn: Callable[[], None], section_key: str) -> None:
    """Prevent one broken module from taking down the whole page."""
    try:
        render_fn()
    except Exception as e:
        st.error(f"Something went wrong while loading **{SECTIONS[section_key]['label']}**.")
        if DEBUG:
            st.exception(e)
        else:
            st.caption("Enable DEBUG in secrets to view full trace.")


# -----------------------------
# Main UI
# -----------------------------
def main():
    st.title("⚖️ Carbon Registry")
    st.caption("Projects, emissions, audit logs, and exports – all in one MRV workbench.")

    # Init canonical state from URL (only once)
    if "registry_section" not in st.session_state:
        qp_tab = _get_query_tab()
        st.session_state.registry_section = qp_tab if qp_tab in SECTIONS else DEFAULT_SECTION_KEY

    # Layout
    col_nav, col_main = st.columns([1, 3], gap="large")

    with col_nav:
        st.markdown("### Sections")

        labels = [SECTIONS[k]["label"] for k in SECTION_ORDER]
        keys = SECTION_ORDER

        # Preselect based on canonical state
        current_key = st.session_state.registry_section
        current_index = keys.index(current_key) if current_key in keys else 0

        selected_index = st.radio(
            label="",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
            index=current_index,
            label_visibility="collapsed",
            key="registry_section_index",
        )

        selected_key = keys[selected_index]

        # Persist selection + update URL only when changed
        if selected_key != st.session_state.registry_section:
            st.session_state.registry_section = selected_key
            _set_query_tab(selected_key)

        st.divider()

        # Quality-of-life: share link to current section
        try:
            # Streamlit provides a helper in newer versions
            st.link_button(
                "🔗 Share link to this section",
                f"?tab={st.session_state.registry_section}",
                use_container_width=True,
            )
        except Exception:
            st.caption(f"Share link: ?tab={st.session_state.registry_section}")

        with st.expander("Help"):
            st.write("Use the left menu to navigate across modules.")
            st.write("Tip: You can deep-link to a section using `?tab=dashboard`, `?tab=emissions`, etc.")

    with col_main:
        section_key = st.session_state.registry_section
        st.subheader(SECTIONS[section_key]["label"])

        # Render selected module safely
        error_boundary(SECTIONS[section_key]["render"], section_key)


if __name__ == "__main__":
    main()
