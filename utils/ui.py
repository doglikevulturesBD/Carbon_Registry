# utils/ui.py
import streamlit as st
from utils.load_css import load_css

APP_TITLE = "Carbon Registry"
APP_ICON = "🌍"
APP_VERSION = "v1.0 (foundation beta)"
APP_TAGLINE = "Boundaries → Assumptions → Calculators → Evidence"

NAV_ITEMS = [
    {
        "card_title": "⚖️ Carbon Registry",
        "desc": "Create projects, log activities, and capture boundaries + assumptions.",
        "button": "Open Carbon Registry",
        "page": "pages/1_Registry.py",
        "badge": "Core",
    },
    {
        "card_title": "📊 Scope 1 / 2 / 3 Calculator",
        "desc": "Baseline estimates across scopes with transparent factors + assumptions.",
        "button": "Open Scope Calculator",
        "page": "pages/2_Scope_Calculator.py",
        "badge": "Core",
    },
    {
        "card_title": "📘 Methodology Tools",
        "desc": "Verra-aligned worked examples (demos, not audit outputs): VM0038, VMR0007, EV, hydrogen.",
        "button": "Open Methodology Examples",
        "page": "pages/3_Methodologies.py",
        "badge": "Beta",
    },
]

def safe_switch_page(page_path: str):
    try:
        st.switch_page(page_path)
    except Exception as e:
        st.error("Navigation failed: page not found or renamed.")
        st.caption(f"Expected file: {page_path}")
        st.caption(f"Details: {e}")

def render_sidebar():
    with st.sidebar:
        st.markdown(f"## {APP_ICON} {APP_TITLE}")
        st.caption(APP_VERSION)
        st.divider()

        st.markdown("### Navigation")
        for i, item in enumerate(NAV_ITEMS):
            if st.button(item["card_title"], key=f"side_nav_{i}", use_container_width=True):
                safe_switch_page(item["page"])

        st.divider()

        with st.expander("How to use (fast)"):
            st.write("1) Define **boundaries** for a project.")
            st.write("2) Record **assumptions + data sources**.")
            st.write("3) Run **calculator demos** with transparent factors.")
            st.write("4) Export notes/results for review.")

        with st.expander("What’s new"):
            st.write("- Home hub repositioned as foundation tool (not audit-grade MRV)")
            st.write("- Quick actions enabled (roadmap/docs/bug/what’s new)")
            st.write("- Clearer module descriptions + disclaimers")

def render_hero(title: str, subtitle_html: str, tagline: str = APP_TAGLINE):
    st.markdown(
        f"""
        <div class="glass-box" style="padding: 26px 26px 14px 26px; margin-bottom: 16px;">
        <h1 style="margin:0; color:#86ffcf; text-shadow:0 0 10px #39ff9f;">
            {title}
        </h1>
        <p style="font-size:18px; margin-top:10px; color:#b3ffdd;">
            {subtitle_html}
        </p>
        <p style="font-size:14px; margin-top:10px; color:#b3ffdd; opacity:0.85;">
            Suggested flow: <b>{tagline}</b>
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def setup_page(page_title: str, page_icon: str = APP_ICON, layout: str = "wide"):
    """
    Call this at the TOP of every page.
    - page config
    - loads the same CSS
    - renders the same sidebar
    """
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
    load_css()
    render_sidebar()
