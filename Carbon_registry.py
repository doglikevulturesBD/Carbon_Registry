import streamlit as st
from utils.load_css import load_css

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
APP_TITLE = "Carbon Registry"
APP_ICON = "🌍"
APP_VERSION = "v0.1 (beta)"

NAV_ITEMS = [
    {
        "col": 0,
        "card_title": "⚖️ Carbon Registry",
        "desc": "Create projects, record activities, and track carbon reductions.",
        "button": "Open Carbon Registry",
        "page": "pages/1_Registry.py",
        "badge": "Core",
    },
    {
        "col": 1,
        "card_title": "📊 Scope 1 / 2 / 3 Calculator",
        "desc": "Compute emissions baseline or project values across all scopes.",
        "button": "Open Scope Calculator",
        "page": "2_Scope_Calculator.py",
        "badge": "Core",
    },
    {
        "col": 2,
        "card_title": "📘 Methodology Tools",
        "desc": "VERA-aligned calculators (VM0038, VMR0007, EV, hydrogen).",
        "button": "Open Methodology Calculators",
        "page": "3_Methodologies.py",
        "badge": "Beta",
    },
]

def safe_switch_page(page_path: str):
    """Switch pages with a friendly error if the target can't be opened."""
    try:
        st.switch_page(page_path)
    except Exception as e:
        st.error("Navigation failed. The target page may have been renamed or moved.")
        st.caption(f"Details: {e}")

# ---------------------------------------------------------
# PAGE SETUP (must be first Streamlit call)
# ---------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
load_css()

# ---------------------------------------------------------
# SIDEBAR (optional, does not replace cards)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.caption(APP_VERSION)
    st.divider()

    # Quick nav
    for i, item in enumerate(NAV_ITEMS):
        if st.button(item["card_title"], key=f"side_nav_{i}", use_container_width=True):
            safe_switch_page(item["page"])

    st.divider()
    with st.expander("What’s new"):
        st.write("- Landing page refactor (reliability + scalability)")
        st.write("- Sidebar navigation added")
        st.write("- Footer + version tag")

# ---------------------------------------------------------
# HOME HERO
# ---------------------------------------------------------
st.markdown(
    f"""
    <div style='padding: 30px;'>
    <h1 style='color:#86ffcf; text-shadow:0 0 10px #39ff9f;'>
    🌱 Carbon Registry & MRV Platform
    </h1>

    <p style='font-size:18px; color:#b3ffdd;'>
    A unified space for carbon accounting, MRV, methodologies and scope calculators.
    <br/>
    <span style='opacity:0.85;'>Version: {APP_VERSION}</span>
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Optional quick actions row (placeholders for now)
qa1, qa2, qa3, qa4 = st.columns(4)
with qa1:
    st.button("📌 Roadmap", key="qa_roadmap", use_container_width=True)
with qa2:
    st.button("📄 Docs", key="qa_docs", use_container_width=True)
with qa3:
    st.button("🐞 Report Bug", key="qa_bug", use_container_width=True)
with qa4:
    st.button("✨ What’s New", key="qa_whatsnew", use_container_width=True)

st.write("")  # spacing

# ---------------------------------------------------------
# NAVIGATION CARDS
# ---------------------------------------------------------
cols = st.columns(3)

for idx, item in enumerate(NAV_ITEMS):
    with cols[item["col"]]:
        st.markdown(
            f"""
            <div class='glass-box'>
            <div style='display:flex; justify-content:space-between; align-items:center; gap:12px;'>
            <h3 style='margin:0;'>{item["card_title"]}</h3>
            <span style='font-size:12px; opacity:0.8;'>{item["badge"]}</span>
            </div>
            <p>{item["desc"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(item["button"], key=f"nav_btn_{idx}", use_container_width=True):
            safe_switch_page(item["page"])

# ---------------------------------------------------------
# TRUST / DISCLAIMER (short, professional)
# ---------------------------------------------------------
st.divider()
st.caption(
    "Disclaimer: This is a beta tool for analysis and learning. "
    "Always validate inputs and results against the applicable standard/methodology and verified data sources."
)
st.caption(f"{APP_TITLE} • {APP_VERSION}")


