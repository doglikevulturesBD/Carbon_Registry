import streamlit as st
from utils.load_css import load_css

st.set_page_config(
    page_title="Carbon Registry",
    page_icon="🌍",
    layout="wide"
)

load_css()

# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
st.markdown("""
<div style='padding: 30px;'>
    <h1 style='color:#86ffcf; text-shadow:0 0 10px #39ff9f;'>
        🌱 Carbon Registry & MRV Platform
    </h1>

    <p style='font-size:18px; color:#b3ffdd;'>
        A unified space for carbon accounting, MRV, methodologies and scope calculators.
        This beta version includes three main modules:
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# NAVIGATION CARDS
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='glass-box'>
        <h3>⚖️ Carbon Registry</h3>
        <p>Create projects, record activities, and track carbon reductions.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Carbon Registry"):
        st.switch_page("pages/1_⚖️_Carbon_Registry.py")

with col2:
    st.markdown("""
    <div class='glass-box'>
        <h3>📊 Scope 1 / 2 / 3 Calculator</h3>
        <p>Compute emissions baseline or project values across all scopes.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Scope Calculator"):
        st.switch_page("pages/2_📊_Scope_Calculator.py")

with col3:
    st.markdown("""
    <div class='glass-box'>
        <h3>📘 Methodology Tools</h3>
        <p>VERA-aligned calculators based on VM0038, VMR0007, EV, and hydrogen methodologies.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Methodology Calculators"):
        st.switch_page("pages/3_📘_Methodologies.py")

