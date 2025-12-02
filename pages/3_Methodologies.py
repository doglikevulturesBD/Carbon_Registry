# pages/3_📘_Methodologies.py
import streamlit as st
from methodologies.vm0038_ev import vm0038_ev
# later: from methodologies.vmR0007_xx import vmr0007_..., etc.

def main():
    st.title("📘 Methodology Calculators")

    choice = st.selectbox(
        "Select methodology:",
        [
            "VM0038 – EV Charging",
            # "VMR0007 – XXX", etc.
        ]
    )

    if choice.startswith("VM0038"):
        vm0038_ev()

if __name__ == "__main__":
    main()

