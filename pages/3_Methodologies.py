# pages/3_📘_Methodologies.py
# ------------------------------------------------------------
# Carbon Registry • Methodology Launcher (Page 3)
#
# Purpose:
# - Acts as a router to methodology-specific calculators living in /methodologies
# - Keeps methodology logic modular and isolated (clean architecture)
#
# Notes:
# - This is a demo/workbench for learning + discussion.
# - It does NOT issue credits and is NOT a registry-of-record.
# - Each methodology module should expose a callable function (e.g., vm0038_ev()).
# ------------------------------------------------------------

import streamlit as st

# Import your methodology entrypoints (one per module).
# Add more as you build them.
from methodologies.vm0038_ev import vm0038_ev
# later:
# from methodologies.vmr0007_plastics import vmr0007_plastics
# from methodologies.vmXXXX_hydrogen import vmXXXX_hydrogen


# Registry-style metadata map (scales cleanly)
METHODOLOGIES = {
    "VM0038 – EV Charging": {
        "id": "VM0038",
        "standard": "Verra VCS (demonstration module)",
        "status": "Demo / Beta",
        "coverage_note": (
            "Worked-example calculator to demonstrate how a methodology module plugs into the MRV base layer. "
            "Not for issuance or compliance."
        ),
        "primary_outputs": ["tCO₂e baseline", "tCO₂e project", "tCO₂e reduction"],
        "func": vm0038_ev,
    },

    # Example placeholders (uncomment when you implement)
    # "VMR0007 – Plastic Waste (placeholder)": {
    #     "id": "VMR0007",
    #     "standard": "Verra VCS (placeholder)",
    #     "status": "Planned",
    #     "coverage_note": "Placeholder entry—module not implemented yet.",
    #     "primary_outputs": [],
    #     "func": None,
    # },
}


def render_header() -> None:
    st.title("📘 Methodology Calculators")
    st.caption(
        "Methodology-specific calculators are implemented as independent modules. "
        "Each module demonstrates how boundaries, assumptions, and equations differ by methodology. "
        "This is a workbench for learning and discussion — not an issuance platform."
    )
    st.divider()


def render_selector() -> str:
    choice = st.selectbox(
        "Select methodology:",
        options=list(METHODOLOGIES.keys()),
        index=0,
    )
    return choice


def render_metadata(meta: dict) -> None:
    # Lightweight "card" using simple Streamlit elements (CSS optional)
    left, right = st.columns([2, 1])

    with left:
        st.markdown(f"### {meta.get('id', '')}")
        st.markdown(f"**Standard:** {meta.get('standard', '—')}")
        st.markdown(f"**Status:** {meta.get('status', '—')}")
        st.write(meta.get("coverage_note", ""))

    with right:
        st.markdown("**Primary outputs:**")
        outs = meta.get("primary_outputs") or []
        if outs:
            for o in outs:
                st.markdown(f"- {o}")
        else:
            st.caption("—")

    st.info(
        "Rigor note: These modules demonstrate structure and logic. "
        "Verification-grade MRV requires evidence chains, governance, and standard-specific validation."
    )
    st.divider()


def render_run(meta: dict) -> None:
    func = meta.get("func")
    if callable(func):
        # Run the module
        func()
    else:
        st.warning("This module is not implemented yet.")


def main() -> None:
    render_header()

    choice = render_selector()
    meta = METHODOLOGIES[choice]

    render_metadata(meta)
    render_run(meta)

    st.divider()
    st.caption(
        "Tip: Keep each methodology in its own file under /methodologies and expose a single entry function. "
        "This keeps your registry base clean and your methodology logic auditable."
    )


if __name__ == "__main__":
    main()
