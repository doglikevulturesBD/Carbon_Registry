# pages/3_📘_Methodologies.py
# ------------------------------------------------------------
# Carbon Registry • Methodology Launcher (Page 3)
#
# Purpose:
# - A router to methodology-specific calculators under /methodologies
# - Keeps methodology logic modular (clean architecture)
#
# Launch-safe features:
# ✅ Metadata map (standard, status, summary)
# ✅ Safe dynamic imports (won't crash the whole app if one module breaks)
# ✅ Graceful error display + optional debug details
# ✅ Scales to many methodologies
#
# Notes:
# - Demo/workbench for learning + discussion.
# - Not an issuance platform / not a registry-of-record.
# ------------------------------------------------------------

import streamlit as st
import importlib
from typing import Callable, Dict, Any, Optional, Tuple


# ---------------------------
# App flags
# ---------------------------
DEBUG = bool(st.secrets.get("DEBUG", False)) if hasattr(st, "secrets") else False


# ---------------------------
# Method registry (add your 4 Verra methodologies here)
# Each entry points to (module_path, function_name)
# ---------------------------
METHODOLOGIES: Dict[str, Dict[str, Any]] = {
    "VM0038 – EV Charging": {
        "id": "VM0038",
        "standard": "Verra VCS",
        "status": "Demo / Beta",
        "scope": "Transport / Energy",
        "module": "methodologies.vm0038_ev",
        "entrypoint": "vm0038_ev",
        "summary": (
            "EV charging emission reductions: baseline ICE fuel vs project electricity, "
            "optional WTT, renewable fraction, grid decarb, annual series, uncertainty, "
            "and save-to-registry hook."
        ),
        "primary_outputs": ["BEy", "PEy", "ERy", "Total ER", "Annual CSV"],
        "launch_ready": True,
    },

    # ---- Replace these placeholders with your real modules ----
    # Example names below; update module + entrypoint to match your folder/files.

    "VMR0007 – Plastics / Waste (your module name)": {
        "id": "VMR0007",
        "standard": "Verra VCS",
        "status": "In progress / Beta",
        "scope": "Waste / Circular Economy",
        "module": "methodologies.vmr0007_plastics",   # <-- change
        "entrypoint": "vmr0007_plastics",            # <-- change
        "summary": "Plastic/waste methodology module (worked example).",
        "primary_outputs": ["Baseline", "Project", "Leakage (if any)", "ER", "QA/QC notes"],
        "launch_ready": False,  # flip to True when implemented
    },

    "VM00XX – Hydrogen (your module name)": {
        "id": "VM00XX",
        "standard": "Verra VCS",
        "status": "Planned",
        "scope": "Hydrogen / Energy",
        "module": "methodologies.vm00xx_hydrogen",    # <-- change
        "entrypoint": "vm00xx_hydrogen",              # <-- change
        "summary": "Hydrogen methodology module (planned).",
        "primary_outputs": [],
        "launch_ready": False,
    },

    "VM00YY – Another Verra Methodology (your module name)": {
        "id": "VM00YY",
        "standard": "Verra VCS",
        "status": "Planned",
        "scope": "TBD",
        "module": "methodologies.vm00yy_other",       # <-- change
        "entrypoint": "vm00yy_other",                 # <-- change
        "summary": "Fourth Verra methodology module (planned).",
        "primary_outputs": [],
        "launch_ready": False,
    },
}


# ---------------------------
# Helpers
# ---------------------------
def safe_load_entrypoint(module_path: str, func_name: str) -> Tuple[Optional[Callable], Optional[Exception]]:
    """
    Dynamically import module and return the callable function.
    Returns (func, error).
    """
    try:
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name, None)
        if not callable(func):
            raise AttributeError(f"Entrypoint '{func_name}' not found or not callable in '{module_path}'.")
        return func, None
    except Exception as e:
        return None, e


def render_header() -> None:
    st.title("📘 Methodology Calculators")
    st.caption(
        "Methodology-specific calculators are implemented as independent modules. "
        "Each module demonstrates how boundaries, assumptions, and equations differ by methodology. "
        "This is a workbench for learning and discussion — not an issuance platform."
    )
    st.info(
        "Rigor note: Modules may use simplified assumptions for demonstration. "
        "Verification-grade MRV requires evidence chains, governance, and standard-specific validation."
    )
    st.divider()


def render_selector() -> str:
    options = list(METHODOLOGIES.keys())
    # Prefer VM0038 first, otherwise alphabetic
    if "VM0038 – EV Charging" in options:
        default = options.index("VM0038 – EV Charging")
    else:
        options = sorted(options)
        default = 0

    return st.selectbox("Select methodology:", options=options, index=default)


def render_metadata(meta: Dict[str, Any]) -> None:
    left, right = st.columns([2, 1])

    with left:
        st.markdown(f"### {meta.get('id', '—')} — {meta.get('standard', '—')}")
        st.markdown(f"**Status:** {meta.get('status', '—')}")
        st.markdown(f"**Scope:** {meta.get('scope', '—')}")
        st.write(meta.get("summary", ""))

    with right:
        st.markdown("**Primary outputs**")
        outs = meta.get("primary_outputs") or []
        if outs:
            for o in outs:
                st.markdown(f"- {o}")
        else:
            st.caption("—")

        ready = meta.get("launch_ready", False)
        if ready:
            st.success("Launch-ready module")
        else:
            st.warning("Not launch-ready yet")

    st.divider()


def render_run(meta: Dict[str, Any]) -> None:
    module_path = meta.get("module")
    func_name = meta.get("entrypoint")

    if not module_path or not func_name:
        st.error("Method entry is missing module/entrypoint configuration.")
        return

    func, err = safe_load_entrypoint(module_path, func_name)

    if err is not None or func is None:
        st.error("This methodology module could not be loaded.")
        st.caption(f"Module: `{module_path}` • Entrypoint: `{func_name}`")
        if DEBUG:
            st.exception(err)
        else:
            st.caption("Enable DEBUG in Streamlit secrets to view detailed error traces.")
        return

    # Run the methodology module
    try:
        func()
    except Exception as e:
        st.error("The methodology module crashed while running.")
        if DEBUG:
            st.exception(e)
        else:
            st.caption("Enable DEBUG in Streamlit secrets to view detailed error traces.")


# ---------------------------
# Page main
# ---------------------------
def main() -> None:
    render_header()
    choice = render_selector()
    meta = METHODOLOGIES[choice]
    render_metadata(meta)
    render_run(meta)

    st.divider()
    st.caption(
        "Developer note: Keep each methodology in its own file under /methodologies and expose a single entry function. "
        "This keeps your registry base clean, auditable, and scalable."
    )


if __name__ == "__main__":
    main()
