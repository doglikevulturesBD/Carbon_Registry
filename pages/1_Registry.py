import streamlit as st
import sqlite3
import pandas as pd
import json
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# ------------------------------------------------------------
# PAGE CONFIG (do NOT call set_page_config here if already in launcher)
# ------------------------------------------------------------
st.title("⚖️ Carbon Registry")
st.caption("Project registry + credits + sales + audit logs (beta MRV workbench)")

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
DB_PATH = Path("data/carbon_registry.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

TAB_KEYS = ["projects", "credits", "audit", "export"]
TAB_LABELS = ["📂 Projects", "💳 Credits & Sales", "📝 Audit Trail", "⬇️ Export"]

# Optional: in Streamlit Cloud you can set secrets:
# [secrets]
# DEBUG = true
DEBUG = bool(st.secrets.get("DEBUG", False)) if hasattr(st, "secrets") else False


# ------------------------------------------------------------
# DB LAYER
# ------------------------------------------------------------
@st.cache_resource
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_exec(query: str, params: Tuple = ()) -> None:
    conn = get_conn()
    conn.execute(query, params)
    conn.commit()


def db_query(query: str, params: Tuple = ()) -> pd.DataFrame:
    conn = get_conn()
    cur = conn.execute(query, params)
    rows = cur.fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def ensure_schema() -> None:
    db_exec("""
    CREATE TABLE IF NOT EXISTS projects (
        project_id TEXT PRIMARY KEY,
        project_code TEXT UNIQUE,
        project_name TEXT NOT NULL,
        owner_org TEXT,
        country TEXT,
        region TEXT,
        sector TEXT,
        methodology TEXT,
        standard TEXT,
        baseline_year INTEGER,
        start_date TEXT,
        end_date TEXT,
        status TEXT DEFAULT 'Active',
        description TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)

    db_exec("""
    CREATE TABLE IF NOT EXISTS credits (
        credit_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        vintage_year INTEGER NOT NULL,
        credits_issued REAL DEFAULT 0,
        issuance_date TEXT,
        registry_program TEXT,
        serial_range TEXT,
        notes TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(project_id) REFERENCES projects(project_id)
    );
    """)

    db_exec("""
    CREATE TABLE IF NOT EXISTS sales (
        sale_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        credit_id TEXT,
        sale_date TEXT NOT NULL,
        buyer TEXT,
        credits_sold REAL NOT NULL,
        price_per_credit REAL,
        currency TEXT DEFAULT 'USD',
        contract_ref TEXT,
        notes TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(project_id) REFERENCES projects(project_id),
        FOREIGN KEY(credit_id) REFERENCES credits(credit_id)
    );
    """)

    db_exec("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        audit_id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        actor TEXT,
        action TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        entity_id TEXT,
        project_id TEXT,
        before_json TEXT,
        after_json TEXT,
        meta_json TEXT
    );
    """)


ensure_schema()


# ------------------------------------------------------------
# AUDIT
# ------------------------------------------------------------
def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def audit_log(
    action: str,
    entity_type: str,
    entity_id: Optional[str] = None,
    project_id: Optional[str] = None,
    before: Optional[Dict[str, Any]] = None,
    after: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None
) -> None:
    actor = st.session_state.get("actor_name", "unknown")
    db_exec(
        """
        INSERT INTO audit_logs (audit_id, timestamp, actor, action, entity_type, entity_id, project_id, before_json, after_json, meta_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            now_iso(),
            actor,
            action,
            entity_type,
            entity_id,
            project_id,
            json.dumps(before) if before else None,
            json.dumps(after) if after else None,
            json.dumps(meta) if meta else None,
        ),
    )


# ------------------------------------------------------------
# QUERY PARAMS (deep-link tabs)
# ------------------------------------------------------------
def get_qp(name: str) -> Optional[str]:
    # New API
    try:
        v = st.query_params.get(name)
        if isinstance(v, list):
            return v[0] if v else None
        return v
    except Exception:
        pass
    # Legacy fallback
    try:
        qp = st.experimental_get_query_params()
        v = qp.get(name)
        if isinstance(v, list):
            return v[0] if v else None
        return v
    except Exception:
        return None


def set_qp(**kwargs) -> None:
    try:
        for k, v in kwargs.items():
            st.query_params[k] = v
    except Exception:
        try:
            st.experimental_set_query_params(**kwargs)
        except Exception:
            pass


# ------------------------------------------------------------
# UI: GLOBAL CONTEXT / ACTOR
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧑‍💼 Session")
    st.text_input("Actor (for audit logs)", key="actor_name", value=st.session_state.get("actor_name", "Brandon"))
    st.divider()

    st.markdown("### 📌 Active Project")
    # Loaded later (after project list) – placeholder here
    st.caption("Select a project inside the Projects tab.")

    st.divider()
    st.markdown("### ✅ Registry Health")
    st.write(f"DB: `{DB_PATH.as_posix()}`")
    st.write("Schema: OK")


# ------------------------------------------------------------
# DATA HELPERS
# ------------------------------------------------------------
@st.cache_data(ttl=10)
def list_projects(active_only: bool = False) -> pd.DataFrame:
    q = "SELECT * FROM projects"
    if active_only:
        q += " WHERE status = 'Active'"
    q += " ORDER BY updated_at DESC"
    return db_query(q)


def get_project(project_id: str) -> Optional[Dict[str, Any]]:
    df = db_query("SELECT * FROM projects WHERE project_id = ?", (project_id,))
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def set_active_project(project_id: Optional[str]) -> None:
    st.session_state.active_project_id = project_id


def active_project() -> Optional[Dict[str, Any]]:
    pid = st.session_state.get("active_project_id")
    return get_project(pid) if pid else None


# ------------------------------------------------------------
# TOP NAV TABS
# ------------------------------------------------------------
initial_tab = get_qp("tab")
if "tab_key" not in st.session_state:
    st.session_state.tab_key = initial_tab if initial_tab in TAB_KEYS else "projects"

tab_index = TAB_KEYS.index(st.session_state.tab_key)
tabs = st.tabs(TAB_LABELS)

def goto_tab(key: str) -> None:
    st.session_state.tab_key = key
    set_qp(tab=key)


# ------------------------------------------------------------
# TAB 1: PROJECTS (CRUD)
# ------------------------------------------------------------
with tabs[0]:
    st.subheader("📂 Projects")

    # Quick list
    cols = st.columns([2, 1, 1])
    with cols[0]:
        show_active_only = st.checkbox("Show active only", value=False)
    with cols[1]:
        if st.button("↻ Refresh list"):
            list_projects.clear()
            st.rerun()
    with cols[2]:
        if st.button("Go to Credits & Sales ➜"):
            goto_tab("credits")
            st.rerun()

    dfp = list_projects(active_only=show_active_only)

    # Project selector
    if dfp.empty:
        st.info("No projects yet. Create your first project below.")
        selected_project_id = None
    else:
        # Ensure selection persists
        current = st.session_state.get("active_project_id")
        options = dfp["project_id"].tolist()
        default_idx = options.index(current) if current in options else 0

        selected_project_id = st.selectbox(
            "Select a project (sets Active Project context)",
            options=options,
            index=default_idx,
            format_func=lambda pid: f"{dfp.loc[dfp.project_id==pid, 'project_code'].values[0]} — {dfp.loc[dfp.project_id==pid, 'project_name'].values[0]}",
        )
        set_active_project(selected_project_id)

    st.divider()

    st.markdown("### ➕ Create new project")
    with st.form("create_project_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            project_code = st.text_input("Project code (unique)", placeholder="CR-0001")
            project_name = st.text_input("Project name*", placeholder="Alicedale Solar + Storage MRV")
            owner_org = st.text_input("Owner/Developer org", placeholder="Davoren Insights / Partner")
        with c2:
            country = st.text_input("Country", value="South Africa")
            region = st.text_input("Region / Province", placeholder="Eastern Cape")
            sector = st.selectbox("Sector", ["Energy", "Transport", "Waste", "AFOLU", "Industry", "Other"])
        with c3:
            standard = st.selectbox("Standard", ["VCS (Verra)", "Gold Standard", "ISO 14064", "Other"])
            methodology = st.text_input("Methodology", placeholder="VM0038 / VMR0007 / ...")
            baseline_year = st.number_input("Baseline year", min_value=1900, max_value=2100, value=2024)

        d1, d2 = st.columns(2)
        with d1:
            start_date = st.date_input("Start date", value=date.today())
        with d2:
            end_date = st.date_input("End date (optional)", value=None)

        description = st.text_area("Description / notes", height=80)

        submitted = st.form_submit_button("Create project", use_container_width=True)

        if submitted:
            if not project_name.strip() or not project_code.strip():
                st.error("Project code and project name are required.")
            else:
                pid = str(uuid.uuid4())
                ts = now_iso()
                db_exec(
                    """
                    INSERT INTO projects (
                        project_id, project_code, project_name, owner_org, country, region, sector,
                        methodology, standard, baseline_year, start_date, end_date, status, description,
                        created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pid,
                        project_code.strip(),
                        project_name.strip(),
                        owner_org.strip() or None,
                        country.strip() or None,
                        region.strip() or None,
                        sector,
                        methodology.strip() or None,
                        standard,
                        int(baseline_year),
                        start_date.isoformat() if start_date else None,
                        end_date.isoformat() if end_date else None,
                        "Active",
                        description.strip() or None,
                        ts,
                        ts,
                    ),
                )
                audit_log(
                    action="CREATE",
                    entity_type="project",
                    entity_id=pid,
                    project_id=pid,
                    after={"project_code": project_code, "project_name": project_name},
                )
                list_projects.clear()
                set_active_project(pid)
                st.success("Project created.")
                st.rerun()

    # Edit / Archive current project
    if selected_project_id:
        proj = active_project()
        if proj:
            st.markdown("### ✏️ Edit active project")
            with st.form("edit_project_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    e_project_code = st.text_input("Project code", value=proj.get("project_code") or "")
                    e_project_name = st.text_input("Project name", value=proj.get("project_name") or "")
                    e_owner_org = st.text_input("Owner/Developer org", value=proj.get("owner_org") or "")
                with c2:
                    e_country = st.text_input("Country", value=proj.get("country") or "")
                    e_region = st.text_input("Region / Province", value=proj.get("region") or "")
                    e_sector = st.selectbox("Sector", ["Energy", "Transport", "Waste", "AFOLU", "Industry", "Other"],
                                            index=["Energy","Transport","Waste","AFOLU","Industry","Other"].index(proj.get("sector") or "Energy"))
                with c3:
                    e_standard = st.selectbox("Standard", ["VCS (Verra)", "Gold Standard", "ISO 14064", "Other"],
                                              index=["VCS (Verra)","Gold Standard","ISO 14064","Other"].index(proj.get("standard") or "VCS (Verra)"))
                    e_methodology = st.text_input("Methodology", value=proj.get("methodology") or "")
                    e_baseline_year = st.number_input("Baseline year", min_value=1900, max_value=2100, value=int(proj.get("baseline_year") or 2024))

                d1, d2, d3 = st.columns(3)
                with d1:
                    e_start_date = st.text_input("Start date (YYYY-MM-DD)", value=proj.get("start_date") or "")
                with d2:
                    e_end_date = st.text_input("End date (YYYY-MM-DD)", value=proj.get("end_date") or "")
                with d3:
                    e_status = st.selectbox("Status", ["Active", "Archived"], index=["Active","Archived"].index(proj.get("status") or "Active"))

                e_description = st.text_area("Description / notes", value=proj.get("description") or "", height=80)

                save = st.form_submit_button("Save changes", use_container_width=True)

                if save:
                    before = proj.copy()
                    ts = now_iso()
                    db_exec(
                        """
                        UPDATE projects SET
                            project_code=?, project_name=?, owner_org=?, country=?, region=?, sector=?,
                            methodology=?, standard=?, baseline_year=?, start_date=?, end_date=?, status=?,
                            description=?, updated_at=?
                        WHERE project_id=?
                        """,
                        (
                            e_project_code.strip(),
                            e_project_name.strip(),
                            e_owner_org.strip() or None,
                            e_country.strip() or None,
                            e_region.strip() or None,
                            e_sector,
                            e_methodology.strip() or None,
                            e_standard,
                            int(e_baseline_year),
                            e_start_date.strip() or None,
                            e_end_date.strip() or None,
                            e_status,
                            e_description.strip() or None,
                            ts,
                            proj["project_id"],
                        ),
                    )
                    after = get_project(proj["project_id"]) or {}
                    audit_log(
                        action="UPDATE",
                        entity_type="project",
                        entity_id=proj["project_id"],
                        project_id=proj["project_id"],
                        before=before,
                        after=after,
                    )
                    list_projects.clear()
                    st.success("Saved.")
                    st.rerun()

            st.markdown("### 📄 Current projects table")
            st.dataframe(dfp, use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# TAB 2: CREDITS & SALES
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("💳 Credits & Sales")

    proj = active_project()
    if not proj:
        st.warning("Select an active project in the Projects tab first.")
    else:
        st.markdown(f"**Active project:** `{proj['project_code']}` — {proj['project_name']}")

        # Credits issued
        st.markdown("### 🧾 Record credit issuance (vintage)")
        with st.form("create_credit_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                vintage_year = st.number_input("Vintage year", min_value=1900, max_value=2100, value=date.today().year)
                credits_issued = st.number_input("Credits issued", min_value=0.0, value=0.0, step=100.0)
            with c2:
                issuance_date = st.date_input("Issuance date", value=date.today())
                registry_program = st.text_input("Registry program", placeholder="Verra VCS / GS / ...")
            with c3:
                serial_range = st.text_input("Serial range (optional)", placeholder="e.g., ABC-0001 to ABC-1000")
            notes = st.text_area("Notes", height=70)

            submit_credit = st.form_submit_button("Add issuance", use_container_width=True)
            if submit_credit:
                cid = str(uuid.uuid4())
                ts = now_iso()
                db_exec(
                    """
                    INSERT INTO credits (
                        credit_id, project_id, vintage_year, credits_issued, issuance_date,
                        registry_program, serial_range, notes, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cid, proj["project_id"], int(vintage_year), float(credits_issued),
                        issuance_date.isoformat() if issuance_date else None,
                        registry_program.strip() or None,
                        serial_range.strip() or None,
                        notes.strip() or None,
                        ts, ts
                    )
                )
                audit_log(
                    action="CREATE",
                    entity_type="credit_issuance",
                    entity_id=cid,
                    project_id=proj["project_id"],
                    after={"vintage_year": vintage_year, "credits_issued": credits_issued},
                )
                st.success("Issuance recorded.")
                st.rerun()

        # Sales
        st.markdown("### 💰 Record sale")
        credits_df = db_query(
            "SELECT credit_id, vintage_year, credits_issued, issuance_date FROM credits WHERE project_id=? ORDER BY vintage_year DESC",
            (proj["project_id"],)
        )
        credit_options = ["(no link)"] + (credits_df["credit_id"].tolist() if not credits_df.empty else [])

        with st.form("create_sale_form", clear_on_submit=True):
            s1, s2, s3 = st.columns(3)
            with s1:
                sale_date = st.date_input("Sale date", value=date.today())
                buyer = st.text_input("Buyer", placeholder="Corporate buyer / trader / broker")
            with s2:
                credits_sold = st.number_input("Credits sold", min_value=0.0, value=0.0, step=10.0)
                price_per_credit = st.number_input("Price per credit", min_value=0.0, value=0.0, step=0.5)
            with s3:
                currency = st.selectbox("Currency", ["USD", "EUR", "ZAR", "GBP", "Other"])
                contract_ref = st.text_input("Contract ref", placeholder="PO / contract ID")

            link_credit = st.selectbox(
                "Link to issuance (optional)",
                options=credit_options,
                format_func=lambda x: "(no link)" if x == "(no link)" else f"{x} (vintage {int(credits_df.loc[credits_df.credit_id==x, 'vintage_year'].values[0])})"
                if (x != "(no link)" and not credits_df.empty and (credits_df.credit_id == x).any()) else str(x)
            )
            notes = st.text_area("Notes", height=70)

            submit_sale = st.form_submit_button("Record sale", use_container_width=True)
            if submit_sale:
                sid = str(uuid.uuid4())
                ts = now_iso()
                db_exec(
                    """
                    INSERT INTO sales (
                        sale_id, project_id, credit_id, sale_date, buyer, credits_sold,
                        price_per_credit, currency, contract_ref, notes, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sid,
                        proj["project_id"],
                        None if link_credit == "(no link)" else link_credit,
                        sale_date.isoformat(),
                        buyer.strip() or None,
                        float(credits_sold),
                        float(price_per_credit) if price_per_credit else None,
                        currency,
                        contract_ref.strip() or None,
                        notes.strip() or None,
                        ts, ts
                    )
                )
                audit_log(
                    action="CREATE",
                    entity_type="sale",
                    entity_id=sid,
                    project_id=proj["project_id"],
                    after={"credits_sold": credits_sold, "price_per_credit": price_per_credit, "currency": currency},
                )
                st.success("Sale recorded.")
                st.rerun()

        # Summary + tables
        st.markdown("### 📈 Summary")
        issued = db_query("SELECT COALESCE(SUM(credits_issued),0) as total_issued FROM credits WHERE project_id=?", (proj["project_id"],)).iloc[0]["total_issued"]
        sold = db_query("SELECT COALESCE(SUM(credits_sold),0) as total_sold FROM sales WHERE project_id=?", (proj["project_id"],)).iloc[0]["total_sold"]
        revenue = db_query(
            "SELECT COALESCE(SUM(credits_sold * COALESCE(price_per_credit,0)),0) as revenue FROM sales WHERE project_id=?",
            (proj["project_id"],)
        ).iloc[0]["revenue"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Credits issued", f"{issued:,.2f}")
        c2.metric("Credits sold", f"{sold:,.2f}")
        c3.metric("Revenue (nominal)", f"{revenue:,.2f}")

        st.markdown("### 🧾 Issuances")
        st.dataframe(credits_df, use_container_width=True, hide_index=True)

        st.markdown("### 💰 Sales")
        sales_df = db_query(
            """
            SELECT sale_id, sale_date, buyer, credits_sold, price_per_credit, currency, contract_ref, credit_id
            FROM sales WHERE project_id=? ORDER BY sale_date DESC
            """,
            (proj["project_id"],)
        )
        st.dataframe(sales_df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# TAB 3: AUDIT
# ------------------------------------------------------------
with tabs[2]:
    st.subheader("📝 Audit Trail")

    proj = active_project()
    scope = st.radio("Scope", ["All", "Active project only"], horizontal=True, index=1 if proj else 0)
    if scope == "Active project only" and proj:
        df = db_query(
            "SELECT timestamp, actor, action, entity_type, entity_id, project_id FROM audit_logs WHERE project_id=? ORDER BY timestamp DESC LIMIT 500",
            (proj["project_id"],)
        )
    else:
        df = db_query(
            "SELECT timestamp, actor, action, entity_type, entity_id, project_id FROM audit_logs ORDER BY timestamp DESC LIMIT 500"
        )

    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("View raw audit JSON (advanced)"):
        audit_id = st.text_input("Paste audit_id to inspect")
        if audit_id:
            row = db_query("SELECT * FROM audit_logs WHERE audit_id=?", (audit_id,))
            if row.empty:
                st.warning("Not found.")
            else:
                st.json(row.iloc[0].to_dict())


# ------------------------------------------------------------
# TAB 4: EXPORT
# ------------------------------------------------------------
with tabs[3]:
    st.subheader("⬇️ Export")

    proj = active_project()
    export_scope = st.radio("Export scope", ["Active project", "All projects"], horizontal=True, index=0)

    def make_exports(project_only: bool):
        if project_only and proj:
            pid = proj["project_id"]
            return {
                "projects": db_query("SELECT * FROM projects WHERE project_id=?", (pid,)),
                "credits": db_query("SELECT * FROM credits WHERE project_id=?", (pid,)),
                "sales": db_query("SELECT * FROM sales WHERE project_id=?", (pid,)),
                "audit": db_query("SELECT * FROM audit_logs WHERE project_id=? ORDER BY timestamp DESC", (pid,)),
            }
        else:
            return {
                "projects": db_query("SELECT * FROM projects ORDER BY updated_at DESC"),
                "credits": db_query("SELECT * FROM credits ORDER BY updated_at DESC"),
                "sales": db_query("SELECT * FROM sales ORDER BY updated_at DESC"),
                "audit": db_query("SELECT * FROM audit_logs ORDER BY timestamp DESC"),
            }

    if export_scope == "Active project" and not proj:
        st.warning("Select an active project in the Projects tab first.")
    else:
        exports = make_exports(project_only=(export_scope == "Active project"))
        st.markdown("### Download CSVs")
        for name, df in exports.items():
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download {name}.csv",
                data=csv,
                file_name=f"{name}.csv",
                mime="text/csv",
                use_container_width=True,
            )

st.divider()
st.caption("Beta registry. Next steps: methodology-specific MRV calculators, document evidence vault, and automated validation rules.")

