# registry/crud.py
import datetime as dt
from typing import Optional, List, Tuple

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from .models import Project, Emission, ProjectLog


# ---------- Dashboard queries ----------

def get_kpis(session: Session) -> tuple[int, float, float]:
    total_projects = session.execute(
        select(func.count(Project.id))
    ).scalar_one()

    total_emissions = session.execute(
        select(func.coalesce(func.sum(Emission.quantity_tCO2e), 0.0))
    ).scalar_one()

    cutoff = dt.date.today() - dt.timedelta(days=30)
    last30 = session.execute(
        select(func.coalesce(func.sum(Emission.quantity_tCO2e), 0.0))
        .where(Emission.date >= cutoff)
    ).scalar_one()

    return total_projects, float(total_emissions), float(last30)


def get_totals_by_business_unit(session: Session) -> List[Tuple[str, float]]:
    rows = session.execute(
        select(Project.business_unit,
               func.coalesce(func.sum(Emission.quantity_tCO2e), 0.0))
        .join(Emission, Emission.project_id == Project.id, isouter=True)
        .group_by(Project.business_unit)
    ).all()
    return [(bu or "", float(total)) for bu, total in rows]


def get_recent_emissions(session: Session, limit: int = 10):
    return session.execute(
        select(
            Emission.id,
            Emission.date,
            Project.name,
            Emission.quantity_tCO2e,
            Emission.activity_type,
        )
        .join(Project, Project.id == Emission.project_id)
        .order_by(Emission.date.desc(), Emission.id.desc())
        .limit(limit)
    ).all()


# ---------- Project CRUD ----------

def list_projects(session: Session):
    return session.execute(
        select(Project).order_by(Project.id.asc())
    ).scalars().all()


def get_project(session: Session, project_id: int) -> Optional[Project]:
    return session.get(Project, project_id)


def find_project_by_name(session: Session, name: str) -> Optional[Project]:
    return session.execute(
        select(Project).where(Project.name == name)
    ).scalars().first()


def create_project(
    session: Session,
    name: str,
    location: str,
    business_unit: str,
    sub_division: str,
    start_date,
    end_date,
    description: str,
) -> Project:
    project = Project(
        name=name,
        location=location,
        business_unit=business_unit,
        sub_division=sub_division,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )
    session.add(project)
    session.flush()
    session.add(ProjectLog(project_id=project.id, action="create", note="Project created."))
    session.commit()
    return project


def update_project(
    session: Session,
    project_id: int,
    name: str,
    location: str,
    business_unit: str,
    sub_division: str,
    start_date,
    end_date,
    description: str,
) -> Optional[Project]:
    project = session.get(Project, project_id)
    if not project:
        return None

    project.name = name
    project.location = location
    project.business_unit = business_unit
    project.sub_division = sub_division
    project.start_date = start_date
    project.end_date = end_date
    project.description = description

    session.add(ProjectLog(project_id=project.id, action="edit", note="Project edited."))
    session.commit()
    return project


def delete_project(session: Session, project_id: int) -> bool:
    project = session.get(Project, project_id)
    if not project:
        return False
    session.add(ProjectLog(project_id=project_id, action="delete", note="Project deleted."))
    session.delete(project)
    session.commit()
    return True


# ---------- Emissions CRUD ----------

def list_emissions_for_project(session: Session, project_id: int):
    return session.execute(
        select(Emission)
        .where(Emission.project_id == project_id)
        .order_by(Emission.date.asc())
    ).scalars().all()


def add_emission(
    session: Session,
    project_id: int,
    date,
    quantity_tCO2e: float,
    activity_type: str,
    notes: str,
) -> Emission:
    emission = Emission(
        project_id=project_id,
        date=date,
        quantity_tCO2e=quantity_tCO2e,
        activity_type=activity_type,
        notes=notes,
    )
    session.add(emission)
    session.add(
        ProjectLog(
            project_id=project_id,
            action="emission_add",
            note=f"Added {quantity_tCO2e} tCO2e",
        )
    )
    session.commit()
    return emission


def delete_emission(session: Session, emission_id: int) -> bool:
    emission = session.get(Emission, emission_id)
    if not emission:
        return False
    pid = emission.project_id
    session.add(
        ProjectLog(
            project_id=pid,
            action="emission_delete",
            note=f"Deleted emission {emission_id}",
        )
    )
    session.delete(emission)
    session.commit()
    return True


# ---------- Logs & export ----------

def get_all_logs(session: Session):
    return session.execute(
        select(ProjectLog)
        .order_by(ProjectLog.timestamp.desc())
    ).scalars().all()


def get_emissions_for_export(session: Session, project_id: int):
    return session.execute(
        select(Emission)
        .where(Emission.project_id == project_id)
        .order_by(Emission.date.asc())
    ).scalars().all()

