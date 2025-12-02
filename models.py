# registry/models.py
import datetime as dt
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from .database import Base, engine

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    location = Column(String, default="")
    business_unit = Column(String, default="")
    sub_division = Column(String, default="")
    start_date = Column(Date)
    end_date = Column(Date)
    description = Column(String, default="")

    emissions = relationship(
        "Emission",
        back_populates="project",
        cascade="all, delete-orphan",
    )


class Emission(Base):
    __tablename__ = "emissions"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    date = Column(Date, nullable=False)
    quantity_tCO2e = Column(Float, nullable=False)
    activity_type = Column(String, default="")
    notes = Column(String, default="")

    project = relationship("Project", back_populates="emissions")


class ProjectLog(Base):
    __tablename__ = "project_logs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer)
    action = Column(String)
    timestamp = Column(DateTime, default=dt.datetime.utcnow)
    note = Column(String, default="")

# same as your PyQt constant
BUSINESS_UNITS = {
    "Energy": ["Oil & Gas", "Renewables", "Utilities"],
    "Agriculture": ["Crop Production", "Livestock", "Agri-Tech"],
    "Natural Resources": ["Forestry", "Water", "Mining"],
}

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

