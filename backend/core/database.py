from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///attendance_analytics.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, default="CS101 Session")
    created_at = Column(DateTime, default=datetime.utcnow)

class SessionPerson(Base):
    __tablename__ = "session_people"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer)
    student_id = Column(String)

class PersonMetric(Base):
    __tablename__ = "person_metrics"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer)
    student_id = Column(String)
    attention = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)
