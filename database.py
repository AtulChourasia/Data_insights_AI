from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base

# SQLite database connection
DATABASE_URL = "sqlite:///./files.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
metadata = MetaData()
Base = declarative_base()

