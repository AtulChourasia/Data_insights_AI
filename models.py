from sqlalchemy import Column, Integer, String, LargeBinary
from database import Base
from typing import List
from pydantic import BaseModel

class File(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    content = Column(LargeBinary, nullable=False)


class FlattenRequest(BaseModel):
    filename: str
    header_rows: int

class FlattenFilesRequest(BaseModel):
    flatten_requests: List[FlattenRequest]

class UnpivotRequest(BaseModel):
    filenames: List[str]

class QuestionRequest(BaseModel):
    question: str

class IterativeQuestionRequest(BaseModel):
    question: str
    iteration: int
    previous_attempts: List[dict] = []

class InsightsRequest(BaseModel):
    data: List[dict]
    columns: List[str]
    question: str
    row_count: int

class CustomQueryRequest(BaseModel):
    sql_query: str
    question: str
    

class WorksheetFlattenRequest(BaseModel):
    filename: str
    sheet_name: str
    header_rows: int

class EnhancedFlattenFilesRequest(BaseModel):
    flatten_requests: List[WorksheetFlattenRequest]

class WorksheetUnpivotRequest(BaseModel):
    filename: str
    sheet_name: str

class EnhancedUnpivotRequest(BaseModel):
    worksheet_requests: List[WorksheetUnpivotRequest]

class WorksheetInfo(BaseModel):
    sheet_name: str
    table_name: str
    rows: int
    columns: int
    tokens: int = 0
    is_complex: bool = False

class FileWithWorksheets(BaseModel):
    filename: str
    worksheets: List[WorksheetInfo]
    total_tokens: int
    total_worksheets: int
    
class UnpivotTablesRequest(BaseModel):
    table_names: List[str]