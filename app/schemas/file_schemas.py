from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class FileBase(BaseModel):
    filename: str
    content_type: str
    size: int

class FileCreate(FileBase):
    pass

class FileResponse(FileBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class FileListResponse(BaseModel):
    files: List[FileResponse]
    total: int

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

class FlattenRequest(BaseModel):
    filename: str
    header_rows: int

class FlattenFilesRequest(BaseModel):
    flatten_requests: List[FlattenRequest]

class WorksheetFlattenRequest(BaseModel):
    filename: str
    sheet_name: str
    header_rows: int

class EnhancedFlattenFilesRequest(BaseModel):
    flatten_requests: List[WorksheetFlattenRequest]
