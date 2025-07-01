from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class QueryBase(BaseModel):
    query: str
    parameters: Optional[Dict[str, Any]] = None

class QueryRequest(QueryBase):
    pass

class QueryResponse(QueryBase):
    id: int
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class QueryHistoryResponse(BaseModel):
    queries: List[QueryResponse]
    total: int

class CustomQueryRequest(BaseModel):
    sql_query: str
    question: str

class QueryResult(BaseModel):
    columns: List[str]
    data: List[Dict[str, Any]]
    row_count: int
    execution_time: float

class QueryAnalysis(BaseModel):
    is_valid: bool
    complexity: str = "low"  # low, medium, high
    estimated_cost: Optional[float] = None
    tables_accessed: List[str] = []
    columns_accessed: List[str] = []
    potential_issues: List[str] = []
    suggestions: List[str] = []
