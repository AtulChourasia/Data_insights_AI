from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class QuestionRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None
    model: str = "gpt-4"

class IterativeQuestionRequest(QuestionRequest):
    iteration: int = 1
    previous_attempts: List[Dict[str, Any]] = Field(default_factory=list)

class InsightsRequest(BaseModel):
    data: List[Dict[str, Any]]
    columns: List[str]
    question: str
    row_count: int

class AIResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]] = []
    confidence: float = 1.0
    metadata: Dict[str, Any] = {}

class UnpivotRequest(BaseModel):
    filenames: List[str]

class WorksheetUnpivotRequest(BaseModel):
    filename: str
    sheet_name: str

class EnhancedUnpivotRequest(BaseModel):
    worksheet_requests: List[WorksheetUnpivotRequest]

class UnpivotTablesRequest(BaseModel):
    table_names: List[str]

class AIAnalysisResult(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float
    model_used: str
    tokens_used: int
    cost: float = 0.0
