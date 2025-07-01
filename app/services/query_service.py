import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import pandas as pd
from sqlalchemy import text
import json

from ..models.models import QueryHistory
from ..schemas.query_schemas import QueryBase, QueryResponse, QueryResult, QueryAnalysis
from ..config.database import SessionLocal

logger = logging.getLogger(__name__)

class QueryService:
    def __init__(self, db: Session):
        self.db = db
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute a SQL query and return the results."""
        try:
            # Log the query
            query_history = QueryHistory(
                query=query,
                status="executing"
            )
            self.db.add(query_history)
            self.db.commit()
            self.db.refresh(query_history)
            
            try:
                # Execute the query
                result = self.db.execute(text(query), params or {})
                
                # Convert to list of dicts
                columns = list(result.keys())
                data = [dict(zip(columns, row)) for row in result.fetchall()]
                
                # Update query history
                query_history.status = "completed"
                query_history.result = json.dumps({"columns": columns, "data": data})
                self.db.commit()
                
                return QueryResult(
                    columns=columns,
                    data=data,
                    row_count=len(data),
                    execution_time=0.0  # TODO: Track execution time
                )
                
            except Exception as e:
                # Log the error
                error_msg = str(e)
                query_history.status = "failed"
                query_history.result = json.dumps({"error": error_msg})
                self.db.commit()
                
                logger.error(f"Query execution failed: {error_msg}")
                raise ValueError(f"Query execution failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"Unexpected error in execute_query: {e}")
            raise
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a SQL query for potential issues and optimizations."""
        # TODO: Implement query analysis
        return QueryAnalysis(
            is_valid=True,
            complexity="medium",
            tables_accessed=[],
            columns_accessed=[],
            potential_issues=[],
            suggestions=[]
        )
    
    def get_query_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve query execution history."""
        history = self.db.query(QueryHistory)\
            .order_by(QueryHistory.created_at.desc())\
            .limit(limit)\
            .all()
            
        return [{
            "id": item.id,
            "query": item.query,
            "status": item.status,
            "created_at": item.created_at,
            "execution_time": 0.0  # TODO: Track execution time
        } for item in history]
    
    def get_query_result(self, query_id: int) -> Optional[Dict[str, Any]]:
        """Get the result of a specific query by ID."""
        query = self.db.query(QueryHistory).filter(QueryHistory.id == query_id).first()
        if not query or not query.result:
            return None
            
        try:
            result = json.loads(query.result)
            return {
                "id": query.id,
                "query": query.query,
                "status": query.status,
                "result": result,
                "created_at": query.created_at,
                "execution_time": 0.0  # TODO: Track execution time
            }
        except json.JSONDecodeError:
            return None

# Factory function to get query service
def get_query_service():
    db = SessionLocal()
    try:
        yield QueryService(db)
    finally:
        db.close()
