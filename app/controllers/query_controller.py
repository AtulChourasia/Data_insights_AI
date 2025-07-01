from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional

from ..services.query_service import QueryService, get_query_service
from ..schemas.query_schemas import (
    QueryBase, 
    QueryResponse, 
    QueryResult,
    QueryAnalysis,
    CustomQueryRequest
)

router = APIRouter()

@router.post("/execute/", response_model=QueryResult)
async def execute_query(
    query: QueryBase,
    query_service: QueryService = Depends(get_query_service)
):
    """
    Execute a SQL query and return the results.
    """
    try:
        result = query_service.execute_query(query.query, query.parameters)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/analyze/", response_model=QueryAnalysis)
async def analyze_query(
    query: QueryBase,
    query_service: QueryService = Depends(get_query_service)
):
    """
    Analyze a SQL query for potential issues and optimizations.
    """
    try:
        return query_service.analyze_query(query.query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/history/", response_model=List[Dict[str, Any]])
async def get_query_history(
    limit: int = 50,
    query_service: QueryService = Depends(get_query_service)
):
    """
    Get query execution history.
    """
    try:
        return query_service.get_query_history(limit)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving query history: {str(e)}"
        )

@router.get("/{query_id}", response_model=Dict[str, Any])
async def get_query_result(
    query_id: int,
    query_service: QueryService = Depends(get_query_service)
):
    """
    Get the result of a specific query by ID.
    """
    result = query_service.get_query_result(query_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query with ID {query_id} not found"
        )
    return result

@router.post("/custom/", response_model=QueryResult)
async def execute_custom_query(
    request: CustomQueryRequest,
    query_service: QueryService = Depends(get_query_service)
):
    """
    Execute a custom SQL query with additional context.
    """
    try:
        # Here you could add additional validation or processing
        # For example, check if the user has permission to execute this query
        
        # Execute the query
        result = query_service.execute_query(request.sql_query)
        
        # You could also log this custom query for auditing
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error executing custom query: {str(e)}"
        )

@router.post("/nl-to-sql/", response_model=QueryResult)
async def natural_language_to_sql(
    request: QueryBase,
    query_service: QueryService = Depends(get_query_service)
):
    """
    Convert natural language to SQL and execute the query.
    This would typically integrate with an AI service.
    """
    try:
        # This is a placeholder - in a real implementation, you would:
        # 1. Use an AI service to convert the natural language to SQL
        # 2. Validate the generated SQL
        # 3. Execute the query
        # 4. Return the results
        
        # For now, just pass through to execute_query
        return await execute_query(QueryBase(query=request.query), query_service)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing natural language query: {str(e)}"
        )
