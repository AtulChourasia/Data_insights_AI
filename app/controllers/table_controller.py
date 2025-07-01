from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional

from ..services.table_service import TableService, get_table_service

router = APIRouter()

@router.get("/", response_model=List[Dict[str, Any]])
async def list_tables(
    table_service: TableService = Depends(get_table_service)
):
    """
    List all tables in the database.
    """
    try:
        return table_service.list_tables()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing tables: {str(e)}"
        )

@router.get("/{table_name}", response_model=Dict[str, Any])
async def get_table(
    table_name: str,
    limit: int = Query(10, ge=1, le=100, description="Number of rows to return"),
    table_service: TableService = Depends(get_table_service)
):
    """
    Get a preview of a table's data.
    """
    try:
        result = table_service.get_table_preview(table_name, limit)
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting table data: {str(e)}"
        )

@router.get("/{table_name}/schema", response_model=Dict[str, Any])
async def get_table_schema(
    table_name: str,
    table_service: TableService = Depends(get_table_service)
):
    """
    Get the schema of a table.
    """
    try:
        result = table_service.get_table_schema(table_name)
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting table schema: {str(e)}"
        )

@router.delete("/{table_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_table(
    table_name: str,
    table_service: TableService = Depends(get_table_service)
):
    """
    Delete a table from the database.
    """
    try:
        result = table_service.delete_table(table_name)
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.get("error", "Failed to delete table")
            )
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting table: {str(e)}"
        )

@router.get("/{table_name}/rows/count", response_model=Dict[str, int])
async def get_row_count(
    table_name: str,
    table_service: TableService = Depends(get_table_service)
):
    """
    Get the number of rows in a table.
    """
    try:
        preview = table_service.get_table_preview(table_name, limit=1)
        if "error" in preview:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=preview["error"]
            )
        
        # In a real implementation, you would run a COUNT(*) query
        # For now, we'll return the preview's row count which is just for the preview
        return {"count": preview.get("row_count", 0)}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting row count: {str(e)}"
        )

@router.get("/{table_name}/columns", response_model=List[Dict[str, Any]])
async def get_table_columns(
    table_name: str,
    table_service: TableService = Depends(get_table_service)
):
    """
    Get the columns of a table.
    """
    try:
        schema = table_service.get_table_schema(table_name)
        if "error" in schema:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=schema["error"]
            )
        return schema.get("columns", [])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting table columns: {str(e)}"
        )
