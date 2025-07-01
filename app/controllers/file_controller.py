from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional
import os
import tempfile
import shutil
from pathlib import Path

from ..services.file_service import FileService, get_file_service
from ..schemas.file_schemas import FileResponse, FileListResponse, FileWithWorksheets, FlattenFilesRequest, EnhancedFlattenFilesRequest
from ..models.models import File

router = APIRouter()

@router.post("/upload/", response_model=List[FileResponse], status_code=status.HTTP_201_CREATED)
async def upload_files(
    files: List[UploadFile] = File(...),
    file_service: FileService = Depends(get_file_service)
):
    """
    Upload multiple files for processing.
    """
    try:
        results = []
        for file in files:
            try:
                # Save file to temporary location
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, file.filename)
                
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Process the file
                db_file = file_service.process_uploaded_file(file.file, file.filename)
                results.append({
                    "id": db_file.id,
                    "filename": db_file.filename,
                    "created_at": db_file.created_at
                })
                
            except Exception as e:
                # Log error but continue with other files
                print(f"Error processing file {file.filename}: {e}")
                continue
                
            finally:
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing files: {str(e)}"
        )

@router.get("/", response_model=FileListResponse)
async def list_uploaded_files(
    file_service: FileService = Depends(get_file_service)
):
    """
    List all uploaded files with their worksheets.
    """
    try:
        files = file_service.get_uploaded_tables()
        return {"files": files, "total": len(files)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing files: {str(e)}"
        )

@router.get("/{file_id}", response_model=FileWithWorksheets)
async def get_file_details(
    file_id: int,
    file_service: FileService = Depends(get_file_service)
):
    """
    Get details of a specific file including its worksheets.
    """
    try:
        file = file_service.get_file_by_id(file_id)
        if not file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File with ID {file_id} not found"
            )
            
        # Get worksheets for this file
        worksheets = file.worksheets
        
        return {
            "filename": file.filename,
            "worksheets": [{
                "sheet_name": ws.sheet_name,
                "table_name": ws.table_name,
                "rows": 0,  # Would need to be populated from actual data
                "columns": 0,  # Would need to be populated from actual data
                "tokens": 0,  # Would need to be calculated
                "is_complex": False  # Would need to be determined
            } for ws in worksheets],
            "total_tokens": 0,  # Would need to be calculated
            "total_worksheets": len(worksheets)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting file details: {str(e)}"
        )

@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    file_id: int,
    file_service: FileService = Depends(get_file_service)
):
    """
    Delete a file and all its associated data.
    """
    try:
        if not file_service.delete_file(file_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File with ID {file_id} not found"
            )
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting file: {str(e)}"
        )

@router.post("/flatten/", status_code=status.HTTP_202_ACCEPTED)
async def flatten_files(
    request: FlattenFilesRequest,
    file_service: FileService = Depends(get_file_service)
):
    """
    Flatten uploaded files for processing.
    """
    try:
        # Implementation would go here
        return {"message": "Flattening process started", "request": request.dict()}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing flatten request: {str(e)}"
        )

@router.post("/enhanced-flatten/", status_code=status.HTTP_202_ACCEPTED)
async def enhanced_flatten_files(
    request: EnhancedFlattenFilesRequest,
    file_service: FileService = Depends(get_file_service)
):
    """
    Enhanced flattening of files with worksheet-specific configurations.
    """
    try:
        # Implementation would go here
        return {"message": "Enhanced flattening process started", "request": request.dict()}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing enhanced flatten request: {str(e)}"
        )
