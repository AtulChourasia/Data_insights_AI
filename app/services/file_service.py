import os
import logging
from typing import List, Dict, Any, Optional, BinaryIO
from sqlalchemy.orm import Session
import pandas as pd
from datetime import datetime
import shutil
import tempfile
from pathlib import Path

from ..models.models import File, Worksheet
from ..schemas.file_schemas import FileCreate, FileResponse, WorksheetInfo, FileWithWorksheets
from ..config.database import SessionLocal

logger = logging.getLogger(__name__)

class FileService:
    def __init__(self, db: Session):
        self.db = db
        self.upload_dir = os.path.join(tempfile.gettempdir(), "data_insights_uploads")
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def save_uploaded_file(self, file: BinaryIO, filename: str) -> str:
        """Save uploaded file to temporary storage and return file path."""
        file_path = os.path.join(self.upload_dir, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file, buffer)
        return file_path
    
    def create_file_record(self, filename: str, content: bytes) -> File:
        """Create a new file record in the database."""
        db_file = File(filename=filename, content=content)
        self.db.add(db_file)
        self.db.commit()
        self.db.refresh(db_file)
        return db_file
    
    def get_file_by_id(self, file_id: int) -> Optional[File]:
        """Retrieve a file by its ID."""
        return self.db.query(File).filter(File.id == file_id).first()
    
    def delete_file(self, file_id: int) -> bool:
        """Delete a file record and its associated data."""
        db_file = self.get_file_by_id(file_id)
        if not db_file:
            return False
            
        # Delete associated worksheets
        self.db.query(Worksheet).filter(Worksheet.file_id == file_id).delete()
        self.db.delete(db_file)
        self.db.commit()
        return True
    
    def process_uploaded_file(self, file: BinaryIO, filename: str) -> File:
        """Process an uploaded file and save it to the database."""
        # Save file content
        content = file.read()
        
        # Create file record
        return self.create_file_record(filename, content)
    
    def extract_worksheet_info(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract worksheet information from an Excel file."""
        try:
            excel_file = pd.ExcelFile(file_path)
            worksheets = []
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=5)
                    worksheets.append({
                        "sheet_name": sheet_name,
                        "rows": len(df),
                        "columns": len(df.columns) if not df.empty else 0,
                        "headers": list(df.columns) if not df.empty else []
                    })
                except Exception as e:
                    logger.error(f"Error reading sheet {sheet_name}: {e}")
                    continue
                    
            return worksheets
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    def get_uploaded_tables(self) -> List[Dict[str, Any]]:
        """Get a list of all uploaded tables with their worksheets."""
        files = self.db.query(File).all()
        result = []
        
        for file in files:
            worksheets = self.db.query(Worksheet).filter(Worksheet.file_id == file.id).all()
            result.append({
                "file_id": file.id,
                "filename": file.filename,
                "created_at": file.created_at,
                "worksheets": [{
                    "id": ws.id,
                    "sheet_name": ws.sheet_name,
                    "table_name": ws.table_name,
                    "is_processed": ws.is_processed
                } for ws in worksheets]
            })
            
        return result

# Factory function to get file service
def get_file_service():
    db = SessionLocal()
    try:
        yield FileService(db)
    finally:
        db.close()
