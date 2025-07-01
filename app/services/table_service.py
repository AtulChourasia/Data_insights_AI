import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import pandas as pd
from sqlalchemy import inspect, text

from ..models.models import File, Worksheet
from ..schemas.file_schemas import WorksheetInfo, FileWithWorksheets
from ..config.database import SessionLocal

logger = logging.getLogger(__name__)

class TableService:
    def __init__(self, db: Session):
        self.db = db
    
    def get_table_preview(self, table_name: str, limit: int = 10) -> Dict[str, Any]:
        """Get a preview of a table's data."""
        try:
            # Get table columns
            inspector = inspect(self.db.get_bind())
            columns = inspector.get_columns(table_name)
            
            if not columns:
                return {"error": f"Table {table_name} not found or has no columns"}
            
            # Get sample data
            query = f"SELECT * FROM \"{table_name}\" LIMIT {limit}"
            result = self.db.execute(text(query))
            
            # Convert to list of dicts
            rows = [dict(row) for row in result.mappings()]
            
            return {
                "table_name": table_name,
                "columns": [{"name": col["name"], "type": str(col["type"])} for col in columns],
                "data": rows,
                "row_count": len(rows),
                "total_columns": len(columns)
            }
            
        except Exception as e:
            logger.error(f"Error getting table preview for {table_name}: {e}")
            return {"error": f"Failed to get table preview: {str(e)}"}
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get the schema of a table."""
        try:
            inspector = inspect(self.db.get_bind())
            columns = inspector.get_columns(table_name)
            
            if not columns:
                return {"error": f"Table {table_name} not found or has no columns"}
            
            # Get primary key
            primary_key = inspector.get_pk_constraint(table_name)
            
            # Get foreign keys
            foreign_keys = inspector.get_foreign_keys(table_name)
            
            return {
                "table_name": table_name,
                "columns": [{"name": col["name"], "type": str(col["type"]), "nullable": col["nullable"]} for col in columns],
                "primary_key": primary_key.get("constrained_columns", []),
                "foreign_keys": foreign_keys
            }
            
        except Exception as e:
            logger.error(f"Error getting schema for {table_name}: {e}")
            return {"error": f"Failed to get table schema: {str(e)}"}
    
    def list_tables(self) -> List[Dict[str, Any]]:
        """List all tables in the database."""
        try:
            inspector = inspect(self.db.get_bind())
            tables = inspector.get_table_names()
            
            result = []
            for table in tables:
                try:
                    # Get row count (with error handling)
                    try:
                        count = self.db.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()
                    except:
                        count = -1  # Indeterminate count
                    
                    # Get columns
                    columns = inspector.get_columns(table)
                    
                    result.append({
                        "name": table,
                        "row_count": count,
                        "column_count": len(columns),
                        "columns": [col["name"] for col in columns[:5]],  # First 5 columns
                        "has_more_columns": len(columns) > 5
                    })
                except Exception as e:
                    logger.error(f"Error getting info for table {table}: {e}")
                    result.append({
                        "name": table,
                        "error": str(e)
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return [{"error": f"Failed to list tables: {str(e)}"}]
    
    def delete_table(self, table_name: str) -> Dict[str, Any]:
        """Delete a table from the database."""
        try:
            # Check if table exists
            inspector = inspect(self.db.get_bind())
            if table_name not in inspector.get_table_names():
                return {"success": False, "error": f"Table {table_name} does not exist"}
            
            # Delete the table
            self.db.execute(text(f'DROP TABLE "{table_name}"'))
            self.db.commit()
            
            # Also delete any file/worksheet records associated with this table
            self.db.query(Worksheet).filter(Worksheet.table_name == table_name).delete(synchronize_session=False)
            self.db.commit()
            
            return {"success": True, "message": f"Table {table_name} deleted successfully"}
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting table {table_name}: {e}")
            return {"success": False, "error": f"Failed to delete table: {str(e)}"}

# Factory function to get table service
def get_table_service():
    db = SessionLocal()
    try:
        yield TableService(db)
    finally:
        db.close()
