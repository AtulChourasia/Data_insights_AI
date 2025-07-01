from fastapi import FastAPI, UploadFile, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
from sqlalchemy.orm import sessionmaker
import models
from typing import List
import logging
import pandas as pd
from backend import generate_insight_from_result,generate_sql_prompt 
from models import File 
from ai_unpivot_handler import get_ai_unpivot_strategy
from Helpers.confidence_calculator import calculate_query_confidence

from file_handlers import (
    process_uploaded_files_with_worksheets, 
    process_flatten_files_with_worksheets,     
    get_llms,
    initialize_llms,
    create_table_name_from_filename,
    normalize_column_names 
)


# Set up logging
logger = logging.getLogger(__name__)

# Create SQLite tables
Base.metadata.create_all(bind=engine)

# Dependency for database session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate-insights/")
async def generate_insights(request: models.InsightsRequest, db: Session = Depends(get_db)):
    """Endpoint to generate insights from query results"""
    try:
        # Get initialized LLM instances
        gpt4_llm, gemini_llm, embeddings_model = get_llms()
        
        # Convert the data to a pandas DataFrame
        import pandas as pd
        result_df = pd.DataFrame(request.data)
        
        # Call your insights generation function
        insights = generate_insight_from_result(
            result_df=result_df,
            user_question=request.question,
            llm=gpt4_llm 
        )
        
        return {
            "error": False,
            "insights": insights,
            "message": "Insights generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        return {
            "error": True,
            "insights": None,
            "message": f"Failed to generate insights: {str(e)}"
        }
    
@app.get("/get-uploaded-tables/")
async def get_uploaded_tables(db: Session = Depends(get_db)):
    """Enhanced to show worksheet information grouped by files"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'uploaded_%';"))
            table_names = [row[0] for row in result.fetchall()]
        
        # Group tables by original filename
        files_with_worksheets = {}
        
        for table_name in table_names:
            try:
                df = pd.read_sql_table(table_name, engine)
                
                # Extract filename and sheet name from table name
                # Format: uploaded_filename_sheetname
                parts = table_name.replace('uploaded_', '').split('_')
                
                if len(parts) >= 2:
                    # Last part is likely sheet name, rest is filename
                    sheet_name = parts[-1]
                    filename_parts = parts[:-1]
                    filename_base = '_'.join(filename_parts)
                else:
                    # Single part, assume it's filename (legacy format)
                    filename_base = parts[0] if parts else 'unknown'
                    sheet_name = 'Sheet1'
                
                # Find actual filename from File table
                possible_extensions = ['.csv', '.xlsx', '.xls']
                actual_filename = None
                
                for ext in possible_extensions:
                    test_filename = f"{filename_base.replace('_', ' ')}{ext}"
                    if db.query(File).filter(File.filename == test_filename).first():
                        actual_filename = test_filename
                        break
                
                if not actual_filename:
                    actual_filename = f"{filename_base.replace('_', ' ')}.xlsx"  # Default
                
                worksheet_info = {
                    "sheet_name": sheet_name.replace('_', ' ').title(),
                    "table_name": table_name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns)
                }
                
                if actual_filename not in files_with_worksheets:
                    files_with_worksheets[actual_filename] = {
                        "filename": actual_filename,
                        "worksheets": [],
                        "total_rows": 0,
                        "total_columns": 0
                    }
                
                files_with_worksheets[actual_filename]["worksheets"].append(worksheet_info)
                files_with_worksheets[actual_filename]["total_rows"] += len(df)
                files_with_worksheets[actual_filename]["total_columns"] = max(
                    files_with_worksheets[actual_filename]["total_columns"], 
                    len(df.columns)
                )
                
            except Exception as e:
                logger.error(f"Error reading table {table_name}: {str(e)}")
                continue
        
        # Convert to list format  
        files_list = list(files_with_worksheets.values())
        
        return {
            "error": False,
            "files": files_list,  # Changed from "tables" to "files"
            "total_files": len(files_list),
            "total_tables": len(table_names)
        }
        
    except Exception as e:
        logger.error(f"Error fetching uploaded tables: {str(e)}")
        return {
            "error": True,
            "message": f"Failed to fetch uploaded tables: {str(e)}",
            "files": []
        }


@app.delete("/cleanup-all-files/")
async def cleanup_all_files(db: Session = Depends(get_db)):
    try:
        cleanup_count = 0
        file_records_count = 0
        
        # First, clean up all uploaded tables
        with engine.connect() as conn:
            # Get all uploaded tables
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'uploaded_%'"))
            tables = result.fetchall()
            
            # Drop all tables
            for table in tables:
                conn.execute(text(f"DROP TABLE IF EXISTS `{table[0]}`"))
                cleanup_count += 1
                logger.info(f"Dropped table: {table[0]}")
            
            conn.commit()
        
        # Also clean up File records from SQLAlchemy
        try:
            file_records = db.query(File).all()
            file_records_count = len(file_records)
            
            for file_record in file_records:
                logger.info(f"Deleting file record: {file_record.filename}")
                db.delete(file_record)
            
            db.commit()
            logger.info(f"Successfully deleted {file_records_count} file records from database")
            
        except Exception as file_cleanup_error:
            logger.error(f"Error cleaning up file records: {str(file_cleanup_error)}")
            # Rollback file record changes but don't fail the whole cleanup
            db.rollback()
        
        logger.info(f"Cleanup completed: {cleanup_count} tables, {file_records_count} file records")
        
        return {
            "error": False,
            "message": f"Cleaned up {cleanup_count} tables and {file_records_count} file records successfully",
            "tables_count": cleanup_count,
            "file_records_count": file_records_count
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        return {
            "error": True,
            "message": f"Cleanup failed: {str(e)}"
        }

@app.get("/verify-cleanup/")
async def verify_cleanup(db: Session = Depends(get_db)):
    """Verify that all data has been cleaned up"""
    try:
        # Check for uploaded tables
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'uploaded_%'"))
            remaining_tables = [row[0] for row in result.fetchall()]
        
        # Check for file records
        file_records = db.query(File).all()
        remaining_file_records = len(file_records)
        
        is_clean = len(remaining_tables) == 0 and remaining_file_records == 0
        
        return {
            "cleanup_complete": is_clean,
            "remaining_tables": remaining_tables,
            "remaining_file_records": remaining_file_records,
            "message": "Database is clean" if is_clean else f"Found {len(remaining_tables)} tables and {remaining_file_records} file records"
        }
        
    except Exception as e:
        logger.error(f"Cleanup verification failed: {str(e)}")
        return {
            "error": True,
            "message": f"Verification failed: {str(e)}"
        }


@app.get("/get-table-preview/{table_name}")
async def get_table_preview(table_name: str, db: Session = Depends(get_db)):
    """Get preview data for a specific uploaded table"""
    try:
        # Security check: only allow tables with 'uploaded_' prefix
        if not table_name.startswith('uploaded_'):
            return {
                "error": True,
                "message": "Access denied: Only uploaded tables can be previewed",
            }
        
        df = pd.read_sql_table(table_name, engine)
        
        if df.empty:
            return {
                "error": False,
                "table_name": table_name,
                "columns": [],
                "sample_data": [],
                "total_rows": 0
            }
        
        # Get sample data (first 10 rows)
        sample_data = df.to_dict(orient="records")
        
        return {
            "error": False,
            "table_name": table_name,
            "columns": list(df.columns),
            "sample_data": sample_data,
            "total_rows": len(df)
        }
        
    except Exception as e:
        logger.error(f"Error getting table preview for {table_name}: {str(e)}")
        return {
            "error": True,
            "message": f"Failed to get table preview: {str(e)}",
        }

@app.get("/download-table/{table_name}")
async def download_table(table_name: str, db: Session = Depends(get_db)):
    """Download a specific uploaded table as CSV"""
    try:
        # Security check: only allow tables with 'uploaded_' prefix
        if not table_name.startswith('uploaded_'):
            raise HTTPException(status_code=403, detail="Access denied: Only uploaded tables can be downloaded")
        
        df = pd.read_sql_table(table_name, engine)
        
        # Convert to CSV
        csv_content = df.to_csv(index=False)
        
        # Get clean filename
        clean_name = table_name.replace('uploaded_', '').replace('_', ' ')
        
        from fastapi.responses import StreamingResponse
        import io
        
        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={clean_name}.csv"}
        )
        
    except Exception as e:
        logger.error(f"Error downloading table {table_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download table: {str(e)}")


@app.post("/upload-files/")
async def upload_files(files: List[UploadFile], db: Session = Depends(get_db)):
    """Endpoint to upload multiple files with automatic complex file processing."""
    return await process_uploaded_files_with_worksheets(files, db)


# @app.post("/flatten-files/")
# async def flatten_files(request: models.EnhancedFlattenFilesRequest, db: Session = Depends(get_db)):
#     """Endpoint to flatten complex worksheets with user-specified header rows"""
#     return process_flatten_files_with_worksheets(request.flatten_requests, db)


@app.post("/unpivot-tables/")
async def unpivot_tables(request: models.UnpivotTablesRequest, db: Session = Depends(get_db)):
    """Endpoint to unpivot selected tables (worksheets) using AI-driven strategy"""
    try:
        # Get initialized LLM instances
        gpt4_llm, gemini_llm, embeddings_model = get_llms()
    except HTTPException as e:
        # Try to initialize if not already done
        try:
            gpt4_llm, gemini_llm, embeddings_model = initialize_llms()
        except Exception as init_error:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize AI models for unpivoting: {str(init_error)}"
            )
    
    # DEBUG: Log what tables actually exist in the database
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'uploaded_%';"))
            existing_tables = [row[0] for row in result.fetchall()]
            logger.info(f"DEBUG: Existing tables in database: {existing_tables}")
            logger.info(f"DEBUG: Requested table names: {request.table_names}")
            
            # Check which requested tables don't exist
            missing_tables = [table for table in request.table_names if table not in existing_tables]
            if missing_tables:
                logger.error(f"DEBUG: Missing tables: {missing_tables}")
    except Exception as debug_error:
        logger.error(f"DEBUG: Error checking existing tables: {str(debug_error)}")
    
    unpivoted_results = []
    
    for table_name in request.table_names:
        try:
            # Verify table exists and starts with 'uploaded_'
            if not table_name.startswith('uploaded_'):
                unpivoted_results.append({
                    "table_name": table_name,
                    "status": "error",
                    "reason": "Invalid table name - must be an uploaded table",
                })
                continue
            
            # Check if table exists before trying to read it
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"))
                table_exists = result.fetchone() is not None
            
            if not table_exists:
                unpivoted_results.append({
                    "table_name": table_name,
                    "status": "error",
                    "reason": f"Table {table_name} not found",
                })
                continue
            
            # Use read_sql_query instead of read_sql_table for better error handling
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", engine)
            except Exception as read_error:
                logger.error(f"Error reading table {table_name}: {str(read_error)}")
                unpivoted_results.append({
                    "table_name": table_name,
                    "status": "error",
                    "reason": f"Error reading table: {str(read_error)}",
                })
                continue
            
            if df.empty:
                logger.warning(f"No data found for table {table_name}")
                unpivoted_results.append({
                    "table_name": table_name,
                    "status": "skipped",
                    "reason": "Table is empty",
                })
                continue
            
            # Get AI unpivot strategy using GPT-4
            strategy = get_ai_unpivot_strategy(df, table_name, gpt4_llm)
            
            if not strategy['should_unpivot']:
                unpivoted_results.append({
                    "table_name": table_name,
                    "status": "skipped",
                    "reason": strategy.get('reason', 'AI determined unpivoting not needed'),
                })
                continue
            
            # Apply column renames if specified
            if 'column_renames' in strategy and strategy['column_renames']:
                df = df.rename(columns=strategy['column_renames'])
                strategy['id_columns'] = [strategy['column_renames'].get(col, col) for col in strategy['id_columns']]
                strategy['value_columns'] = [strategy['column_renames'].get(col, col) for col in strategy['value_columns']]
            
            # Validate columns exist
            available_columns = list(df.columns)
            strategy['id_columns'] = [col for col in strategy['id_columns'] if col in available_columns]
            strategy['value_columns'] = [col for col in strategy['value_columns'] if col in available_columns]
            
            # If AI didn't find good columns, use fallback logic
            if not strategy['id_columns'] or not strategy['value_columns']:
                logger.warning(f"AI strategy incomplete for {table_name}, using fallback logic")
                
                id_cols = []
                val_cols = []
                
                for col in available_columns:
                    if df[col].dtype == 'object' or df[col].nunique() <= len(df) * 0.3:
                        id_cols.append(col)
                    elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        val_cols.append(col)
                
                # Ensure we have at least some columns
                if not id_cols and available_columns:
                    id_cols = [available_columns[0]]
                if not val_cols and len(available_columns) > 1:
                    val_cols = available_columns[1:]
                
                strategy['id_columns'] = id_cols
                strategy['value_columns'] = val_cols
                
                if not val_cols:
                    unpivoted_results.append({
                        "table_name": table_name,
                        "status": "skipped",
                        "reason": "No numeric columns found for unpivoting",
                    })
                    continue
            
            # Perform unpivot using pd.melt
            logger.info(f"Unpivoting {table_name}: ID cols: {strategy['id_columns']}, Value cols: {strategy['value_columns']}")
            
            try:
                if strategy['id_columns']:
                    unpivoted_df = pd.melt(
                        df,
                        id_vars=strategy['id_columns'],
                        value_vars=strategy['value_columns'],
                        var_name=strategy['var_name'],
                        value_name=strategy['value_name']
                    )
                else:
                    # If no ID columns, create a row index
                    df_with_index = df.reset_index()
                    unpivoted_df = pd.melt(
                        df_with_index,
                        id_vars=['index'],
                        value_vars=strategy['value_columns'],
                        var_name=strategy['var_name'],
                        value_name=strategy['value_name']
                    )
                
                # Clean the unpivoted data
                initial_rows = len(unpivoted_df)
                unpivoted_df = unpivoted_df.dropna(subset=[strategy['value_name']])
                final_rows = len(unpivoted_df)
                
                logger.info(f"Unpivoting {table_name}: {initial_rows} -> {final_rows} rows after removing nulls")
                
                if unpivoted_df.empty:
                    unpivoted_results.append({
                        "table_name": table_name,
                        "status": "error",
                        "reason": "Unpivoted data is empty after removing null values",
                    })
                    continue
                    
            except Exception as melt_error:
                logger.error(f"Error during melt operation for {table_name}: {str(melt_error)}")
                unpivoted_results.append({
                    "table_name": table_name,
                    "status": "error",
                    "reason": f"Melt operation failed: {str(melt_error)}",
                })
                continue
            
            # Replace the original table with unpivoted data
            try:
                # Normalize column names before storing
                unpivoted_df = normalize_column_names(unpivoted_df)
                unpivoted_df.to_sql(table_name, engine, if_exists='replace', index=False)
                logger.info(f"Successfully replaced original table {table_name} with unpivoted data")
            except Exception as sql_error:
                logger.error(f"Error replacing original table {table_name}: {str(sql_error)}")
                unpivoted_results.append({
                    "table_name": table_name,
                    "status": "error",
                    "reason": f"Failed to replace original table: {str(sql_error)}",
                })
                continue
            
            unpivoted_results.append({
                "table_name": table_name,
                "status": "success",
                "original_shape": df.shape,
                "new_shape": unpivoted_df.shape,
                "strategy": {
                    "id_columns": strategy['id_columns'],
                    "value_columns": strategy['value_columns'],
                    "var_name": strategy['var_name'],
                    "value_name": strategy['value_name']
                },
                "description": f"Table replaced with unpivoted data. {strategy.get('description', 'Unpivoted successfully')}"
            })
            
            logger.info(f"Successfully unpivoted {table_name}: {df.shape} -> {unpivoted_df.shape}")
            
        except Exception as e:
            logger.error(f"Error unpivoting {table_name}: {str(e)}")
            unpivoted_results.append({
                "table_name": table_name,
                "status": "error",
                "reason": str(e),
            })
    
    # Prepare response
    successful_unpivots = [r for r in unpivoted_results if r['status'] == 'success']
    failed_unpivots = [r for r in unpivoted_results if r['status'] == 'error']
    skipped_unpivots = [r for r in unpivoted_results if r['status'] == 'skipped']
    
    logger.info(f"Unpivoting summary - Success: {len(successful_unpivots)}, Failed: {len(failed_unpivots)}, Skipped: {len(skipped_unpivots)}")
    
    # Include debug info in response for troubleshooting
    debug_info = {
        "requested_tables": request.table_names,
        "existing_tables": existing_tables if 'existing_tables' in locals() else [],
        "missing_tables": missing_tables if 'missing_tables' in locals() else []
    }
    
    # Replace the existing message creation logic with:
    if failed_unpivots:
        error_details = "; ".join([f"{r['table_name']}: {r['reason']}" for r in failed_unpivots])
        message = f"🤖 AI Analysis completed with some issues. Successfully analyzed: {len(successful_unpivots)}, Failed: {len(failed_unpivots)}, Skipped (no pivot structure): {len(skipped_unpivots)}. Issues: {error_details}"
    else:
        if successful_unpivots:
            message = f"🤖 AI Analysis completed! {len(successful_unpivots)} file(s) contained pivot tables and were successfully unpivoted. {len(skipped_unpivots)} file(s) were analyzed but didn't need unpivoting."
        else:
            message = f"🤖 AI Analysis completed! None of the {len(skipped_unpivots)} selected file(s) contained pivot table structures that needed unpivoting."
    
    return {
        "error": len(failed_unpivots) > 0,
        "message": message,
        "results": unpivoted_results,
        "debug": debug_info,
        "summary": {
            "total_tables": len(request.table_names),
            "successful": len(successful_unpivots),
            "failed": len(failed_unpivots),
            "skipped": len(skipped_unpivots)
        }
    }


@app.post("/ask-question-iterative/")
async def ask_question_iterative(request: models.IterativeQuestionRequest, db: Session = Depends(get_db)):
    """Simplified iterative query optimization - just pass history properly"""
    try:
        # Get initialized LLM instances
        gpt4_llm, gemini_llm, embeddings_model = get_llms()
        
        # Get all tables information from database
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'uploaded_%';"))
            table_names = [row[0] for row in result.fetchall()]
        
        if not table_names:
            return {
                "error": True,
                "message": "No tables found in database. Please upload files first.",
                "sql_query": "",
                "data": [],
                "confidence_score": 0
            }
        
        # Create table_dict and table_samples for your function
        table_dict = {}
        table_samples = {}
        
        for table_name in table_names:
            try:
                df = pd.read_sql_table(table_name, engine)
                if not df.empty:
                    # Extract filename from table name
                    filename = table_name.replace('uploaded_', '').replace('_', ' ')
                    table_dict[filename] = table_name
                    table_samples[table_name] = df.head(5)  # Sample data
            except Exception as e:
                logger.error(f"Error reading table {table_name}: {str(e)}")
                continue
        
        # Create simple memory object from previous attempts
        query_memory = None
        if request.previous_attempts:
            # Create a simple memory object that your existing generate_sql_prompt function can use
            class SimpleMemory:
                def __init__(self, attempts):
                    self.buffer = self._create_buffer_from_attempts(attempts)
                
                def _create_buffer_from_attempts(self, attempts):
                    buffer_parts = []
                    buffer_parts.append("Previous attempts:")
                    
                    for attempt in attempts:
                        query = attempt.get('query', 'N/A')
                        confidence = attempt.get('confidence', 0)
                        attempt_num = attempt.get('attempt', 0)
                        error = attempt.get('error', False)
                        
                        if error:
                            buffer_parts.append(f"Attempt {attempt_num}: FAILED - {query}")
                            buffer_parts.append(f"  Error: {attempt.get('message', 'Unknown error')}")
                        else:
                            buffer_parts.append(f"Attempt {attempt_num}: SUCCESS ({confidence}% confidence) - {query}")
                            buffer_parts.append(f"  Results: {attempt.get('data', 0)} rows")
                    
                    buffer_parts.append("")
                    buffer_parts.append("Generate a DIFFERENT approach than the previous attempts above.")
                    buffer_parts.append("Try alternative table joins, different aggregations, or different filtering methods.")
                    
                    return "\n".join(buffer_parts)
            
            query_memory = SimpleMemory(request.previous_attempts)
        
        # Generate SQL query using your EXISTING function - just pass the memory now
        sql_query = generate_sql_prompt(
            user_question=request.question,
            table_dict=table_dict,
            table_samples=table_samples,
            llm=gpt4_llm,
            query_memory=query_memory  # Now actually pass the memory instead of None
        )
        
        # Calculate confidence score using the existing confidence calculator
        confidence_scores = {}
        overall_confidence = 50  # Default fallback
        
        try:
            confidence_scores = calculate_query_confidence(
                user_query=request.question,
                sql_query=sql_query,
                llm=gemini_llm,
                embeddings=embeddings_model
            )
            overall_confidence = confidence_scores.get('overall', 50)
            
        except Exception as conf_error:
            logger.warning(f"Confidence calculation failed: {str(conf_error)}")
            confidence_scores = {
                'semantic': 50,
                'execution': 50,
                'llm_comparison': 50,
                'sql_quality': 50,
                'overall': 50,
                'error': str(conf_error)
            }
            overall_confidence = 50
        
        # Execute the generated SQL query
        try:
            logger.info(f"Executing SQL query (iteration {request.iteration}): {sql_query}")
            
            # Check if query is valid
            if "SQL Query can't be generated" in sql_query or "Error generating query" in sql_query:
                return {
                    "error": True,
                    "message": sql_query,
                    "sql_query": sql_query,
                    "data": [],
                    "confidence_score": overall_confidence,
                    "confidence_breakdown": confidence_scores,
                    "iteration": request.iteration
                }
            
            # Execute the query
            df_result = pd.read_sql_query(text(sql_query), engine)
            data = df_result.to_dict(orient="records")
            
            return {
                "error": False,
                "message": "Query executed successfully",
                "sql_query": sql_query,
                "data": data,
                "row_count": len(data),
                "columns": list(df_result.columns) if not df_result.empty else [],
                "question": request.question,
                "confidence_score": overall_confidence,
                "confidence_breakdown": confidence_scores,
                "iteration": request.iteration
            }
            
        except Exception as sql_error:
            return {
                "error": True,
                "message": f"Error executing SQL query: {str(sql_error)}",
                "sql_query": sql_query,
                "data": [],
                "confidence_score": overall_confidence,
                "confidence_breakdown": confidence_scores,
                "iteration": request.iteration,
                "question": request.question
            }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing iterative question: {str(e)}")
        return {
            "error": True,
            "message": f"Failed to process question: {str(e)}",
            "sql_query": "",
            "data": [],
            "confidence_score": 0,
            "iteration": request.iteration,
            "question": request.question
        }


@app.get("/health/llm")
async def check_llm_health():
    """Check if LLMs are initialized and working"""
    try:
        gpt4_llm, gemini_llm, embeddings_model = get_llms()
        return {
            "status": "initialized",
            "message": "All LLMs initialized and ready",
            "models": {
                "gpt4": gpt4_llm is not None,
                "gemini": gemini_llm is not None,
                "embeddings": embeddings_model is not None
            }
        }
    except HTTPException:
        return {
            "status": "not_initialized",
            "message": "LLMs not yet initialized. Upload files first.",
            "models": {
                "gpt4": False,
                "gemini": False,
                "embeddings": False
            }
        }

@app.post("/initialize-llms/")
async def force_initialize_llms():
    """Force initialize LLMs without uploading files (for testing)"""
    try:
        gpt4, gemini, emb = initialize_llms()
        return {
            "success": True,
            "message": "LLMs initialized successfully",
            "models": {
                "gpt4": gpt4 is not None,
                "gemini": gemini is not None,
                "embeddings": emb is not None
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to initialize LLMs: {str(e)}",
            "models": {
                "gpt4": False,
                "gemini": False,
                "embeddings": False
            }
        }


@app.get("/debug/table/{table_name}")
async def inspect_table_structure(table_name: str):
    """Debug endpoint to inspect a specific table structure"""
    import pandas as pd
    from database import engine
    
    try:
        df = pd.read_sql_table(table_name, engine)
        
        if df.empty:
            return {"error": f"Table {table_name} is empty"}
        
        column_analysis = []
        for col in df.columns:
            col_data = df[col]
            
            analysis = {
                "column_name": col,
                "dtype": str(col_data.dtype),
                "unique_count": int(col_data.nunique()),
                "null_count": int(col_data.isnull().sum()),
                "total_rows": len(col_data),
                "sample_values": []
            }
            
            sample_values = col_data.dropna().unique()[:5]
            for val in sample_values:
                if pd.isna(val):
                    analysis["sample_values"].append(None)
                else:
                    analysis["sample_values"].append(str(val))
            
            if col_data.dtype == 'object':
                numeric_count = 0
                for val in col_data.dropna():
                    try:
                        float(val)
                        numeric_count += 1
                    except:
                        pass
                analysis["numeric_convertible"] = numeric_count
                analysis["numeric_percentage"] = numeric_count / len(col_data.dropna()) if len(col_data.dropna()) > 0 else 0
            
            column_analysis.append(analysis)
        
        return {
            "table_name": table_name,
            "shape": df.shape,
            "columns": column_analysis,
            "sample_data": df.head(5).to_dict('records')
        }
        
    except Exception as e:
        return {"error": f"Failed to inspect table {table_name}: {str(e)}"}

@app.get("/debug/tables")
async def list_database_tables():
    """Debug endpoint to list all tables in the database"""
    try:
        import pandas as pd
        from database import engine
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
            table_names = [row[0] for row in result.fetchall()]
        
        table_info = []
        for table_name in table_names:
            try:
                df = pd.read_sql_table(table_name, engine)
                table_info.append({
                    "table_name": table_name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns)
                })
            except Exception as e:
                table_info.append({
                    "table_name": table_name,
                    "error": str(e)
                })
        
        return {
            "total_tables": len(table_names),
            "tables": table_info
        }
    except Exception as e:
        return {
            "error": str(e),
            "tables": []
        }
        
@app.delete("/delete-table/{table_name}")
async def delete_table(table_name: str, db: Session = Depends(get_db)):
    """Delete a specific uploaded table and its associated file record"""
    try:
        # Security check: only allow tables with 'uploaded_' prefix
        if not table_name.startswith('uploaded_'):
            raise HTTPException(
                status_code=403, 
                detail="Access denied: Only uploaded tables can be deleted"
            )
        
        # Check if table exists in database
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"))
            table_exists = result.fetchone() is not None
        
        if not table_exists:
            raise HTTPException(
                status_code=404,
                detail=f"Table '{table_name}' not found"
            )
        
        # Extract filename from table name for File record lookup
        # Convert 'uploaded_filename' back to 'filename.ext' format
        filename_base = table_name.replace('uploaded_', '').replace('_', ' ')
        
        # Try to find the file record in the database
        # We'll search for files that could match this table name
        file_record = None
        
        # First try exact match variations
        possible_filenames = [
            f"{filename_base}.csv",
            f"{filename_base}.xlsx",
            f"{filename_base}.xls"
        ]
        
        for possible_name in possible_filenames:
            file_record = db.query(File).filter(File.filename == possible_name).first()
            if file_record:
                break
        
        # If no exact match, try fuzzy matching (in case of filename variations)
        if not file_record:
            all_files = db.query(File).all()
            for file in all_files:
                # Create table name from this file and see if it matches
                test_table_name = create_table_name_from_filename(file.filename)
                if test_table_name == table_name:
                    file_record = file
                    break
        
        # Delete the SQLite table
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            conn.commit()
        
        # Delete the file record if found
        if file_record:
            db.delete(file_record)
            db.commit()
            logger.info(f"Deleted file record: {file_record.filename}")
        else:
            logger.warning(f"No file record found for table {table_name}")
        
        logger.info(f"Successfully deleted table: {table_name}")
        
        return {
            "error": False,
            "message": f"Table '{table_name}' deleted successfully",
            "table_name": table_name,
            "file_record_deleted": file_record is not None
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error deleting table {table_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete table: {str(e)}"
        )

@app.post("/execute-custom-query/")
async def execute_custom_query(request: models.CustomQueryRequest, db: Session = Depends(get_db)):
    """Endpoint to execute user-modified SQL queries"""
    try:
        # Validate that the query is not empty
        if not request.sql_query.strip():
            return {
                "error": True,
                "message": "SQL query cannot be empty"
            }
        
        # Security check: only allow SELECT statements
        query_upper = request.sql_query.strip().upper()
        if not query_upper.startswith('SELECT'):
            return {
                "error": True,
                "message": "Only SELECT queries are allowed for security reasons"
            }
        
        # Additional security: prevent dangerous SQL operations
        dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE']
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return {
                    "error": True,
                    "message": f"Query contains forbidden keyword: {keyword}"
                }
        
        logger.info(f"Executing custom SQL query: {request.sql_query}")
        
        # Execute the custom SQL query
        df_result = pd.read_sql_query(text(request.sql_query), engine)
        data = df_result.to_dict(orient="records")
        
        return {
            "error": False,
            "message": "Custom query executed successfully",
            "sql_query": request.sql_query,
            "data": data,
            "row_count": len(data),
            "columns": list(df_result.columns) if not df_result.empty else [],
            "question": request.question
        }
        
    except Exception as sql_error:
        logger.error(f"Error executing custom SQL query: {str(sql_error)}")
        return {
            "error": True,
            "message": f"Error executing custom SQL query: {str(sql_error)}",
            "sql_query": request.sql_query,
            "data": [],
            "columns": [],
            "question": request.question
        }