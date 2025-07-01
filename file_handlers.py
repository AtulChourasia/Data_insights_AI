import pandas as pd
from sqlalchemy.orm import Session
from fastapi import HTTPException, UploadFile
from io import BytesIO
from typing import List
import re
import json
import logging
from models import File
from database import engine
from backend import is_complex_sheet, setup_azure_openai, preprocess_column_name  
from Class_ExcelFlattener_V3 import ExcelFlattener
from token_estimation import check_file_token_limit
import numpy as np
from sqlalchemy import text 

from header_detection_helper import detect_header_rows_automatically

logger = logging.getLogger(__name__)

# Global storage for temporarily uploaded files
temp_file_storage = {}

# Global variables for LLM instances
llm_gpt4 = None
llm_gemini = None
embeddings = None
llm_initialized = False


def create_table_name_from_filename_and_sheet(filename, sheet_name):
    """Create a clean table name from filename and sheet name"""
    file_base = filename.split('.')[0]
    clean_file = re.sub(r'[^a-zA-Z0-9_]', '_', file_base.lower())
    clean_sheet = re.sub(r'[^a-zA-Z0-9_]', '_', sheet_name.lower()) if sheet_name else 'sheet1'
    return f"uploaded_{clean_file}_{clean_sheet}"

def get_excel_worksheets(file_content):
    """Get all worksheet names from an Excel file"""
    try:
        with pd.ExcelFile(BytesIO(file_content)) as xls:
            return xls.sheet_names
    except Exception as e:
        logger.error(f"Error reading Excel worksheets: {str(e)}")
        return []

def read_excel_worksheet(file_content, sheet_name):
    """Read a specific worksheet from Excel file"""
    try:
        return pd.read_excel(BytesIO(file_content), sheet_name=sheet_name, header=None)
    except Exception as e:
        logger.error(f"Error reading worksheet {sheet_name}: {str(e)}")
        return None


async def process_uploaded_files_with_worksheets(files: List[UploadFile], db: Session):
    """Enhanced version with automatic complex file processing"""
    # Initialize LLMs on first file upload
    try:
        initialize_llms()
    except Exception as e:
        logger.warning(f"LLM initialization failed during upload: {str(e)}")
    
    # Token checking logic (UNCHANGED - keep existing code)
    total_tokens = 0
    file_token_data = []
    
    for file in files:
        try:
            is_valid, message, token_count, token_breakdown = await check_file_token_limit(file, max_tokens=float('inf'))
            file_token_data.append({
                "filename": file.filename,
                "tokens": token_count,
                "breakdown": token_breakdown
            })
            total_tokens += token_count
            await file.seek(0)
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error analyzing file '{file.filename}': {str(e)}"
            )
    
    # Token limit check (UNCHANGED - keep existing code)
    max_tokens = 100000
    if total_tokens > max_tokens:
        percentage = (total_tokens / max_tokens) * 100
        error_message = f"❌ Combined files exceed token limit: {total_tokens:,} tokens ({percentage:.1f}% of {max_tokens:,}). Please reduce the number of files or file sizes."
        
        return {
            "error": True,
            "message": error_message,
            "total_tokens": total_tokens,
            "max_tokens": max_tokens,
            "file_breakdown": file_token_data
        }
    
    # Process files - MODIFIED LOGIC STARTS HERE
    all_processed_files = []
    auto_processing_summary = {
        "simple_files": 0,
        "complex_files_auto_processed": 0,
        "total_files": len(files)
    }
    
    for file in files:
        try:
            file_content = await file.read()
            
            # Handle different file types (UNCHANGED)
            if file.filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(file_content), header=None)
                worksheets_data = [{"sheet_name": "Sheet1", "dataframe": df}]
            elif file.filename.endswith((".xlsx", ".xls")):
                sheet_names = get_excel_worksheets(file_content)
                if not sheet_names:
                    raise HTTPException(status_code=400, detail=f"Could not read worksheets from {file.filename}")
                
                worksheets_data = []
                for sheet_name in sheet_names:
                    df = read_excel_worksheet(file_content, sheet_name)
                    if df is not None and not df.empty:
                        worksheets_data.append({"sheet_name": sheet_name, "dataframe": df})
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}")
            
            # Process each worksheet - MAIN CHANGE HERE
            file_worksheets = []
            file_total_tokens = 0
            
            for worksheet_info in worksheets_data:
                sheet_name = worksheet_info["sheet_name"]
                df = worksheet_info["dataframe"]
                
                # Check if complex
                is_complex = is_complex_sheet(df)
                
                if is_complex:
                    # AUTO-PROCESS COMPLEX FILES - NO UI NEEDED
                    logger.info(f"Auto-processing complex worksheet: {sheet_name}")
                    
                    # Auto-detect header rows
                    optimal_headers = detect_header_rows_automatically(df)
                    
                    # Auto-flatten
                    flattener = ExcelFlattener(df)
                    flattened_df = flattener.flatten(h=optimal_headers, method='wide')
                    
                    # Clean and store
                    flattened_df = flattened_df.replace([float('inf'), float('-inf')], None).fillna(value='')
                    flattened_df = normalize_column_names(flattened_df)
                    
                    table_name = create_table_name_from_filename_and_sheet(file.filename, sheet_name)
                    flattened_df.to_sql(table_name, engine, if_exists='replace', index=False)
                    
                    data = flattened_df.to_dict(orient="records")
                    worksheet_tokens = len(str(data)) // 4
                    
                    file_worksheets.append({
                        "sheet_name": sheet_name,
                        "table_name": table_name,
                        "data": data,
                        "tokens": worksheet_tokens,
                        "rows": len(flattened_df),
                        "columns": len(flattened_df.columns),
                        "is_complex": True,
                        "auto_processed": True,
                        "header_rows_used": optimal_headers
                    })
                    
                    auto_processing_summary["complex_files_auto_processed"] += 1
                    
                else:
                    # Simple worksheet - process normally (UNCHANGED)
                    new_header = df.iloc[0]

# 2) Create a new DataFrame with the rest of the rows and new header
                    df_with_header = df[1:].copy()  # skip the first row
                    df_with_header.columns = new_header  # set header

                    # 3) Reset index to keep it clean
                    df_with_header.reset_index(drop=True, inplace=True)
                                    # Process simple file immediately
                    df = df_with_header

                    df = df.replace([float('inf'), float('-inf')], None).fillna(value='')
                    df = normalize_column_names(df)
                    
                    table_name = create_table_name_from_filename_and_sheet(file.filename, sheet_name)
                    df.to_sql(table_name, engine, if_exists='replace', index=False)
                    
                    data = df.to_dict(orient="records")
                    worksheet_tokens = len(str(data)) // 4
                    
                    file_worksheets.append({
                        "sheet_name": sheet_name,
                        "table_name": table_name,
                        "data": data,
                        "tokens": worksheet_tokens,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "is_complex": False,
                        "auto_processed": False
                    })
                    
                    auto_processing_summary["simple_files"] += 1
                
                file_total_tokens += worksheet_tokens
            
            # Add file to results
            file_token_info = next((item for item in file_token_data if item["filename"] == file.filename), {"tokens": 0})
            all_processed_files.append({
                "filename": file.filename,
                "worksheets": file_worksheets,
                "total_tokens": file_token_info["tokens"],
                "total_worksheets": len(file_worksheets)
            })
            
            # Store file metadata (UNCHANGED)
            existing_file = db.query(File).filter(File.filename == file.filename).first()
            if existing_file:
                existing_file.content = file_content
            else:
                db_file = File(filename=file.filename, content=file_content)
                db.add(db_file)
            db.commit()
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error with file '{file.filename}': {str(e)}")
    
    # Create success message
    percentage = (total_tokens / max_tokens) * 100
    if auto_processing_summary["complex_files_auto_processed"] > 0:
        success_message = f"✅ Files uploaded successfully: {total_tokens:,} tokens ({percentage:.1f}% of {max_tokens:,}) | 🤖 {auto_processing_summary['complex_files_auto_processed']} complex file(s) auto-processed"
    else:
        success_message = f"✅ Files uploaded successfully: {total_tokens:,} tokens ({percentage:.1f}% of {max_tokens:,})"
    
    return {
        "error": False,
        "message": success_message,
        "files": all_processed_files,
        "total_tokens": total_tokens,
        "max_tokens": max_tokens,
        "file_breakdown": file_token_data,
        "auto_processing_summary": auto_processing_summary,
        # "skip_pivot_ui": True  # SKIP PIVOT UI ENTIRELY
    }

def process_flatten_files_with_worksheets(flatten_requests, db: Session):
    """Enhanced version of process_flatten_files with worksheet support"""
    flattened_data = []
    total_tokens = 0
    file_token_data = []
    
    for flatten_req in flatten_requests:
        filename = flatten_req.filename
        sheet_name = getattr(flatten_req, 'sheet_name', None)  # New field needed
        header_rows = flatten_req.header_rows
        
        # Create worksheet key
        worksheet_key = f"{filename}_{sheet_name}" if sheet_name else filename
        
        if worksheet_key not in temp_file_storage:
            raise HTTPException(
                status_code=400, 
                detail=f"Worksheet '{sheet_name}' from file '{filename}' not found in temporary storage. Please re-upload."
            )
        
        try:
            # Get the stored DataFrame and file content
            stored_data = temp_file_storage[worksheet_key]
            df = stored_data['dataframe']
            file_content = stored_data['content']
            actual_sheet_name = stored_data['sheet_name']
            
            # Apply flattening using ExcelFlattener class with DataFrame directly
            flattener = ExcelFlattener(df)
            flattened_df = flattener.flatten(h=header_rows, method='wide')
            
            # Replace invalid values and normalize column names
            flattened_df = flattened_df.replace([float('inf'), float('-inf')], None).fillna(value='')
            flattened_df = normalize_column_names(flattened_df)

            # Store DataFrame as a separate table for SQL queries
            table_name = create_table_name_from_filename_and_sheet(filename, actual_sheet_name)
            flattened_df.to_sql(table_name, engine, if_exists='replace', index=False)

            # Convert DataFrame to JSON for response
            data = flattened_df.to_dict(orient="records")
            
            # Calculate tokens for flattened data (simplified)
            estimated_tokens = len(str(data)) // 4
            total_tokens += estimated_tokens
            
            file_token_data.append({
                "filename": filename,
                "sheet_name": actual_sheet_name,
                "tokens": estimated_tokens
            })
            
            flattened_data.append({
                "filename": filename,
                "sheet_name": actual_sheet_name,
                "table_name": table_name,
                "data": data,
                "tokens": estimated_tokens,
                "rows": len(flattened_df),
                "columns": len(flattened_df.columns),
                "is_complex": True  # Was complex, now flattened
            })
            
            # Clean up temporary storage
            del temp_file_storage[worksheet_key]

        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error flattening worksheet '{sheet_name}' from file '{filename}': {str(e)}"
            )

    max_tokens = 100000
    percentage = (total_tokens / max_tokens) * 100
    success_message = f"✅ Complex worksheets flattened and uploaded successfully: {total_tokens:,} tokens ({percentage:.1f}% of {max_tokens:,})"
    
    return {
        "error": False,
        "message": success_message,
        "files": flattened_data,
        "total_tokens": total_tokens,
        "max_tokens": max_tokens,
        "file_breakdown": file_token_data
    }

def delete_table_from_database(table_name, db_session, engine):
    """
    Delete a table from SQLite database and its associated file record
    
    Args:
        table_name: Name of the table to delete
        db_session: SQLAlchemy database session
        engine: SQLAlchemy engine for direct SQL execution
    
    Returns:
        dict: Result of the deletion operation
    """
    try:
        # Verify table exists
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"))
            table_exists = result.fetchone() is not None
        
        if not table_exists:
            return {
                "error": True,
                "message": f"Table '{table_name}' does not exist"
            }
        
        # Extract original filename from table name
        filename_base = table_name.replace('uploaded_', '').replace('_', ' ')
        
        # Find corresponding file record
        file_record = None
        possible_extensions = ['.csv', '.xlsx', '.xls']
        
        for ext in possible_extensions:
            possible_filename = f"{filename_base}{ext}"
            file_record = db_session.query(File).filter(File.filename == possible_filename).first()
            if file_record:
                break
        
        # Delete the table from SQLite
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE {table_name}"))
            conn.commit()
        
        # Delete file record if found
        if file_record:
            db_session.delete(file_record)
            db_session.commit()
        
        return {
            "error": False,
            "message": f"Successfully deleted table '{table_name}'" + (
                f" and file record '{file_record.filename}'" if file_record else ""
            ),
            "file_record_deleted": file_record is not None
        }
        
    except Exception as e:
        return {
            "error": True,
            "message": f"Failed to delete table: {str(e)}"
        }

def get_table_file_mapping(db_session, engine):
    """
    Get mapping of all uploaded tables to their original filenames
    
    Returns:
        dict: Mapping of table_name -> filename
    """
    try:
        mapping = {}
        
        # Get all uploaded tables
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'uploaded_%';"))
            table_names = [row[0] for row in result.fetchall()]
        
        # Get all file records
        file_records = db_session.query(File).all()
        
        # Create mapping
        for table_name in table_names:
            # Try to find corresponding file
            for file_record in file_records:
                expected_table_name = create_table_name_from_filename(file_record.filename)
                if expected_table_name == table_name:
                    mapping[table_name] = file_record.filename
                    break
        
        return mapping
        
    except Exception as e:
        logger.error(f"Error creating table-file mapping: {str(e)}")
        return {}

def create_table_name_from_filename(filename):
    """Create a clean table name from filename"""
    name = filename.split('.')[0]
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
    return f"uploaded_{clean_name}"

def normalize_column_names(df):
    """Normalize column names using the preprocess_column_name function"""
    # Create a copy to avoid modifying original
    df_copy = df.copy()
    
    # Apply preprocess_column_name to all columns
    new_columns = []
    for col in df_copy.columns:
        normalized_col = preprocess_column_name(col)
        
        # Handle duplicates
        original_col = normalized_col
        counter = 1
        while normalized_col in new_columns:
            normalized_col = f"{original_col}_{counter}"
            counter += 1
        
        new_columns.append(normalized_col)
    
    # Rename columns
    df_copy.columns = new_columns
    logger.info(f"Normalized columns: {dict(zip(df.columns, new_columns))}")
    
    return df_copy

def store_dataframe_as_table(df, filename, engine):
    """Store DataFrame as a separate table in the database with normalized column names"""
    table_name = create_table_name_from_filename(filename)
    
    # Normalize column names before storing
    df_normalized = normalize_column_names(df)
    
    df_normalized.to_sql(table_name, engine, if_exists='replace', index=False)
    return table_name

def initialize_llms():
    """Initialize LLMs once when first file is uploaded"""
    global llm_gpt4, llm_gemini, embeddings, llm_initialized
    
    if llm_initialized:
        return llm_gpt4, llm_gemini, embeddings
    
    try:
        logger.info("Initializing LLMs for the first time...")
        llm_gpt4, embeddings, llm_gemini = setup_azure_openai()
        llm_initialized = True
        logger.info("LLMs initialized successfully")
        return llm_gpt4, llm_gemini, embeddings
    except Exception as e:
        logger.error(f"Failed to initialize LLMs: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to initialize AI models: {str(e)}"
        )

def get_llms():
    """Get initialized LLM instances"""
    global llm_gpt4, llm_gemini, embeddings
    
    if not llm_initialized:
        raise HTTPException(
            status_code=500, 
            detail="LLMs not initialized. Please upload files first."
        )
    
    return llm_gpt4, llm_gemini, embeddings

def get_ai_unpivot_strategy(df, table_name, llm):
    """Use AI to determine the best unpivot strategy for a table"""
    logger.info(f"Starting AI unpivot strategy analysis for table: {table_name}")
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"DataFrame columns: {list(df.columns)}")
    logger.info(f"DataFrame dtypes: {dict(df.dtypes)}")
    
    # Create a sample of the data for AI analysis
    logger.info("Creating sample data for AI analysis...")
    sample_data = df.head(10).to_string()
    logger.debug(f"Sample data:\n{sample_data}")
    
    columns_info = []
    logger.info("Analyzing each column...")
    
    for col in df.columns:
        logger.debug(f"Analyzing column: {col}")
        
        dtype = str(df[col].dtype)
        unique_count = int(df[col].nunique())
        null_count = int(df[col].isnull().sum())
        
        logger.debug(f"  - dtype: {dtype}, unique: {unique_count}, nulls: {null_count}")
        
        # Get sample values and convert numpy types to Python types
        sample_values = df[col].dropna().unique()[:5]
        converted_values = []
        for val in sample_values:
            if isinstance(val, (np.integer, np.int64)):
                converted_values.append(int(val))
            elif isinstance(val, (np.floating, np.float64)):
                converted_values.append(float(val))
            elif pd.isna(val):
                converted_values.append(None)
            else:
                converted_values.append(str(val))
        
        logger.debug(f"  - sample values: {converted_values}")
        
        columns_info.append({
            'column': col,
            'dtype': dtype,
            'unique_count': unique_count,
            'null_count': null_count,
            'sample_values': str(converted_values)
        })
    
    logger.info(f"Column analysis complete. Total columns analyzed: {len(columns_info)}")
    
    prompt = f"""
    Analyze this table and determine the best unpivoting strategy.
    
    Table Name: {table_name}
    Shape: {df.shape[0]} rows × {df.shape[1]} columns
    
    Columns Information:
    {json.dumps(columns_info, indent=2)}
    
    Sample Data:
    {sample_data}
    
    Determine:
    1. Should this table be unpivoted? (Look for wide-format data with metrics spread across columns)
    2. Which columns should be ID columns (remain as identifiers)?
    3. Which columns should be unpivoted (value columns)?
    4. What should the unpivoted column names be?
    5. What should the new table name be?
    
    Return a JSON object with this exact structure:
    {{
        "should_unpivot": true/false,
        "reason": "explanation of decision",
        "id_columns": ["col1", "col2"],
        "value_columns": ["col3", "col4", "col5"],
        "var_name": "metric_name",
        "value_name": "metric_value",
        "new_table_name": "suggested_table_name",
        "column_renames": {{"old_name": "new_name"}},
        "description": "Brief description of the transformation"
    }}
    
    Guidelines:
    - ID columns are typically: names, codes, categories, dates, or low-cardinality string columns
    - Value columns are typically: numeric columns representing metrics, especially if column names suggest time periods or categories
    - Use SQL-friendly names (lowercase, underscores, no spaces)
    - If columns represent dates/months/years, var_name could be "period" or "date"
    - If columns represent products/categories, var_name could be "category" or "product"
    - Make column names descriptive but concise
    
    RETURN ONLY THE JSON OBJECT, NO ADDITIONAL TEXT.
    """
    
    logger.info("Sending prompt to LLM...")
    logger.debug(f"Prompt length: {len(prompt)} characters")
    
    try:
        logger.info("Calling LLM for strategy analysis...")
        response = llm.invoke(prompt).content.strip()
        logger.info(f"LLM response received. Length: {len(response)} characters")
        logger.debug(f"Raw LLM response: {response}")
        
        # Clean up response to ensure it's valid JSON
        logger.info("Cleaning up LLM response...")
        original_response = response
        
        if response.startswith("```json"):
            response = response.split("```json")[1]
            logger.debug("Removed ```json prefix")
        if response.startswith("```"):
            response = response.split("```")[1]
            logger.debug("Removed ``` prefix")
        if "```" in response:
            response = response.split("```")[0]
            logger.debug("Removed ``` suffix")
        
        if response != original_response:
            logger.debug(f"Cleaned response: {response}")
        
        logger.info("Parsing JSON response...")
        strategy = json.loads(response)
        logger.info("JSON parsing successful")
        logger.debug(f"Parsed strategy: {strategy}")
        
        # Validate the response
        logger.info("Validating strategy response...")
        required_keys = ['should_unpivot', 'id_columns', 'value_columns', 'var_name', 'value_name', 'new_table_name']
        missing_keys = []
        for key in required_keys:
            if key not in strategy:
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"Missing required keys in strategy: {missing_keys}")
            raise ValueError(f"Missing required keys: {missing_keys}")
        else:
            logger.info("All required keys present in strategy")
        
        # Log the AI's recommendations
        logger.info(f"AI recommendation - Should unpivot: {strategy['should_unpivot']}")
        logger.info(f"AI recommendation - ID columns: {strategy['id_columns']}")
        logger.info(f"AI recommendation - Value columns: {strategy['value_columns']}")
        logger.info(f"AI recommendation - Reason: {strategy.get('reason', 'No reason provided')}")
        
        logger.info(f"Final ID columns: {strategy['id_columns']}")
        logger.info(f"Final value columns: {strategy['value_columns']}")
        
        # If no valid columns found, don't unpivot
        if not strategy['id_columns'] or not strategy['value_columns']:
            logger.warning("No valid ID or value columns found after validation")
            strategy['should_unpivot'] = False
            strategy['reason'] = "Could not identify valid ID or value columns"
            logger.info("Strategy updated to not unpivot due to missing columns")
        
        logger.info("AI strategy analysis completed successfully")
        return strategy
        
    except json.JSONDecodeError as json_error:
        logger.error(f"JSON parsing failed for {table_name}: {str(json_error)}")
        logger.error(f"Problematic response: {response}")
        logger.info("Falling back to heuristic strategy...")
    except Exception as e:
        logger.error(f"AI unpivot strategy failed for {table_name}: {str(e)}")
        logger.info("Falling back to heuristic strategy...")
    
    # Fallback strategy if AI fails
    logger.info("Starting fallback heuristic analysis...")
    
    id_columns = []
    value_columns = []
    
    logger.info("Analyzing columns for fallback strategy...")
    for col in df.columns:
        col_dtype = df[col].dtype
        col_unique = df[col].nunique()
        col_total = len(df)
        uniqueness_ratio = col_unique / col_total if col_total > 0 else 0
        
        logger.debug(f"Column {col}: dtype={col_dtype}, unique={col_unique}, ratio={uniqueness_ratio:.3f}")
        
        if col_dtype == 'object' and uniqueness_ratio < 0.5:
            id_columns.append(col)
            logger.debug(f"  -> Added to ID columns (object type with low uniqueness)")
        elif col_dtype in ['int64', 'float64']:
            value_columns.append(col)
            logger.debug(f"  -> Added to value columns (numeric type)")
        else:
            logger.debug(f"  -> Not classified")
    
    logger.info(f"Heuristic analysis - ID columns: {id_columns}")
    logger.info(f"Heuristic analysis - Value columns: {value_columns}")
    
    # Ensure we have at least some columns
    if not id_columns and len(df.columns) > 0:
        id_columns = [df.columns[0]]
        logger.info(f"No ID columns found, using first column as ID: {id_columns}")
    
    # If still no value columns, try a more lenient approach
    if not value_columns:
        logger.warning("No numeric value columns found, trying lenient approach...")
        for col in df.columns:
            if col not in id_columns:
                # Check if column contains mostly numeric-convertible values
                try:
                    numeric_count = 0
                    total_non_null = 0
                    for val in df[col].dropna():
                        total_non_null += 1
                        try:
                            float(val)
                            numeric_count += 1
                        except:
                            pass
                    
                    if total_non_null > 0 and (numeric_count / total_non_null) > 0.7:
                        value_columns.append(col)
                        logger.info(f"Added {col} to value columns (70%+ numeric convertible)")
                except Exception as conv_error:
                    logger.debug(f"Error checking numeric conversion for {col}: {conv_error}")
    
    should_unpivot = len(value_columns) > 1
    logger.info(f"Fallback decision - Should unpivot: {should_unpivot} (based on {len(value_columns)} value columns)")
    
    fallback_strategy = {
        'should_unpivot': should_unpivot,
        'reason': f'Fallback strategy used. Found {len(id_columns)} ID columns and {len(value_columns)} value columns',
        'id_columns': id_columns,
        'value_columns': value_columns,
        'var_name': 'metric',
        'value_name': 'value',
        'new_table_name': f"{table_name}_unpivoted",
        'description': 'Standard unpivot transformation using heuristic analysis'
    }
    
    logger.info(f"Fallback strategy completed: {fallback_strategy}")
    return fallback_strategy

async def process_uploaded_files(files: List[UploadFile], db: Session):
    """Process uploaded files and return results"""
    # Initialize LLMs on first file upload
    try:
        initialize_llms()
    except Exception as e:
        logger.warning(f"LLM initialization failed during upload: {str(e)}")
    
    # Check cumulative token count across all files
    total_tokens = 0
    file_token_data = []
    
    for file in files:
        try:
            is_valid, message, token_count, token_breakdown = await check_file_token_limit(file, max_tokens=float('inf'))
            file_token_data.append({
                "filename": file.filename,
                "tokens": token_count,
                "breakdown": token_breakdown
            })
            total_tokens += token_count
            await file.seek(0)  # Reset file pointer
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error analyzing file '{file.filename}': {str(e)}"
            )
    
    # Check if cumulative tokens exceed limit
    max_tokens = 100000
    if total_tokens > max_tokens:
        percentage = (total_tokens / max_tokens) * 100
        error_message = f"❌ Combined files exceed token limit: {total_tokens:,} tokens ({percentage:.1f}% of {max_tokens:,}). Please reduce the number of files or file sizes."
        
        return {
            "error": True,
            "message": error_message,
            "total_tokens": total_tokens,
            "max_tokens": max_tokens,
            "file_breakdown": file_token_data
        }
    
    # Process files and check for complex ones
    complex_files = []
    simple_files_data = []
    
    for file in files:
        try:
            file_content = await file.read()

            # Parse the file content as a DataFrame
            if file.filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(file_content),header=None)
            elif file.filename.endswith(".xlsx"):
                df = pd.read_excel(BytesIO(file_content), header=None)
            else:
                raise HTTPException(
                    status_code=400, detail=f"Unsupported file format: {file.filename}"
                )

            # Check if the file is complex
            is_complex = is_complex_sheet(df)

            if is_complex:
                # Store file temporarily for later flattening
                temp_file_storage[file.filename] = {
                    'content': file_content,
                    'dataframe': df
                }
                
                # Get raw data for preview (first 15 rows)
                raw_data = df.head(15).fillna('').astype(str).values.tolist()
                
                complex_files.append({
                    "filename": file.filename,
                    "raw_data": raw_data,
                    "shape": df.shape
                })
            else:

                new_header = df.iloc[0]

# 2) Create a new DataFrame with the rest of the rows and new header
                df_with_header = df[1:].copy()  # skip the first row
                df_with_header.columns = new_header  # set header

                # 3) Reset index to keep it clean
                df_with_header.reset_index(drop=True, inplace=True)
                                # Process simple file immediately
                df = df_with_header
                df = df.replace([float('inf'), float('-inf')], None).fillna(value='')
                
                # Normalize column names before storing
                df = normalize_column_names(df)
                
                table_name = store_dataframe_as_table(df, file.filename, engine)

                # Store file metadata in original File table
                existing_file = db.query(File).filter(File.filename == file.filename).first()
                if existing_file:
                    raise HTTPException(
                        status_code=400, detail=f"File '{file.filename}' already exists."
                    )

                db_file = File(filename=file.filename, content=file_content)
                db.add(db_file)
                db.commit()
                db.refresh(db_file)

                # Convert DataFrame to JSON for response
                data = df.to_dict(orient="records")
                
                # Find token data for this file
                file_tokens = next((item for item in file_token_data if item["filename"] == file.filename), {"tokens": 0, "breakdown": {}})
                
                simple_files_data.append({
                    "filename": file.filename, 
                    "table_name": table_name,
                    "data": data,
                    "tokens": file_tokens["tokens"],
                    "token_breakdown": file_tokens["breakdown"],
                    "is_complex": False
                })

        except ValueError as ve:
            raise HTTPException(
                status_code=500, detail=f"Value error in file '{file.filename}': {str(ve)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error with file '{file.filename}': {str(e)}"
            )

    # If there are complex files, return them for flattening UI
    if complex_files:
        return {
            "error": False,
            "complex_files": complex_files,
            "message": f"Found {len(complex_files)} complex files that need flattening configuration."
        }
    
    # If no complex files, return success with simple files
    percentage = (total_tokens / max_tokens) * 100
    success_message = f"✅ Files uploaded successfully: {total_tokens:,} tokens ({percentage:.1f}% of {max_tokens:,})"
    
    return {
        "error": False,
        "message": success_message,
        "files": simple_files_data,
        "total_tokens": total_tokens,
        "max_tokens": max_tokens,
        "file_breakdown": file_token_data
    }

def process_flatten_files(flatten_requests, db: Session):
    """Process file flattening requests"""
    flattened_data = []
    total_tokens = 0
    file_token_data = []
    
    for flatten_req in flatten_requests:
        filename = flatten_req.filename
        header_rows = flatten_req.header_rows
        
        if filename not in temp_file_storage:
            raise HTTPException(
                status_code=400, 
                detail=f"File '{filename}' not found in temporary storage. Please re-upload."
            )
        
        try:
            # Get the stored DataFrame and file content
            df = temp_file_storage[filename]['dataframe']
            file_content = temp_file_storage[filename]['content']
            
            # Apply flattening using ExcelFlattener class with DataFrame directly
            flattener = ExcelFlattener(df)
            flattened_df = flattener.flatten(h=header_rows, method='wide')
            
            # Replace invalid values and normalize column names
            flattened_df = flattened_df.replace([float('inf'), float('-inf')], None).fillna(value='')
            flattened_df = normalize_column_names(flattened_df)

            # Store DataFrame as a separate table for SQL queries
            table_name = store_dataframe_as_table(flattened_df, filename, engine)

            # Store file metadata in original File table
            existing_file = db.query(File).filter(File.filename == filename).first()
            if existing_file:
                # Update existing file
                existing_file.content = file_content
            else:
                db_file = File(filename=filename, content=file_content)
                db.add(db_file)
            
            db.commit()

            # Convert DataFrame to JSON for response
            data = flattened_df.to_dict(orient="records")
            
            # Calculate tokens for flattened data (simplified - you may want to recalculate)
            estimated_tokens = len(str(data)) // 4  # Rough estimation
            total_tokens += estimated_tokens
            
            file_token_data.append({
                "filename": filename,
                "tokens": estimated_tokens
            })
            
            flattened_data.append({
                "filename": filename, 
                "table_name": table_name,
                "data": data,
                "tokens": estimated_tokens,
                "is_complex": True  # Was complex, now flattened
            })
            
            # Clean up temporary storage
            del temp_file_storage[filename]

        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error flattening file '{filename}': {str(e)}"
            )

    max_tokens = 100000
    percentage = (total_tokens / max_tokens) * 100
    success_message = f"✅ Complex files flattened and uploaded successfully: {total_tokens:,} tokens ({percentage:.1f}% of {max_tokens:,})"
    
    return {
        "error": False,
        "message": success_message,
        "files": flattened_data,
        "total_tokens": total_tokens,
        "max_tokens": max_tokens,
        "file_breakdown": file_token_data
    }

def process_unpivot_files(filenames: List[str], db: Session):
    """Process file unpivoting requests"""
    # Get initialized LLM instances
    try:
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
    
    unpivoted_results = []
    
    for filename in filenames:
        try:
            # Get the original table name for this file
            table_name = create_table_name_from_filename(filename)
            
            # Read the data from the database table
            df = pd.read_sql_table(table_name, engine)
            
            if df.empty:
                logger.warning(f"No data found for table {table_name}")
                continue
            
            # Get AI unpivot strategy using GPT-4
            strategy = get_ai_unpivot_strategy(df, table_name, gpt4_llm)
            
            if not strategy['should_unpivot']:
                unpivoted_results.append({
                    "filename": filename,
                    "status": "skipped",
                    "reason": strategy.get('reason', 'AI determined unpivoting not needed'),
                    "table_name": table_name
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
                logger.warning(f"AI strategy incomplete for {filename}, using fallback logic")
                
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
                        "filename": filename,
                        "status": "skipped",
                        "reason": "No numeric columns found for unpivoting",
                        "table_name": table_name
                    })
                    continue
            
            # Perform unpivot using pd.melt
            logger.info(f"Unpivoting {filename}: ID cols: {strategy['id_columns']}, Value cols: {strategy['value_columns']}")
            
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
                
                logger.info(f"Unpivoting {filename}: {initial_rows} -> {final_rows} rows after removing nulls")
                
                if unpivoted_df.empty:
                    unpivoted_results.append({
                        "filename": filename,
                        "status": "error",
                        "reason": "Unpivoted data is empty after removing null values",
                        "table_name": table_name
                    })
                    continue
                    
            except Exception as melt_error:
                logger.error(f"Error during melt operation for {filename}: {str(melt_error)}")
                unpivoted_results.append({
                    "filename": filename,
                    "status": "error",
                    "reason": f"Melt operation failed: {str(melt_error)}",
                    "table_name": table_name
                })
                continue
            
            # Replace the original table with unpivoted data
            try:
                # Normalize column names before storing
                unpivoted_df = normalize_column_names(unpivoted_df)
                unpivoted_df.to_sql(table_name, engine, if_exists='replace', index=False)
                logger.info(f"Successfully replaced original table {table_name} with unpivoted data")
            except Exception as sql_error:
                logger.error(f"Error replacing original table {filename}: {str(sql_error)}")
                unpivoted_results.append({
                    "filename": filename,
                    "status": "error",
                    "reason": f"Failed to replace original table: {str(sql_error)}",
                    "table_name": table_name
                })
                continue
            
            unpivoted_results.append({
                "filename": filename,
                "status": "success",
                "original_table": table_name,
                "table_name": table_name,
                "original_shape": df.shape,
                "new_shape": unpivoted_df.shape,
                "strategy": {
                    "id_columns": strategy['id_columns'],
                    "value_columns": strategy['value_columns'],
                    "var_name": strategy['var_name'],
                    "value_name": strategy['value_name']
                },
                "description": f"Original table replaced with unpivoted data. {strategy.get('description', 'Unpivoted successfully')}"
            })
            
            logger.info(f"Successfully unpivoted {filename}: {df.shape} -> {unpivoted_df.shape}")
            
        except Exception as e:
            logger.error(f"Error unpivoting {filename}: {str(e)}")
            unpivoted_results.append({
                "filename": filename,
                "status": "error",
                "reason": str(e),
                "table_name": create_table_name_from_filename(filename)
            })
    
    # Prepare response
    successful_unpivots = [r for r in unpivoted_results if r['status'] == 'success']
    failed_unpivots = [r for r in unpivoted_results if r['status'] == 'error']
    skipped_unpivots = [r for r in unpivoted_results if r['status'] == 'skipped']
    
    logger.info(f"Unpivoting summary - Success: {len(successful_unpivots)}, Failed: {len(failed_unpivots)}, Skipped: {len(skipped_unpivots)}")
    
    if failed_unpivots:
        error_details = "; ".join([f"{r['filename']}: {r['reason']}" for r in failed_unpivots])
        message = f"⚠️ Unpivoting completed with errors. Successful: {len(successful_unpivots)}, Failed: {len(failed_unpivots)}, Skipped: {len(skipped_unpivots)}. Errors: {error_details}"
    else:
        message = f"✅ Unpivoting completed successfully! Processed: {len(successful_unpivots)}, Skipped: {len(skipped_unpivots)}"
    
    return {
        "error": len(failed_unpivots) > 0,
        "message": message,
        "results": unpivoted_results,
        "summary": {
            "total_files": len(filenames),
            "successful": len(successful_unpivots),
            "failed": len(failed_unpivots),
            "skipped": len(skipped_unpivots)
        }
    }