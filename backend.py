# import os
from langchain_openai import AzureChatOpenAI
# from langchain_openai import AzureOpenAIEmbeddings
# from azure.identity import ClientSecretCredential
# from azure.keyvault.secrets import SecretClient
import streamlit as st
import pandas as pd
import sqlite3
import logging
import json
import numpy as np  
import time
import re
import yaml
# from langchain_google_vertexai import VertexAIEmbeddings
from Helpers.query_extractor import NLQueryExtractor
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from Helpers.text_sanitizer import safe_insight_display
from Helpers.confidence_calculator import EnhancedConfidenceCalculator

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def setup_azure_openai():
    """
    Setup Azure OpenAI client with credentials from Streamlit secrets.
    Returns LLM instance for use in generating insights.
    """
    try:
        logger.info("Setting up Azure OpenAI connection")
        
        load_dotenv()
    
        # os.environ["AZURE_OPENAI_API_KEY"] = "a20ac46cd86c4c49bb795b713e814afb"
        # os.environ["AZURE_OPENAI_ENDPOINT"] = "https://gpt4openaidev.openai.azure.com/"
        # os.environ["AZURE_OPENAI_API_VERSION"] = "2023-03-15-preview"
        # os.environ["GOOGLE_API_KEY"] = "AIzaSyDlcgvFTK8AaZrS2e416z6YTJVlCKHh0ck"
        
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4",
            api_version="2023-03-15-preview",
            temperature=0.0
        )
        # Get credentials from Streamlit secrets
        # TENANT = st.secrets.TENANT
        # CLIENT_ID = st.secrets.CLIENT_ID
        # CLIENT_SECRET = st.secrets.CLIENT_SECRET
        
        # # Set up authentication
        # credential = ClientSecretCredential(TENANT, CLIENT_ID, CLIENT_SECRET)
        # VAULT_URL = st.secrets.VAULT_URL
        # client = SecretClient(vault_url=VAULT_URL, credential=credential)
        
        # # Get API key from vault
        # openai_key = client.get_secret("GenAIBIMInternalCapabilityOpenAIKey")
        
        # # Set environment variables
        # os.environ["OPENAI_API_KEY"] = openai_key.value
        
        
        # # Initialize LLM
        # llm = AzureChatOpenAI(
        #     azure_deployment="gpt-4",  # exact deployment name in Azure
        #     azure_endpoint=st.secrets.azure_endpoint,
        #     api_version="2023-12-01-preview",
        #     temperature=0.0  
        # )
        
        # # Initialize embeddings (keeping for future use)
        # embeddings = AzureOpenAIEmbeddings(
        #     azure_deployment="embeddings",  # must match your Azure deployment
        #     azure_endpoint=st.secrets.azure_endpoint,
        #     api_version="2023-12-01-preview"
        # )
        llm2 = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

        # embeddings = VertexAIEmbeddings(model_name = 'text-embedding-005')
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
       
        logger.info("Azure OpenAI setup successful")
        return llm,embeddings,llm2
     
    except Exception as e:
        logger.error(f"Error setting up Azure OpenAI: {str(e)}")
        raise Exception(f"Failed to initialize Azure OpenAI: {str(e)}")

SQL_CAPABILITIES = config['sql_capabilities']

def get_schema_description(df):
    """
    Generate a detailed schema description from the dataframe, 
    including column names, data types, and example values.
    """
    schema = []
    
    for col in df.columns:
        # Determine the data type
        dtype = str(df[col].dtype)
        
        # Get example values, handling potential errors
        try:
            sample_vals = df[col].dropna().unique()[:3]
            if len(sample_vals) == 0:
                examples = "No non-null examples available"
            else:
                # Format example values based on data type
                if dtype == 'object':
                    # For string values, truncate if too long
                    formatted_vals = [f'"{str(val)[:20]}..."' if len(str(val)) > 20 else f'"{val}"' for val in sample_vals]
                    examples = ", ".join(formatted_vals)
                elif 'datetime' in dtype:
                    # For datetime values, format as ISO
                    examples = ", ".join([f'"{val}"' for val in sample_vals])
                else:
                    # For numeric values
                    examples = ", ".join([str(val) for val in sample_vals])
            
            # Count null values
            null_count = df[col].isna().sum()
            null_percentage = (null_count / len(df)) * 100 #calculate Null percentage
            
            # Add detailed column information
            schema.append(f"- `{col}` ({dtype}) | Nulls: {null_count} ({null_percentage:.1f}%) | Examples: {examples}")
        
        except Exception as e:
            # Handle any errors in processing column info
            schema.append(f"- `{col}` ({dtype}) | Error getting examples: {str(e)}")
    
    return "\n".join(schema)

def preprocess_column_name(col):
    """
    Standardize column names for SQL compatibility while preserving semantics.
    Handles multi-level column names better and makes them SQL-friendly.
    """
    import re
    
    if col is None:
        return "unknown_column"
        
    # If it's a tuple (from MultiIndex columns), join the parts with underscores
    if isinstance(col, tuple):
        # Filter out None and empty strings, join valid parts with underscores
        parts = [str(part).strip() for part in col if part is not None and str(part).strip()]
      

    else:
        col_str = str(col)
    
    # Replace any characters that aren't alphanumeric or underscores with an underscore
    cleaned = re.sub(r'[^\w]', '_', col_str)
    
    # Remove leading numbers (SQLite doesn't like column names starting with numbers)
    # cleaned = re.sub(r'^[0-9]+', '_', cleaned)
    cleaned = re.sub(r'^([0-9]+)', r'_\1', cleaned)

    
    # Ensure the column name isn't empty after cleaning
    if not cleaned or cleaned.isdigit():
        cleaned = f"col_{cleaned if cleaned else 'unknown'}"
        
    # Strip leading/trailing underscores and lowercase
    cleaned = cleaned.strip('_').lower()
    
    # Replace multiple consecutive underscores with a single one
    cleaned = re.sub(r'_+', '_', cleaned)
    
    return cleaned

def normalize_indian_phone_number(val):
    """
    Normalize Indian phone numbers to a standard 10-digit format.

    - Handles null/NaN values by returning them unchanged
    - Removes "+91" or "+91-" prefix if present
    - Removes all non-digit characters
    - Returns last 10 digits if number is too long
    - Returns NaN if cleaned number has less than 10 digits

    Args:
        val: The phone number value to normalize
        
    Returns:
        Normalized phone number as string (10 digits) or np.nan
    """
    if pd.isna(val):
        return val
    
    # Convert to string and strip whitespaces
    str_val = str(val).strip()
    
    # Remove "+91-" or "+91" prefix
    if str_val.startswith('+91-'):
        str_val = str_val[4:]
    elif str_val.startswith('+91'):
        str_val = str_val[3:]
    
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', str_val)
    
    # If cleaned number is empty or less than 10 digits, return NaN
    if not digits_only or len(digits_only) < 10:
        return np.nan
    
    # If more than 10 digits, take the last 10 digits
    if len(digits_only) > 10:
        return np.nan
    
    # Exactly 10 digits
    return digits_only


def execute_and_store_results(query, user_question):
    """Execute the SQL query and return results for storage"""
    try:
        result_df = fetch_sql_result("data_store.db", query)
        
        # Check for errors in result
        if 'Error' in result_df.columns and len(result_df) == 1:
            return {
                'type': 'error',
                'error_message': result_df.iloc[0]['Error'],
                'query': query
            }
        else:
            # Generate insight
            answer = generate_insight_from_result(result_df, user_question, st.session_state.llm)
            
            # Add to history
            st.session_state.history.append((user_question, answer))
            
            return {
                'type': 'success',
                'dataframe': result_df,
                'insight': answer,
                'query': query,
                'row_count': len(result_df)
            }
    
    except Exception as e:
        return {
            'type': 'error',
            'error_message': str(e),
            'query': query
        }

def display_stored_results(results):
    """Display the stored query results"""
    if results['type'] == 'error':
        st.error("❌ SQL Error:")
        st.error(results['error_message'])
    else:
        # Show success message
        st.success(f"✅ Query executed successfully! Found {results['row_count']} rows.")
        
        # Show results dataframe
        st.subheader("📊 Query Results")
        st.dataframe(results['dataframe'], use_container_width=True)
        
        # Show insight (LaTeX issues already fixed, formatting preserved)
        st.subheader("💡 Insight")
        # Safe to use st.write - formatting preserved, LaTeX issues fixed
        st.write(results['insight'])
        st.markdown("</div>", unsafe_allow_html=True)

# def is_complex_sheet(df, ws=None):
#     """
#     Balanced detection of complex sheets that need flattening.
#     Uses multiple indicators with reasonable thresholds.
    
#     Args:
#         df: DataFrame representation of the worksheet/CSV
#         ws: openpyxl worksheet object (None for CSV files)
        
#     Returns:
#         bool: True if the sheet is complex and needs flattening
#     """
#     import numpy as np
#     import re
    
#     # Check for empty dataframe
#     if df.empty or df.dropna(how='all').empty:
#         return False
    
#     # If very small dataset, probably not complex
#     if len(df) < 3 or len(df.columns) < 2:
#         return False
    
#     complexity_score = 0
#     max_possible_score = 10  # Increased for more granular scoring
    
#     # 1. Check for merged cells (Excel only) - VERY STRONG indicator
#     if ws is not None:
#         has_merged_cells = len(ws.merged_cells.ranges) > 0
#         if has_merged_cells:
#             complexity_score += 4  # Very strong indicator of complex structure
#             print(f"  + Merged cells detected: +4 points")
    
#     # 2. Check for blank rows at top - LOWERED threshold
#     blank_top_rows = 0
#     for i in range(min(6, len(df))):
#         if df.iloc[i].isnull().all():
#             blank_top_rows += 1
#         else:
#             break
    
#     if blank_top_rows >= 2:  # Lowered from 3
#         complexity_score += 2
#         print(f"  + {blank_top_rows} blank top rows: +2 points")
#     elif blank_top_rows >= 1:
#         complexity_score += 1
#         print(f"  + {blank_top_rows} blank top row: +1 point")
    
#     # 3. Check for inconsistent row lengths - ADJUSTED thresholds
#     row_lengths = []
#     sample_rows = min(8, len(df))  # Check more rows
#     for i in range(sample_rows):
#         row_length = df.iloc[i].count()  # Count non-null values
#         row_lengths.append(row_length)
    
#     if len(set(row_lengths)) > 1:
#         # Calculate coefficient of variation for row lengths
#         mean_length = np.mean(row_lengths)
#         std_length = np.std(row_lengths)
#         if mean_length > 0:
#             cv = std_length / mean_length
#             if cv > 0.4:  # Lowered from 0.5
#                 complexity_score += 2
#                 print(f"  + High row length variation (CV={cv:.2f}): +2 points")
#             elif cv > 0.2:  # Lowered from 0.3
#                 complexity_score += 1
#                 print(f"  + Moderate row length variation (CV={cv:.2f}): +1 point")
    
#     # 4. Check for multiple text-heavy rows - LOWERED threshold
#     potential_headers = 0
#     for i in range(min(5, len(df))):
#         row = df.iloc[i].dropna()
#         if len(row) == 0:
#             continue
            
#         # Count text vs numeric in this row
#         text_count = 0
#         for val in row:
#             val_str = str(val).strip()
#             # Check if it's numeric (including formatted numbers)
#             is_numeric = False
            
#             # Remove common formatting and check if numeric
#             clean_val = val_str.replace(',', '').replace(' ', '').replace('$', '').replace('%', '')
#             try:
#                 float(clean_val)
#                 is_numeric = True
#             except ValueError:
#                 # Try regex for scientific notation
#                 if re.match(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$', clean_val):
#                     is_numeric = True
            
#             if not is_numeric:
#                 text_count += 1
        
#         text_ratio = text_count / len(row)
#         if text_ratio > 0.7:  # Lowered from 0.8
#             potential_headers += 1
    
#     if potential_headers >= 2:  # Lowered from 3
#         complexity_score += 2
#         print(f"  + {potential_headers} text-heavy rows: +2 points")
#     elif potential_headers >= 1 and potential_headers < 2:
#         complexity_score += 1
#         print(f"  + {potential_headers} text-heavy row: +1 point")
    
#     # 5. Check for hierarchical/indented structure - ADJUSTED
#     indented_cells = 0
#     sample_size = min(12, len(df))
#     for i in range(sample_size):
#         for col_idx in range(min(4, len(df.columns))):  # Check more columns
#             if i < len(df):
#                 val = df.iloc[i, col_idx]
#                 if isinstance(val, str) and (val.startswith('  ') or val.startswith('\t') or 
#                                            val.startswith('    ')):  # Multiple space patterns
#                     indented_cells += 1
    
#     if indented_cells >= 3:  # Lowered from 5
#         complexity_score += 1
#         print(f"  + {indented_cells} indented cells: +1 point")
    
#     # 6. Check for sparse/irregular column patterns - ADJUSTED
#     if len(df.columns) >= 3:  # Lowered requirement
#         col_densities = []
#         for col_idx in range(min(6, len(df.columns))):
#             col_data = df.iloc[:min(8, len(df)), col_idx]  # Check more rows
#             density = col_data.count() / len(col_data)
#             col_densities.append(density)
        
#         # Check for significant density variations
#         pattern_breaks = 0
#         for i in range(len(col_densities) - 1):
#             if abs(col_densities[i] - col_densities[i+1]) > 0.5:  # Lowered from 0.6
#                 pattern_breaks += 1
        
#         if pattern_breaks >= 2:  # Lowered from 3
#             complexity_score += 1
#             print(f"  + {pattern_breaks} column density breaks: +1 point")
    
#     # 7. Check for inconsistent column usage - ADJUSTED
#     if len(df) >= 3:
#         first_rows = df.head(4)  # Check more rows
#         col_usage = []
#         for col in first_rows.columns:
#             usage = first_rows[col].count() / len(first_rows)
#             col_usage.append(usage)
        
#         # If many columns are barely used, might be complex
#         barely_used = sum(1 for usage in col_usage if usage < 0.3)  # Lowered from 0.5
#         if barely_used > len(col_usage) * 0.3:  # Lowered from 0.4
#             complexity_score += 1
#             print(f"  + {barely_used} barely used columns: +1 point")
    
#     # 8. NEW: Check for obvious report-style patterns
#     # Look for single-column spanning titles or descriptions
#     single_value_rows = 0
#     for i in range(min(4, len(df))):
#         row = df.iloc[i].dropna()
#         if len(row) == 1 and len(df.columns) > 2:  # Single value in multi-column sheet
#             val = str(row.iloc[0]).strip()
#             if len(val) > 10:  # Substantial text suggesting a title/description
#                 single_value_rows += 1
    
#     if single_value_rows >= 1:
#         complexity_score += 1
#         print(f"  + {single_value_rows} title-like rows: +1 point")
    
#     # 9. NEW: Check for date/time patterns that suggest reports
#     date_header_pattern = False
#     for i in range(min(3, len(df))):
#         row = df.iloc[i].dropna()
#         for val in row:
#             val_str = str(val).lower()
#             if any(word in val_str for word in ['report', 'summary', 'analysis', 'quarter', 'month', 'year', 'period']):
#                 date_header_pattern = True
#                 break
#         if date_header_pattern:
#             break
    
#     if date_header_pattern:
#         complexity_score += 1
#         print(f"  + Report-style language detected: +1 point")
    
#     # Calculate final complexity ratio
#     complexity_ratio = complexity_score / max_possible_score
    
#     # BALANCED threshold - not too sensitive, not too conservative
#     is_complex = complexity_ratio >= 0.2  # 1 out of 10 points needed

#     # Debug info
#     print(f"Complexity Analysis for sheet:")
#     print(f"  - Shape: {df.shape}")
#     print(f"  - Total complexity score: {complexity_score}/{max_possible_score} = {complexity_ratio:.3f}")
#     print(f"  - Threshold: 0.25 (25%)")
#     print(f"  - Classification: {'COMPLEX' if is_complex else 'SIMPLE'}")
#     print("-" * 50)
    
#     return is_complex

def is_complex_sheet(df, ws=None):
    """
    Simplified complexity detection focused on multiple header rows.
    A sheet is considered complex if it has more than 1 header row.
    
    Args:
        df: DataFrame representation of the worksheet/CSV
        ws: openpyxl worksheet object (None for CSV files)
        
    Returns:
        bool: True if the sheet has multiple headers (complex), False otherwise
    """
    import numpy as np
    import re
    df = df.reset_index(drop=True)
    df.columns = range(df.shape[1])
    # Check for empty dataframe
    if df.empty or df.dropna(how='all').empty:
        return False
    
    # If very small dataset, probably not complex
    if len(df) < 3 or len(df.columns) < 2:
        return False
    
    print(f"Analyzing sheet for multiple headers - Shape: {df.shape}")
    
    # Use the header detection function to find number of header rows
    try:
        from header_detection_helper import detect_header_rows_automatically
        detected_headers = detect_header_rows_automatically(df)
        print(df)
        print(f"  - Detected headers: {detected_headers}")
        
        # Complex if more than 1 header row detected
        is_complex = detected_headers > 1
        
        print(f"  - Classification: {'COMPLEX' if is_complex else 'SIMPLE'} (based on header count)")
        print("-" * 50)
        
        return is_complex
        
    except Exception as e:
        print(f"  - Header detection failed: {str(e)}")
        print(f"  - Falling back to heuristic method")
        
        # Fallback: Simple heuristic check for obvious multiple headers
        return fallback_multiple_header_check(df)

def fallback_multiple_header_check(df):
    """
    Fallback method to detect multiple headers using simple heuristics.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        bool: True if likely to have multiple headers
    """
    # Check first few rows for text-heavy patterns (likely headers)
    text_heavy_rows = 0
    
    for i in range(min(4, len(df))):
        row = df.iloc[i].dropna()
        if len(row) == 0:
            continue
            
        # Count non-numeric values in this row
        text_count = 0
        for val in row:
            val_str = str(val).strip()
            # Check if it's numeric
            try:
                float(val_str.replace(',', '').replace(' ', '').replace('$', '').replace('%', ''))
            except ValueError:
                text_count += 1
        
        text_ratio = text_count / len(row) if len(row) > 0 else 0
        if text_ratio > 0.7:  # Row is mostly text (likely header)
            text_heavy_rows += 1
    
    # Complex if we found 2 or more text-heavy rows (multiple headers)
    is_complex = text_heavy_rows >= 2
    
    print(f"  - Fallback method: {text_heavy_rows} text-heavy rows found")
    print(f"  - Classification: {'COMPLEX' if is_complex else 'SIMPLE'}")
    print("-" * 50)
    
    return is_complex

def quick_complexity_check(df):
    """
    Quick check - can be simplified since we're focusing on headers.
    """
    if df.empty or len(df) < 3:
        return False
    
    # Quick check for obvious multiple text rows at the top
    first_3_rows = df.head(3)
    text_rows = 0
    
    for i in range(len(first_3_rows)):
        row = first_3_rows.iloc[i].dropna()
        if len(row) == 0:
            continue
        
        # Count text vs numeric
        text_count = 0
        for val in row:
            try:
                float(str(val).replace(',', '').replace(' ', ''))
            except (ValueError, TypeError):
                text_count += 1
        
        if text_count / len(row) > 0.6:  # Mostly text
            text_rows += 1
    
    return text_rows >= 2  # Multiple header rows detected


def preprocess_and_store_sheet(df, sheet_name, db_path="data_store.db"):
    """
    Preprocess a single dataframe (sheet) and store it in SQLite.
    Returns the table name for reference.
    """
    try:
        # Sanitize sheet name for SQL table name
        table_name = preprocess_column_name(sheet_name)
        logger.info(f"Processing sheet '{sheet_name}' with {len(df)} rows and {len(df.columns)} columns")
        
        # Create a copy to avoid modifying the original
        processed_df = df.copy()

        # Standardize column names and ensure uniqueness
        clean_columns = [preprocess_column_name(col) for col in processed_df.columns]
      

        #solving dublicate column names
        seen = set()
        final_columns = []
        for col in clean_columns:
            if col not in seen:
                final_columns.append(col)
                seen.add(col)
            else:
                suffix = 1
                new_col = f"{col}_{suffix}"
                while new_col in seen:
                    suffix += 1
                    new_col = f"{col}_{suffix}"
                final_columns.append(new_col)
                seen.add(new_col)
        processed_df.columns = final_columns

        # Normalize phone number columns
        for col in processed_df.columns:
            if 'phone' in col.lower() or 'mobile' in col.lower():
                logger.info(f"Normalizing phone numbers in column '{col}'")
                processed_df[col] = processed_df[col].astype(str)
                processed_df[col] = processed_df[col].apply(normalize_indian_phone_number)
        
        # Convert potential date strings to datetime objects
        for col in processed_df.columns:
            try:
                col_data = processed_df[col]
                if isinstance(col_data, pd.Series) and col_data.dtype == 'object':
                    date_sample = col_data.dropna().iloc[0] if not col_data.dropna().empty else None
                    if date_sample and isinstance(date_sample, str) and len(date_sample) > 6:
                        if any(x in date_sample for x in ['/', '-', ':', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
                            processed_df[col] = pd.to_datetime(col_data, errors='ignore')
            except Exception as date_err:
                logger.warning(f"Skipping column '{col}' during datetime parsing: {date_err}")
        
        # Replace inf/-inf with NA
        processed_df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        
        # Save to SQLite
        conn = sqlite3.connect(db_path)
        processed_df.to_sql(table_name, conn, if_exists='replace', index=False)

        # Log schema info
        cursor = conn.cursor()
        schema_info = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
        logger.info(f"Created SQLite table '{table_name}' with {len(schema_info)} columns")
        conn.close()
        
        return table_name

    except Exception as e:
        logger.error(f"Error in preprocessing sheet '{sheet_name}': {str(e)}")
        raise Exception(f"Failed to process and store sheet '{sheet_name}': {str(e)}")

def preprocess_and_store_in_sqlite(df_dict, db_path="data_store.db"):
    """
    Preprocess multiple dataframes from different sheets and store them in SQLite.
    Returns a dictionary mapping sheet names to table names.
    """
    try:
        logger.info(f"Processing {len(df_dict)} sheets/tables")
        
        table_dict = {}
        
        # Process each sheet in the Excel file
        for sheet_name, df in df_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                table_name = preprocess_and_store_sheet(df, sheet_name, db_path)
                table_dict[sheet_name] = table_name
        
        # Get all tables from SQLite for verification
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        logger.info(f"SQLite database now contains {len(tables)} tables: {[t[0] for t in tables]}")
        conn.close()
        
        return table_dict
        
    except Exception as e:
        logger.error(f"Error in preprocessing data: {str(e)}")
        raise Exception(f"Failed to process and store data: {str(e)}")

def get_database_schema(db_path="data_store.db"):
    """
    Get schema information for all tables in the database.
    Returns a dictionary mapping table names to their column information.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        
        schema_info = {}
        for table in tables:
            table_name = table[0]
            # Get column info for each table
            columns = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
            schema_info[table_name] = [
                {"name": col[1], "type": col[2]} for col in columns
            ]
        
        conn.close()
        return schema_info
        
    except Exception as e:
        logger.error(f"Error getting database schema: {str(e)}")
        return {}



def fetch_sql_result(db_path, query):
    """
    Execute an SQL query against the SQLite database and return the results.
    """
    start_time = time.time()
    try:
        logger.info(f"Executing SQL query: {query}")
        conn = sqlite3.connect(db_path)
        
        # Add timeout to prevent long-running queries
        conn.execute("PRAGMA timeout = 5000")  # 5 second timeout
        
        # Execute the query and get results
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        # Log query execution time
        logger.info(f"Query executed in {time.time() - start_time:.2f} seconds, returned {len(result)} rows")
        return result
        
    except Exception as e:
        logger.error(f"Error executing SQL query: {str(e)}")
        # Create an error dataframe with the error message
        error_df = pd.DataFrame({
            "Error": [f"SQL Error: {str(e)}"],
            "Query": [query]
        })
        return error_df

def generate_insight_from_result(result_df, user_question, llm):
    """
    Generate natural language insights from SQL query results.
    """
    try:
        # Prepare the data for the prompt
        if len(result_df) > 0:
            # Convert DataFrame to markdown table
            result_markdown = result_df.to_markdown(index=False)
            
            # Get result statistics - FIX FOR INT64 JSON SERIALIZATION ERROR
            result_stats = {}
            for col in result_df.select_dtypes(include=['number']).columns:
                # Convert NumPy types to Python native types
                result_stats[col] = {
                    "min": float(result_df[col].min()),
                    "max": float(result_df[col].max()),
                    "mean": float(result_df[col].mean()),
                    "median": float(result_df[col].median())
                }
            
            # Use the NumPy encoder for JSON serialization
            stats_str = json.dumps(result_stats, indent=2, cls=NumpyEncoder) if result_stats else "No numeric columns available"
            
            row_count = len(result_df)
            column_count = len(result_df.columns)
        else:
            result_markdown = "No results returned from the query."
            stats_str = "No results available"
            row_count = 0
            column_count = 0
        
        # Build the prompt
        prompt = f"""
You are a data analyst explaining query results to a business user.

USER QUESTION:
"{user_question}"

QUERY RESULT STATISTICS:
Total rows: {row_count}
Total columns: {column_count}

NUMERIC COLUMN STATISTICS (if available):
{stats_str}

RESULT DATA:
{result_markdown}

INSTRUCTIONS:
Analyze the given dataset and provide concise business insights that can support strategic or operational decision-making and give some recommendations for decision making according to user question and intent. Follow these guidelines strictly:
1. Do NOT describe or summarize the data structure. Focus only on business-relevant insights.
2. Highlight key patterns, trends, or anomalies that could influence business decisions.
3. Use clear, business-friendly language — avoid technical or statistical terms.
4. If the analysis returns no data or an error, explain clearly why the question cannot be answered with the current dataset.
5. Back up each insight with specific numbers or comparisons where applicable.
6. Keep the response under 200 words.
7.Maintain a consistent and executive-friendly format for all responses.

INSIGHT:
""".strip()
        
        # Generate insight using LLM
        insight = llm.invoke(prompt).content.strip()
       
        logger.info(f"Generated insight from results with {row_count} rows")
        safe_insight = safe_insight_display(insight)
        return safe_insight
        
    except Exception as e:
        logger.error(f"Error generating insight: {str(e)}")
        return f"Sorry, I couldn't generate an insight from the results. Error: {str(e)}"

def generate_general_data_insight(user_question, table_samples, llm):
    """
    Generate insights from dataframe samples when SQL query isn't appropriate.
    Uses information from multiple tables if available.
    """
    try:
        # Generate combined information from all table samples
        table_info = []
        
        for table_name, df_head in table_samples.items():
            # Generate schema information
            schema_description = get_schema_description(df_head)
            
            # Add table information
            table_info.append(f"TABLE: {table_name}")
            table_info.append(f"ROWS: {len(df_head)}")
            table_info.append(f"COLUMNS: {len(df_head.columns)}")
            table_info.append(f"SCHEMA:\n{schema_description}")
            
            # Generate sample data
            table_info.append(f"SAMPLE DATA:\n{df_head.head(5).to_markdown(index=False)}")
            table_info.append("-" * 40)
        
        # Join all table information
        all_tables_str = "\n".join(table_info)
        
        # Build the prompt
        prompt = f"""
You are a data analyst providing insights based on samples from multiple tables.

USER QUESTION:
"{user_question}"

DATASET INFORMATION:
{all_tables_str}

INSTRUCTIONS:
1. Analyze the data samples carefully across all tables
2. Consider how the tables might be related based on column names
3. Answer the user's question as accurately as possible based on the available information
4. If the question applies to specific tables, focus on those tables
5. If the question cannot be answered with the provided samples, explain why
6. Be honest about limitations due to the sample size
7. Keep your response under 200 words and focus on the most important insights

INSIGHT:
""".strip()
        
        # Generate insight
        insight = llm.invoke(prompt).content.strip()

        logger.info("Generated general data insight from samples of multiple tables")
        safe_insight = safe_insight_display(insight)
        return safe_insight
        
    except Exception as e:
        logger.error(f"Error generating general insight: {str(e)}")
        return f"I couldn't generate an insight from the data samples. Error: {str(e)}"


def clean_sql_response(response):
    """Clean up SQL response by removing markdown formatting."""
    if response.startswith("```sql"):
        response = response.split("```sql")[1]
    if response.startswith("```"):
        response = response.split("```")[1]
    if "```" in response:
        response = response.split("```")[0]
    return response.strip()


def is_query_relevant(sql_query: str, user_query: str, llm, embeddings,llm2):
    """
    Enhanced confidence calculation using multi-dimensional approach
    Returns overall confidence score for backward compatibility
    """
    try:
        # Initialize the enhanced calculator
        calculator = EnhancedConfidenceCalculator(embeddings, llm2, "data_store.db")
        
        # Get schema info
        schema_info = get_database_schema()
        
        # Get table samples from session state if available
        table_samples = {}
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'table_samples'):
            table_samples = st.session_state.table_samples
        
        # Calculate confidence scores with proper table samples
        confidence_scores = calculator.calculate_confidence(
            user_query, sql_query, schema_info, table_samples
        )

               
        # Return overall score for backward compatibility
        return confidence_scores['overall']
        
    except Exception as e:
        logger.warning(f"Enhanced confidence calculation failed, using fallback: {e}")
        
        # Fallback to original method if enhanced calculation fails
        prompt = f"""Convert the following SQL query into a concise natural language question. 
        Only return the question without any explanation, SQL, or formatting.
        If given sql query is not correct, then generate i dont know.

        SQL: {sql_query}         
        """
        
        try:
            sql_nl = llm.invoke(prompt).content.strip()
            user_embed = np.array(embeddings.embed_query(user_query))
            sql_embed = np.array(embeddings.embed_query(sql_nl))
            
            similarity = np.dot(user_embed, sql_embed) / (
                np.linalg.norm(user_embed) * np.linalg.norm(sql_embed)
            )
            confidence = round(((similarity + 1) / 2) * 100, 2)
            return confidence
        except Exception:
            return 50.0  # Default fallback score

def get_detailed_confidence(sql_query: str, user_query: str, llm, embeddings, table_samples: dict = None):
    """
    Get detailed confidence breakdown for UI display
    """
    try:
        calculator = EnhancedConfidenceCalculator(embeddings, llm, "data_store.db")
        schema_info = get_database_schema()
        
        # Use provided table_samples or get from session state
        if table_samples is None and hasattr(st, 'session_state') and hasattr(st.session_state, 'table_samples'):
            table_samples = st.session_state.table_samples
        elif table_samples is None:
            table_samples = {}
        
        confidence_scores = calculator.calculate_confidence(
            user_query, sql_query, schema_info, table_samples
        )
        
        return confidence_scores
        
    except Exception as e:
        logger.warning(f"Detailed confidence calculation failed: {e}")
        
        # Return basic confidence if detailed calculation fails
        basic_confidence = is_query_relevant(sql_query, user_query, llm, embeddings)
        return {
            'semantic': basic_confidence,
            'execution': 70.0,
            'syntax': 70.0,
            'coverage': 70.0,
            'overall': basic_confidence
        }

# Also update your generate_sql_prompt function to pass table_samples
def generate_sql_prompt(user_question, table_dict, table_samples, llm, query_memory=None):
    """
    Generate an SQL query from a natural language question considering multiple tables.
    Enhanced with Natural Language Query Element Extraction and ConversationBufferMemory.
    
    Args:
        user_question: The natural language question from the user
        table_dict: Dictionary mapping sheet names to table names
        table_samples: Dictionary mapping table names to sample dataframes
        llm: The language model instance to use
        query_memory: ConversationBufferMemory instance containing previous attempts
    """
    try:
        # Initialize the NL Query Extractor
        extractor = NLQueryExtractor()
        
        # Prepare available tables and columns for extraction
        available_tables = list(table_dict.values())
        available_columns = []
        for df in table_samples.values():
            available_columns.extend(df.columns.tolist())
        
        # Extract elements from user query
        nl_elements = extractor.extract_nl_elements(
            user_question, 
            available_tables=available_tables,
            available_columns=available_columns
        )
        
        # Build schema information for all tables
        schema_descriptions = {}
        for sheet_name, table_name in table_dict.items():
            if table_name in table_samples:
                df_head = table_samples[table_name]
                # Ensure column names are preprocessed
                df_head.columns = [preprocess_column_name(col) for col in df_head.columns]
                schema_descriptions[table_name] = get_schema_description(df_head)
        
        # Build combined schema information
        all_tables_info = []
        for table_name, sample_df in table_samples.items():
            column_names = ", ".join([f"`{col}`" for col in sample_df.columns])
            all_tables_info.append(f"TABLE: `{table_name}`\nCOLUMNS: {column_names}\n")
            all_tables_info.append(f"SCHEMA:\n{schema_descriptions.get(table_name, 'No schema information')}\n")
            all_tables_info.append(f"SAMPLE DATA:\n{sample_df.to_markdown(index=False)}\n")
            all_tables_info.append("-" * 40 + "\n")
        
        all_tables_str = "\n".join(all_tables_info)
        tables_list = ", ".join([f"`{t}`" for t in table_dict.values()])
        
        # Build extracted elements information
        extracted_info = ""
        if any(nl_elements.values()):  # Only add if we extracted something
            extracted_info = f"""
        EXTRACTED QUERY ELEMENTS FROM USER QUESTION:
        - Query Intent: {', '.join(nl_elements['select_type']) if nl_elements['select_type'] else 'General selection'}
        - Likely Tables: {', '.join(nl_elements['tables']) if nl_elements['tables'] else 'Not specified'}
        - Likely Columns: {', '.join(nl_elements['columns']) if nl_elements['columns'] else 'Not specified'}
        - Conditions/Filters: {', '.join(nl_elements['conditions']) if nl_elements['conditions'] else 'None specified'}
        - Aggregations Needed: {', '.join(nl_elements['aggregations']) if nl_elements['aggregations'] else 'None'}
        - Sorting Requirements: {', '.join(nl_elements['sorting']) if nl_elements['sorting'] else 'None'}
        - Limit Requirements: {', '.join(nl_elements['limits']) if nl_elements['limits'] else 'None'}
        - Join Indicators: {', '.join(nl_elements['joins']) if nl_elements['joins'] else 'None'}
        - Comparison Operators: {', '.join(nl_elements['operators']) if nl_elements['operators'] else 'None'}
        
        USE THESE EXTRACTED ELEMENTS TO GUIDE YOUR SQL GENERATION.
        """
        
        # Get conversation history from memory
        previous_context = ""
        if query_memory and hasattr(query_memory, 'buffer'):
            # Get the conversation history
            history = query_memory.buffer
            if history:
                previous_context = f"""
        PREVIOUS QUERY ATTEMPTS CONTEXT:
        {history}
        
        IMPORTANT: 
        - Analyze the previous attempts and their patterns
        - Generate a DIFFERENT approach than what was tried before
        - Consider alternative table joins, different aggregations, or different filtering methods
        - Learn from previous attempts to create a more accurate query
        - If previous queries failed, try a simpler or more specific approach
        """
        
        prompt = f"""
        You are a SQL expert tasked with translating natural language questions into accurate SQLite queries.

        USER QUESTION:
        "{user_question}"
        {extracted_info}
        AVAILABLE TABLES:
        {tables_list}

        TABLES INFORMATION:
        {all_tables_str}
        
        {previous_context}

         INSTRUCTIONS:
            1. Generate ONLY a valid SQLite SQL query that answers the user's question, otherwise generate "SQL Query can't be generated".
            2. Pay special attention to the EXTRACTED QUERY ELEMENTS above to understand user intent.
            3. If specific tables/columns were identified, prioritize using those in your query.
            4. If aggregations were detected, make sure to include appropriate GROUP BY clauses.
            5. If conditions were found, implement appropriate WHERE clauses.
            6. **If sorting is mentioned OR implied (e.g., top results, latest records, alphabetical display), include an ORDER BY clause using the most relevant column.**
            7. **If the user does not explicitly mention sorting, still add ORDER BY if it improves clarity — such as ordering by name, date, quantity, or price. Use ASC by default unless context implies DESC.**
            8. If limits were specified, include LIMIT clauses with appropriate numbers.
            9. Use appropriate JOIN clauses if multiple tables are needed based on extracted elements.
            10. Format dates appropriately if date-related operations are needed. **Always cast date strings to proper date format using DATE(column_name) when comparing or filtering dates.**
            11. Do not include any explanations, only output the SQL query.
            12. Use TOP only when specifically requested.
            13. If encountered with CREATEDDATE column convert it to date (mm/dd/yyyy) format.
            14. You MUST NOT hallucinate about tables, data, columns and query. If user query is not self-explanatory, ask user to rephrase the query. Do not make anything by yourself.
            15. Only provide a single query in the response.
            16. If previous attempts are shown above, ensure this query uses a DIFFERENT approach or perspective.
            17. Try alternative column combinations, different aggregations, or different filtering approaches.
            18. Analyze each column in available tables carefully.
            19. Learn from the conversation history to improve query accuracy.
            20. **MANDATORY: Always use SELECT DISTINCT for all queries to eliminate duplicate records unless performing aggregation functions (COUNT, SUM, AVG, etc.) where DISTINCT is not applicable.**
            21. **When using DISTINCT, apply it to the entire SELECT clause: SELECT DISTINCT column1, column2, ... FROM table.**
            22. **For aggregation queries, use DISTINCT within the aggregation function when appropriate: COUNT(DISTINCT column), SUM(DISTINCT column).**
            23. **MANDATORY: When filtering or comparing date values, cast them using DATE(column_name) to ensure proper date handling. Example: WHERE DATE(end_date) <= '2023-06-30'.**
            24. **MANDATORY: When appropriate, include ORDER BY clauses to improve result readability or relevance, especially for name, date, quantity, price, or rank-related fields.**
            25. **MANDATORY: If ORDER BY is added, sort the results by the first column in the SELECT clause unless the user clearly requests sorting by a different column.**        
       
         SQL QUERY:
        """.strip()
        
        # Generate SQL query using LLM
        response = llm.invoke(prompt).content.strip()
        
        # Log the extracted elements for debugging
        logger.info(f"Extracted NL elements: {nl_elements}")
        logger.info(f"Generated SQL query: {response}")
        
        # Clean up the SQL query if needed (remove any markdown formatting artifacts)
        if response.startswith("```sql"):
            response = response.split("```sql")[1]
        if "```" in response:
            response = response.split("```")[0]
        
        return response.strip()
        
    except Exception as e:
        logger.error(f"Error generating SQL query: {str(e)}")
        # Choose the first table as a fallback
        fallback_table = next(iter(table_dict.values())) if table_dict else "unknown_table"
        return f"SELECT * FROM {fallback_table} LIMIT 5 -- Error generating query: {str(e)}"