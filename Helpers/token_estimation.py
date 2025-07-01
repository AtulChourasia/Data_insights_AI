import pandas as pd
import numpy as np
import tiktoken
from typing import Dict, Tuple
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)

def get_tiktoken_encoding(model_name: str = "gpt-4"):
    """
    Get the appropriate tiktoken encoding for a given model
    
    Args:
        model_name: OpenAI model name
        
    Returns:
        tiktoken encoding object
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(f"Model {model_name} not found, using cl100k_base encoding")
        return tiktoken.get_encoding("cl100k_base")

def count_tokens_accurate(text: str, model_name: str = "gpt-4") -> int:
    """
    Count tokens using tiktoken - the actual tokenizer used by OpenAI
    
    Args:
        text: Text to tokenize
        model_name: OpenAI model name
        
    Returns:
        int: Accurate token count
    """
    if not text or pd.isna(text):
        return 0
    
    try:
        encoding = get_tiktoken_encoding(model_name)
        return len(encoding.encode(str(text)))
    except Exception as e:
        logger.warning(f"Token counting failed, using fallback: {e}")
        # Fallback to character-based estimation
        return len(str(text)) // 4

def calculate_tokens_from_dataframe(df: pd.DataFrame, model_name: str = "gpt-4", use_tiktoken: bool = True) -> int:
    """
    Calculate accurate token count from a DataFrame
    
    Args:
        df: pandas DataFrame to calculate tokens for
        model_name: OpenAI model name for tiktoken
        use_tiktoken: Whether to use tiktoken (True) or legacy method (False)
        
    Returns:
        int: Number of tokens
    """
    if df.empty:
        return 0
    
    if use_tiktoken:
        return _calculate_tokens_tiktoken(df, model_name)
    else:
        return _calculate_tokens_legacy(df)

def _calculate_tokens_tiktoken(df: pd.DataFrame, model_name: str) -> int:
    """Calculate tokens using tiktoken encoding"""
    try:
        encoding = get_tiktoken_encoding(model_name)
        total_tokens = 0
        
        # Count tokens in column headers
        header_text = ",".join(str(col) for col in df.columns)
        total_tokens += len(encoding.encode(header_text))
        
        # Count tokens in data - process in batches for large DataFrames
        batch_size = 1000
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            # Convert batch to CSV-like format
            batch_text = ""
            for _, row in batch_df.iterrows():
                row_values = []
                for value in row:
                    if pd.isna(value):
                        row_values.append("")
                    else:
                        str_value = str(value)
                        # Escape commas and quotes for CSV format
                        if ',' in str_value or '"' in str_value:
                            str_value = '"' + str_value.replace('"', '""') + '"'
                        row_values.append(str_value)
                
                batch_text += ",".join(row_values) + "\n"
            
            # Count tokens for this batch
            total_tokens += len(encoding.encode(batch_text))
        
        # Add overhead for processing (schema description, prompts, etc.)
        processing_overhead = _calculate_processing_overhead(df, encoding)
        total_tokens += processing_overhead
        
        return total_tokens
        
    except Exception as e:
        logger.error(f"tiktoken calculation failed: {e}")
        return _calculate_tokens_legacy(df)

def _calculate_tokens_legacy(df: pd.DataFrame) -> int:
    """Legacy token calculation method (character-based)"""
    total_chars = 0
    
    # Count characters in column headers
    for col in df.columns:
        total_chars += len(str(col))
    
    # Count characters in all cell values
    for col in df.columns:
        for value in df[col]:
            if pd.notna(value):
                total_chars += len(str(value))
    
    # Add separators (commas, newlines, spaces)
    separators = (len(df.columns) * len(df)) + len(df)
    total_chars += separators
    
    # Convert to tokens (1 token ≈ 4 characters)
    tokens = int(total_chars / 4)
    
    return tokens

def _calculate_processing_overhead(df: pd.DataFrame, encoding) -> int:
    """Calculate processing overhead tokens for LLM context"""
    # Schema description overhead
    schema_tokens = len(df.columns) * 25  # Estimated tokens per column description
    
    # Sample data overhead (first 5-10 rows typically shown)
    sample_rows = min(10, len(df))
    sample_overhead = sample_rows * len(df.columns) * 3  # Average tokens per cell in samples
    
    # Base prompt overhead
    prompt_overhead = 500  # System prompts, instructions, etc.
    
    return schema_tokens + sample_overhead + prompt_overhead

def check_file_token_limit(uploaded_file, max_tokens: int = 100000, model_name: str = "gpt-4") -> Tuple[bool, str, int, dict]:
    """
    Check if uploaded file fits within token limit using accurate tiktoken estimation
    
    Args:
        uploaded_file: Streamlit uploaded file object
        max_tokens: Maximum allowed tokens
        model_name: OpenAI model name for tokenizer selection
        
    Returns:
        Tuple[bool, str, int, dict]: (fits_limit, message, total_tokens, file_breakdown)
    """
    file_type = uploaded_file.name.split('.')[-1].lower()
    total_tokens = 0
    file_breakdown = {}
    
    try:
        if file_type == "csv":
            # Read the entire CSV file
            df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)  # Reset file pointer
            
            tokens = calculate_tokens_from_dataframe(df, model_name=model_name, use_tiktoken=True)
            total_tokens = tokens
            
            # Get detailed breakdown
            detailed_breakdown = get_detailed_token_breakdown(df, model_name)
            
            file_breakdown["main_data"] = {
                "rows": len(df),
                "columns": len(df.columns),
                "tokens": tokens,
                "detailed": detailed_breakdown
            }
            
        elif file_type in ["xlsx", "xls"]:
            # Read all sheets from Excel file
            excel_file = pd.ExcelFile(uploaded_file)
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                tokens = calculate_tokens_from_dataframe(df, model_name=model_name, use_tiktoken=True)
                total_tokens += tokens
                
                # Get detailed breakdown
                detailed_breakdown = get_detailed_token_breakdown(df, model_name)
                
                file_breakdown[sheet_name] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "tokens": tokens,
                    "detailed": detailed_breakdown
                }
        else:
            return False, f"Unsupported file type: {file_type}", 0, {}
        
        # Check if within limit
        if total_tokens <= max_tokens:
            percentage = (total_tokens / max_tokens) * 100
            message = f"✅ File fits within limit: {total_tokens:,} tokens ({percentage:.1f}% of {max_tokens:,}) [tiktoken accurate]"
            return True, message, total_tokens, file_breakdown
        else:
            percentage = (total_tokens / max_tokens) * 100
            message = f"❌ File exceeds limit: {total_tokens:,} tokens ({percentage:.1f}% of {max_tokens:,}) [tiktoken accurate]"
            return False, message, total_tokens, file_breakdown
            
    except Exception as e:
        return False, f"Error reading file: {str(e)}", 0, {}

def get_detailed_token_breakdown(df: pd.DataFrame, model_name: str = "gpt-4") -> Dict:
    """
    Get detailed breakdown of token usage for a DataFrame
    
    Args:
        df: pandas DataFrame
        model_name: OpenAI model name
        
    Returns:
        Dict with detailed token breakdown
    """
    if df.empty:
        return {
            'headers': 0,
            'data': 0,
            'structure': 0,
            'processing_overhead': 0,
            'total': 0
        }
    
    try:
        encoding = get_tiktoken_encoding(model_name)
        
        # Header tokens
        header_text = ",".join(str(col) for col in df.columns)
        header_tokens = len(encoding.encode(header_text))
        
        # Data tokens (sample to estimate)
        sample_size = min(100, len(df))  # Sample for estimation
        sample_df = df.head(sample_size)
        
        data_tokens = 0
        structure_tokens = 0
        
        for _, row in sample_df.iterrows():
            row_text = ""
            for value in row:
                if pd.isna(value):
                    row_text += ","
                else:
                    str_value = str(value)
                    if ',' in str_value:
                        str_value = f'"{str_value}"'
                    row_text += str_value + ","
            
            row_text = row_text.rstrip(',') + "\n"
            row_tokens = len(encoding.encode(row_text))
            
            # Separate content from structure
            content_only = "".join(str(val) for val in row if pd.notna(val))
            content_tokens = len(encoding.encode(content_only))
            
            data_tokens += content_tokens
            structure_tokens += (row_tokens - content_tokens)
        
        # Scale up from sample to full dataset
        if sample_size < len(df):
            scale_factor = len(df) / sample_size
            data_tokens = int(data_tokens * scale_factor)
            structure_tokens = int(structure_tokens * scale_factor)
        
        # Processing overhead
        processing_overhead = _calculate_processing_overhead(df, encoding)
        
        total_tokens = header_tokens + data_tokens + structure_tokens + processing_overhead
        
        return {
            'headers': header_tokens,
            'data': data_tokens,
            'structure': structure_tokens,
            'processing_overhead': processing_overhead,
            'total': total_tokens
        }
        
    except Exception as e:
        logger.error(f"Detailed breakdown calculation failed: {e}")
        # Fallback to simple calculation
        total = calculate_tokens_from_dataframe(df, model_name, use_tiktoken=False)
        return {
            'headers': int(total * 0.1),
            'data': int(total * 0.7),
            'structure': int(total * 0.1),
            'processing_overhead': int(total * 0.1),
            'total': total
        }

def display_file_breakdown(file_breakdown: dict, file_name: str, show_detailed: bool = True, use_expander: bool = True):
    """Display detailed breakdown of file token usage"""
    import streamlit as st
    
    st.markdown(f"### 📊 Token Breakdown for '{file_name}' (tiktoken accurate):")
    
    # Main summary table
    breakdown_data = []
    for sheet_name, stats in file_breakdown.items():
        breakdown_data.append({
            "Sheet/Data": sheet_name,
            "Rows": f"{stats['rows']:,}",
            "Columns": stats['columns'],
            "Tokens": f"{stats['tokens']:,}"
        })
    
    breakdown_df = pd.DataFrame(breakdown_data)
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
    
    # Detailed breakdown if requested
    if show_detailed:
        st.markdown("#### 🔍 Detailed Token Analysis:")
        
        for sheet_name, stats in file_breakdown.items():
            detailed = stats.get('detailed', {})
            if detailed:
                # Only use expander if we're not already inside one
                if use_expander:
                    with st.expander(f"📋 {sheet_name} - Token Breakdown"):
                        _display_detailed_breakdown(sheet_name, stats, detailed)
                else:
                    st.markdown(f"**📋 {sheet_name} - Token Breakdown:**")
                    _display_detailed_breakdown(sheet_name, stats, detailed)
                    st.markdown("---")

def _display_detailed_breakdown(sheet_name: str, stats: dict, detailed: dict):
    """Helper function to display detailed breakdown without expander"""
    import streamlit as st
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Content Tokens:**")
        st.write(f"• Headers: {detailed.get('headers', 0):,}")
        st.write(f"• Data: {detailed.get('data', 0):,}")
        st.write(f"• Structure: {detailed.get('structure', 0):,}")
    
    with col2:
        st.markdown("**Processing:**")
        st.write(f"• Overhead: {detailed.get('processing_overhead', 0):,}")
        st.write(f"• **Total: {detailed.get('total', 0):,}**")
    
    # Token density metrics
    if stats['rows'] > 0 and stats['columns'] > 0:
        tokens_per_cell = detailed.get('total', 0) / (stats['rows'] * stats['columns'])
        st.markdown(f"**Efficiency:** ~{tokens_per_cell:.2f} tokens per cell")

def get_size_recommendations(max_tokens: int = 128000, model_name: str = "gpt-4"):
    """Get accurate file size recommendations using tiktoken estimation"""
    recommendations = []
    column_counts = [5, 10, 20, 50, 100, 200]
    
    try:
        encoding = get_tiktoken_encoding(model_name)
        
        for cols in column_counts:
            # More accurate estimation based on tiktoken patterns
            avg_tokens_per_cell = 2.5  # Conservative estimate for mixed data
            header_tokens = cols * 3  # Average tokens per column header
            structure_tokens_per_row = cols * 0.3  # Commas, quotes, etc.
            processing_overhead = 500 + (cols * 20)  # Schema + prompts
            
            # Available tokens for actual data
            available_for_data = max_tokens - processing_overhead - header_tokens
            
            # Tokens per row (data + structure)
            tokens_per_row = (cols * avg_tokens_per_cell) + structure_tokens_per_row
            
            # Calculate maximum rows
            max_rows = int(available_for_data / tokens_per_row) if tokens_per_row > 0 else 0
            max_rows = max(0, max_rows)  # Ensure non-negative
            
            recommendations.append(f"**{cols} columns**: Up to ~{max_rows:,} rows")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        # Fallback to simple calculation
        fallback_recommendations = []
        for cols in column_counts:
            # Simple estimation: tokens ≈ rows × columns × 5
            max_rows = max_tokens // (cols * 5)
            fallback_recommendations.append(f"**{cols} columns**: Up to ~{max_rows:,} rows (estimated)")
        return fallback_recommendations

def estimate_query_processing_tokens(user_question: str, schema_info: dict, sample_data: dict, model_name: str = "gpt-4") -> Dict[str, int]:
    """
    Estimate total tokens needed for processing a user query including all context
    
    Args:
        user_question: User's natural language question
        schema_info: Database schema information
        sample_data: Sample data from tables
        model_name: OpenAI model name
        
    Returns:
        Dict with token breakdown for query processing
    """
    try:
        encoding = get_tiktoken_encoding(model_name)
        
        # Question tokens
        question_tokens = len(encoding.encode(user_question))
        
        # Schema tokens
        schema_text = json.dumps(schema_info, indent=2)
        schema_tokens = len(encoding.encode(schema_text))
        
        # Sample data tokens
        sample_tokens = 0
        for table_name, df_sample in sample_data.items():
            if isinstance(df_sample, pd.DataFrame) and not df_sample.empty:
                # Convert sample to text representation
                sample_text = df_sample.to_csv(index=False)
                sample_tokens += len(encoding.encode(sample_text))
        
        # System prompt and instruction overhead
        system_prompt_tokens = 800  # Estimated tokens for system instructions
        
        # SQL generation and processing overhead
        sql_overhead = 300  # Tokens for SQL generation prompts
        
        total_tokens = {
            'question': question_tokens,
            'schema': schema_tokens,
            'samples': sample_tokens,
            'system_prompt': system_prompt_tokens,
            'sql_overhead': sql_overhead,
            'total': question_tokens + schema_tokens + sample_tokens + system_prompt_tokens + sql_overhead
        }
        
        return total_tokens
        
    except Exception as e:
        logger.error(f"Query token estimation failed: {e}")
        # Fallback estimation
        fallback_total = len(user_question) * 0.75 + 2000  # Conservative fallback
        return {
            'question': len(user_question) // 4,
            'schema': 500,
            'samples': 1000,
            'system_prompt': 800,
            'sql_overhead': 300,
            'total': int(fallback_total)
        }

def monitor_token_usage(df_dict: dict, model_name: str = "gpt-4") -> Dict[str, any]:
    """
    Monitor token usage across multiple DataFrames/tables
    
    Args:
        df_dict: Dictionary of DataFrames (table_name -> DataFrame)
        model_name: OpenAI model name
        
    Returns:
        Dict with comprehensive token usage statistics
    """
    try:
        total_tokens = 0
        table_stats = {}
        
        for table_name, df in df_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                table_tokens = calculate_tokens_from_dataframe(df, model_name=model_name)
                breakdown = get_detailed_token_breakdown(df, model_name)
                
                table_stats[table_name] = {
                    'tokens': table_tokens,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'breakdown': breakdown,
                    'tokens_per_row': table_tokens / len(df) if len(df) > 0 else 0,
                    'tokens_per_cell': table_tokens / (len(df) * len(df.columns)) if len(df) > 0 and len(df.columns) > 0 else 0
                }
                
                total_tokens += table_tokens
        
        # Overall statistics
        total_rows = sum(stats['rows'] for stats in table_stats.values())
        total_columns = sum(stats['columns'] for stats in table_stats.values())
        
        summary = {
            'total_tokens': total_tokens,
            'total_tables': len(table_stats),
            'total_rows': total_rows,
            'total_columns': total_columns,
            'avg_tokens_per_table': total_tokens / len(table_stats) if table_stats else 0,
            'table_stats': table_stats,
            'model_used': model_name
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Token monitoring failed: {e}")
        return {
            'total_tokens': 0,
            'total_tables': 0,
            'total_rows': 0,
            'total_columns': 0,
            'avg_tokens_per_table': 0,
            'table_stats': {},
            'model_used': model_name,
            'error': str(e)
        }

def get_token_efficiency_metrics(df: pd.DataFrame, model_name: str = "gpt-4") -> Dict[str, float]:
    """
    Calculate token efficiency metrics for a DataFrame
    
    Args:
        df: pandas DataFrame
        model_name: OpenAI model name
        
    Returns:
        Dict with efficiency metrics
    """
    if df.empty:
        return {'tokens_per_row': 0, 'tokens_per_cell': 0, 'tokens_per_kb': 0}
    
    try:
        total_tokens = calculate_tokens_from_dataframe(df, model_name=model_name)
        
        # Calculate data size in KB
        data_size_kb = df.memory_usage(deep=True).sum() / 1024
        
        metrics = {
            'tokens_per_row': total_tokens / len(df),
            'tokens_per_cell': total_tokens / (len(df) * len(df.columns)),
            'tokens_per_kb': total_tokens / data_size_kb if data_size_kb > 0 else 0,
            'compression_ratio': data_size_kb / (total_tokens * 4) if total_tokens > 0 else 0  # Assuming 4 chars per token
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Efficiency calculation failed: {e}")
        return {'tokens_per_row': 0, 'tokens_per_cell': 0, 'tokens_per_kb': 0, 'compression_ratio': 0}

def validate_token_limits_before_processing(file_path: str, max_tokens: int = 100000, model_name: str = "gpt-4") -> Dict[str, any]:
    """
    Validate token limits before processing a file (for local files)
    
    Args:
        file_path: Path to the file
        max_tokens: Maximum allowed tokens
        model_name: OpenAI model name
        
    Returns:
        Dict with validation results
    """
    try:
        file_ext = file_path.lower().split('.')[-1]
        
        if file_ext == 'csv':
            df = pd.read_csv(file_path)
            tokens = calculate_tokens_from_dataframe(df, model_name=model_name)
            
            return {
                'valid': tokens <= max_tokens,
                'tokens': tokens,
                'max_tokens': max_tokens,
                'percentage': (tokens / max_tokens) * 100,
                'file_type': 'csv',
                'sheets': {'main': {'rows': len(df), 'columns': len(df.columns), 'tokens': tokens}}
            }
            
        elif file_ext in ['xlsx', 'xls']:
            excel_file = pd.ExcelFile(file_path)
            total_tokens = 0
            sheets_info = {}
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheet_tokens = calculate_tokens_from_dataframe(df, model_name=model_name)
                total_tokens += sheet_tokens
                
                sheets_info[sheet_name] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'tokens': sheet_tokens
                }
            
            return {
                'valid': total_tokens <= max_tokens,
                'tokens': total_tokens,
                'max_tokens': max_tokens,
                'percentage': (total_tokens / max_tokens) * 100,
                'file_type': file_ext,
                'sheets': sheets_info
            }
        else:
            return {
                'valid': False,
                'error': f'Unsupported file type: {file_ext}',
                'tokens': 0,
                'max_tokens': max_tokens,
                'percentage': 0,
                'file_type': file_ext,
                'sheets': {}
            }
            
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'tokens': 0,
            'max_tokens': max_tokens,
            'percentage': 0,
            'file_type': 'unknown',
            'sheets': {}
        }