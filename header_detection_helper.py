# Replace the entire content of header_detection_helper.py with this:

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def simple_header_detector(df):
    """
    Simple header detector that finds header rows by comparing data types
    with the last row (assumed to be representative data).
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        dict: {
            'header_row_indices': list of header row indices,
            'confidence': confidence score (0.0 to 1.0)
        }
    """
    if df.empty or len(df) == 0:
        return {'header_row_indices': [], 'confidence': 0.0}
    
    # Convert DataFrame to list of lists (like the JS tableData)
    table_data = []
    for i in range(len(df)):
        row = []
        for j in range(len(df.columns)):
            cell_value = df.iloc[i, j]
            # Convert to string representation like JS version
            if pd.isna(cell_value):
                row.append("")
            else:
                row.append(str(cell_value))
        table_data.append(row)
    
    # Get the last row as reference (expected data format)
    last_row = table_data[-1]
    expected_column_count = len(last_row)
    expected_types = [get_cell_type(cell) for cell in last_row]
    
    first_data_row_idx = -1
    
    # Find first row that matches the expected data types
    for i in range(len(table_data)):
        row = table_data[i]
        if len(row) != expected_column_count:
            continue
            
        row_types = [get_cell_type(cell) for cell in row]
        
        if types_match(expected_types, row_types):
            first_data_row_idx = i
            break
    
    # Header rows are all rows before the first data row
    if first_data_row_idx > 0:
        header_row_indices = list(range(first_data_row_idx))
    else:
        header_row_indices = []
    
    confidence = 1.0 if len(header_row_indices) > 0 else 0.0
    
    return {
        'header_row_indices': header_row_indices,
        'confidence': confidence
    }

def get_cell_type(cell):
    """
    Determine the type of a cell value.
    
    Args:
        cell: Cell value (string)
        
    Returns:
        str: 'empty', 'number', or 'string'
    """
    if cell is None or cell == "":
        return "empty"
    
    try:
        float(cell)
        return "number"
    except (ValueError, TypeError):
        return "string"

def types_match(ref_types, row_types):
    """
    Check if two lists of types match exactly.
    
    Args:
        ref_types: Reference type list
        row_types: Row type list to compare
        
    Returns:
        bool: True if types match, False otherwise
    """
    if len(ref_types) != len(row_types):
        return False
    
    for i in range(len(ref_types)):
        if ref_types[i] != row_types[i]:
            return False
    
    return True

def detect_header_rows_automatically(df, max_rows=10):
    """
    Main function to detect header rows in a DataFrame using the simple algorithm.
    This replaces the old detect_header_rows_automatically function.
    
    Args:
        df: DataFrame to analyze
        max_rows: Maximum number of header rows to consider (kept for compatibility)
        
    Returns:
        int: Number of header rows detected (minimum 1)
    """
    if df.empty or len(df) < 2:
        return 1
    
    result = simple_header_detector(df)
    
    header_count = len(result['header_row_indices'])
    confidence = result['confidence']
    
    logger.info(f"Simple header detection - Found {header_count} header row(s), confidence: {confidence:.1%}")
    
    # Return at least 1 header row (fallback)
    return max(1, header_count)