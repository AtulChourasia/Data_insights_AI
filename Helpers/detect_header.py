import pandas as pd
import numpy as np

def detect_and_read_headers(df, max_header_rows=5, min_data_rows=10):
    """
    Detects the number of header rows in a DataFrame and returns a properly formatted DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to analyze
    max_header_rows (int): Maximum number of rows to consider as potential headers
    min_data_rows (int): Minimum number of data rows required after headers
    
    Returns:
    tuple: (cleaned_dataframe, num_header_rows, header_info)
    """
    
    if len(df) < min_data_rows:
        return df, 0, "DataFrame too small to analyze"
    
    def is_likely_header_row(row):
        """Check if a row is likely to be a header based on various criteria"""
        # Convert to string and check characteristics
        str_values = row.astype(str)
        
        # Count non-null, non-empty values
        non_empty = sum(1 for val in str_values if val and str(val).strip() and str(val) != 'nan')
        
        if non_empty == 0:
            return False
            
        # Check for header-like characteristics
        header_indicators = 0
        
        for val in str_values:
            val_str = str(val).strip().lower()
            if not val_str or val_str == 'nan':
                continue
                
            # Look for common header patterns
            if any(keyword in val_str for keyword in ['total', 'sum', 'count', 'average', 'mean', 'id', 'name', 'date', 'time']):
                header_indicators += 1
            
            # Check if it's mostly text (not numbers)
            try:
                float(val_str)
            except (ValueError, TypeError):
                header_indicators += 1
        
        # Header likely if most values are text-like
        return header_indicators >= non_empty * 0.6
    
    def has_consistent_data_types(start_row, sample_size=5):
        """Check if rows starting from start_row have consistent data types"""
        if start_row >= len(df) - sample_size:
            return False
            
        sample_rows = df.iloc[start_row:start_row + sample_size]
        
        # Check each column for type consistency
        consistent_cols = 0
        total_cols = len(df.columns)
        
        for col in df.columns:
            col_data = sample_rows[col].dropna()
            if len(col_data) < 2:
                continue
                
            # Try to determine if column has consistent numeric data
            numeric_count = 0
            for val in col_data:
                try:
                    float(str(val))
                    numeric_count += 1
                except (ValueError, TypeError):
                    pass
            
            # Column is consistent if it's mostly numeric or mostly text
            if numeric_count >= len(col_data) * 0.8 or numeric_count <= len(col_data) * 0.2:
                consistent_cols += 1
        
        return consistent_cols >= total_cols * 0.6
    
    # Analyze potential header rows
    best_header_rows = 0
    best_score = 0
    
    for num_headers in range(max_header_rows + 1):
        if num_headers >= len(df) - min_data_rows:
            break
            
        score = 0
        
        # Check if the rows before data start look like headers
        if num_headers > 0:
            header_rows = df.iloc[:num_headers]
            header_score = sum(1 for _, row in header_rows.iterrows() if is_likely_header_row(row))
            score += header_score * 2
        
        # Check if data after headers has consistent types
        if has_consistent_data_types(num_headers):
            score += 3
            
        # Prefer fewer header rows if scores are equal
        if score > best_score:
            best_score = score
            best_header_rows = num_headers
    
    # Create the properly formatted DataFrame
    if best_header_rows == 0:
        cleaned_df = df.copy()
        header_info = "No header rows detected"
    else:
        # Use the detected header rows to create column names
        header_rows = df.iloc[:best_header_rows]
        data_rows = df.iloc[best_header_rows:]
        
        # Combine header rows into column names
        if best_header_rows == 1:
            new_columns = [str(col) for col in header_rows.iloc[0]]
        else:
            new_columns = []
            for col_idx in range(len(df.columns)):
                col_parts = []
                for row_idx in range(best_header_rows):
                    val = str(header_rows.iloc[row_idx, col_idx])
                    if val and val != 'nan':
                        col_parts.append(val.strip())
                new_columns.append('_'.join(col_parts) if col_parts else f'Column_{col_idx}')
        
        # Create cleaned DataFrame
        cleaned_df = data_rows.copy()
        cleaned_df.columns = new_columns
        cleaned_df = cleaned_df.reset_index(drop=True)
        
        header_info = f"Detected {best_header_rows} header row(s)"
    
    return cleaned_df, best_header_rows, header_info

# Example usage function for reading from file
def read_file_with_auto_headers(file_path, **kwargs):
    """
    Read a file and automatically detect header rows.
    
    Parameters:
    file_path (str): Path to the file
    **kwargs: Additional arguments to pass to pd.read_csv/pd.read_excel
    
    Returns:
    tuple: (cleaned_dataframe, num_header_rows, header_info)
    """
    # Read file without assuming headers
    if file_path.endswith('.csv'):
        raw_df = pd.read_csv(file_path, header=None, **kwargs)
    elif file_path.endswith(('.xlsx', '.xls')):
        raw_df = pd.read_excel(file_path, header=None, **kwargs)
    else:
        raise ValueError("Unsupported file format")
    
    return detect_and_read_headers(raw_df)


# cleaned_df, num_headers, info = detect_and_read_headers(sample_data)
   