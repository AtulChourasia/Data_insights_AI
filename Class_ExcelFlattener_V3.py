import pandas as pd
import numpy as np
import warnings
import logging
from typing import Optional, Tuple, List, Union
import os

# Configure logging (consider making this configurable outside the class if used as a library)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExcelFlattener:
    """
    Dynamically detects, parses, and flattens complex Excel sheets or DataFrames.

    This class processes sheets with multi-level headers and/or multi-column
    row indices into a simple 'wide' or 'long' tabular format suitable for
    databases or further analysis.

    It uses heuristics to automatically detect the header and index boundaries,
    assuming headers/indices are primarily textual and the data block is
    primarily numeric or contains common placeholders (e.g., '-', blank).
    The accuracy of auto-detection depends on the sheet structure adhering
    to these patterns. Manual specification of header rows (h) and index
    columns (c) is also supported for precise control.

    Core Steps:
    1. Load raw data (preserving strings) or use provided DataFrame.
    2. Detect header/index split (h, c) using scoring heuristic (if not specified).
    3. Extract header, index, and data blocks.
    4. Build an intermediate multi-indexed DataFrame.
    5. Clean data (placeholders to NaN) and coerce to numeric types.
    6. Flatten the result to the desired 'wide' or 'long' format.
    """

    def __init__(
        self,
        data_input: Union[str, pd.DataFrame],
        sheet_name: Union[str, int, None] = 0, # Default to first sheet (only used for file paths)
        max_header_rows: int = 7,
        max_index_cols: int = 7
    ):
        """
        Initializes the ExcelFlattener.

        Args:
            data_input (Union[str, pd.DataFrame]): Either a path to Excel file or a DataFrame.
            sheet_name (Union[str, int, None]): Target sheet name or 0-based index.
                                                Only used when data_input is a file path.
            max_header_rows (int): Max rows considered for header during auto-detection.
            max_index_cols (int): Max columns considered for index during auto-detection.
        """
        self.data_input = data_input
        self.sheet_name = sheet_name
        self.max_header_rows = max_header_rows
        self.max_index_cols = max_index_cols

        self._df_raw: Optional[pd.DataFrame] = None # Stores raw data, lazy loaded

    def _load_raw_data(self) -> pd.DataFrame:
        """Loads the raw Excel/CSV file or processes the provided DataFrame, preserving strings and handling common NAs."""
        if self._df_raw is None:
            if isinstance(self.data_input, pd.DataFrame):
                # If input is already a DataFrame, use it directly
                logging.info("Using provided DataFrame")
                self._df_raw = self.data_input.copy()
                
                # Convert all columns to string type to match file loading behavior
                self._df_raw = self._df_raw.astype(str)
                
                # Reset column names to None (integer positions) to match file loading
                self._df_raw.columns = range(len(self._df_raw.columns))
                
                # Reset index to match file loading behavior
                self._df_raw = self._df_raw.reset_index(drop=True)
                
            else:
                # Original file loading logic
                logging.info(f"Loading data from '{self.data_input}'")
                try:
                    na_strings = [
                        '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN',
                        '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL',
                        'NaN', 'n/a', 'nan', 'null', ''
                    ]
        
                    # Check if path is an UploadedFile (e.g., from Streamlit)
                    if hasattr(self.data_input, "name") and hasattr(self.data_input, "read"):
                       
                        filename = self.data_input.name
                        file_ext = os.path.splitext(filename)[1].lower()
         
                        if file_ext in ['.xls', '.xlsx']:
                            self._df_raw = pd.read_excel(
                                self.data_input,
                                header=None,
                                dtype=str,
                                sheet_name=self.sheet_name,
                                keep_default_na=False
                            )
                        elif file_ext == '.csv':
                            self._df_raw = pd.read_csv(
                                self.data_input,
                                header=None,
                                dtype=str,
                                keep_default_na=False
                            )
                        else:
                            raise ValueError(f"Unsupported file format: {file_ext}")
                    else:
                        # Assume path is a string path
                        import os
                        file_ext = os.path.splitext(self.data_input)[1].lower()
         
                        if file_ext in ['.xls', '.xlsx']:
                            
                            self._df_raw = pd.read_excel(
                                self.data_input,
                                header=None,
                                dtype=str,
                                sheet_name=self.sheet_name,
                                keep_default_na=False
                            )
                        elif file_ext == '.csv':
                            
                            self._df_raw = pd.read_csv(
                                self.data_input,
                                header=None,
                                dtype=str,
                                keep_default_na=False
                            )
                          
                        else:
                            raise ValueError(f"Unsupported file format: {file_ext}")
                   
                    
         
                except FileNotFoundError:
                    logging.error(f"Error: File not found at {self.data_input}")
                    raise
                except Exception as e:
                    logging.error(f"Error reading file '{self.data_input}': {e}")
                    raise
            
            # Common processing for both DataFrame and file inputs
            na_strings = [
                '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN',
                '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL',
                'NaN', 'n/a', 'nan', 'null', ''
            ]
            self._df_raw = self._df_raw.replace(na_strings, np.nan)
                    
        return self._df_raw

    def _score_split(self, df_raw: pd.DataFrame, h: int, c: int) -> float:
        """
        Scores a potential header/index split (h, c) based on improved heuristics.

        Args:
            df_raw (pd.DataFrame): The raw DataFrame loaded as strings.
            h (int): Number of header rows being considered.
            c (int): Number of index columns being considered.

        Returns:
            float: Combined score between 0 and 1, or -1.0 if invalid.
        """
        # Invalid split if dimensions impossible
        if h >= df_raw.shape[0] or c >= df_raw.shape[1]:  # Fixed: allow c == df_raw.shape[1]-1
            return -1.0

        # Need at least one data row and one data column
        if df_raw.shape[0] - h <= 0 or df_raw.shape[1] - c <= 0:
            return -1.0

        # Define index vs. data blocks
        idx_block = df_raw.iloc[h:, :c] if c > 0 else pd.DataFrame()
        data_block = df_raw.iloc[h:, c:]

        # --- Score data block per column ---
        if data_block.empty:
            return -1.0
        
        val_scores = []
        for col in data_block.columns:
            col_data = data_block[col].dropna()  # Only consider non-null values
            if len(col_data) == 0:
                val_scores.append(0.0)
                continue
                
            # Improved numeric detection
            numeric_mask = pd.to_numeric(col_data, errors='coerce').notna()
            empty_indicators = col_data.isin(['-', '--', '', 'N/A', 'n/a', 'NA'])
            
            valid_fraction = (numeric_mask | empty_indicators).mean()
            val_scores.append(valid_fraction)
        
        val_score = sum(val_scores) / len(val_scores) if val_scores else 0.0

        # --- Score index block per column ---
        if c == 0:  # No index columns
            idx_score = 1.0
        else:
            idx_scores = []
            for col in idx_block.columns:
                col_data = idx_block[col].dropna()
                if len(col_data) == 0:
                    idx_scores.append(0.0)
                    continue
                    
                # Check for text (non-numeric) content
                numeric_mask = pd.to_numeric(col_data, errors='coerce').notna()
                text_fraction = (~numeric_mask).mean()
                
                # Bonus for uniqueness in index columns
                uniqueness = col_data.nunique() / len(col_data) if len(col_data) > 0 else 0
                
                # Combine text content with uniqueness
                combined_score = 0.7 * text_fraction + 0.3 * uniqueness
                idx_scores.append(combined_score)
            
            idx_score = sum(idx_scores) / len(idx_scores) if idx_scores else 0.0

        # Adjusted minimum validity thresholds
        min_val_threshold = 0.5  # Slightly lower for data columns
        min_idx_threshold = 0.4  # Lower for index columns (they might have some numbers)
        
        if val_score < min_val_threshold:
            return -1.0
        
        if c > 0 and idx_score < min_idx_threshold:  # Only check idx_score if we have index columns
            return -1.0

        # Improved scoring combination
        base_score = val_score * (0.3 + 0.7 * idx_score)  # Weight data columns more heavily
        
        # Refined penalty system
        total_cells = df_raw.shape[0] * df_raw.shape[1]
        consumed_cells = h * df_raw.shape[1] + (df_raw.shape[0] - h) * c
        penalty = max(0.1, 1 - (consumed_cells / total_cells))  # Ensure minimum penalty of 0.1
        
        return base_score * penalty

    def _detect_header_index_split(self) -> tuple[int, int]:
        """
        Auto-detects optimal number of header rows (h) and index columns (c)
        by combining header-content heuristics with split scoring.

        Returns:
            tuple[int, int]: The detected (h, c) split.
        """
        df_raw = self._load_raw_data()
        best_score = -1.0
        best_h, best_c = 0, 0

        # Determine search limits with better defaults
        max_h = min(self.max_header_rows, max(1, df_raw.shape[0] // 4))  # Max 25% of rows
        max_c = min(self.max_index_cols, max(1, df_raw.shape[1] // 3))   # Max 33% of columns

        # Explore all combinations including no header
        for h in range(1, max_h + 1):
            # --- Improved header heuristic score ---
            if h == 0:
                hdr_score = 0.5  # Neutral score for no headers
            else:
                ncols = df_raw.shape[1]
                metrics = []
                
                for r in range(h):
                    row = df_raw.iloc[r]
                    row_clean = row.dropna()
                    
                    if len(row_clean) == 0:
                        metrics.append((0, 0, 0))
                        continue
                    
                    # Non-blank ratio
                    nonblank = len(row_clean) / ncols
                    
                    # Text content ratio (non-numeric)
                    numeric_mask = pd.to_numeric(row_clean, errors='coerce').notna()
                    text = (~numeric_mask).mean()
                    
                    # Uniqueness ratio
                    uniq = row_clean.nunique() / len(row_clean)
                    
                    metrics.append((uniq, text, nonblank))
                
                if metrics:
                    avg_uniq = sum(m[0] for m in metrics) / len(metrics)
                    avg_text = sum(m[1] for m in metrics) / len(metrics)
                    avg_nonblank = sum(m[2] for m in metrics) / len(metrics)
                    
                    # Adjusted weights - emphasize text content and non-blank
                    hdr_score = 0.15 * avg_uniq + 0.45 * avg_text + 0.4 * avg_nonblank
                else:
                    hdr_score = 0.0

            for c in range(0, max_c + 1):
                split_score = self._score_split(df_raw, h, c)
                if split_score < 0:
                    continue
                
                # Improved combination: balance header quality with split quality
                if h == 0:
                    total = split_score  # Pure split score when no headers
                else:
                    total = 0.4 * hdr_score + 0.6 * split_score  # Weight split more heavily

                # Tie-breaking with preference for simpler solutions
                is_better = (
                    total > best_score or 
                    (abs(total - best_score) < 1e-4 and (h + c) < (best_h + best_c))
                )
                
                if is_better:
                    best_score = total
                    best_h, best_c = h, c
                
                # Early exit if very good score found
                if best_score > 0.95:
                    break
            
            if best_score > 0.95:
                break

        # Final validation and clipping
        h_final = max(0, min(best_h, df_raw.shape[0] - 1))
        c_final = max(0, min(best_c, df_raw.shape[1] - 1))
        
        # Ensure we have at least one data row and column
        if df_raw.shape[0] - h_final <= 0:
            h_final = max(0, df_raw.shape[0] - 1)
        if df_raw.shape[1] - c_final <= 0:
            c_final = max(0, df_raw.shape[1] - 1)
        
        logging.info(
            f"Detected split: header_rows={h_final}, index_cols={c_final} "
            f"(Score: {best_score:.4f})"
        )
        return (h_final, c_final)

    def _extract_blocks(self, df_raw: pd.DataFrame, h: int, c: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Slices the raw DataFrame into header, index, and data blocks based on (h, c).

        Args:
            df_raw (pd.DataFrame): The raw DataFrame.
            h (int): Number of header rows.
            c (int): Number of index columns.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the
            header_block, index_block, and data_block DataFrames. Returns empty
            DataFrames for blocks that don't exist (e.g., index_block if c=0).
        """
        logging.debug(f"Extracting blocks with h={h}, c={c}")
        rows, cols = df_raw.shape

        # Header block: Starts at (0, c). Use ffill(axis=1) to handle merged header cells.
        header_block = df_raw.iloc[:h, c:].ffill(axis=1) if c < cols and h > 0 else pd.DataFrame()

        # Index block: Starts at (h, 0). Reset index for alignment.
        index_block = df_raw.iloc[h:, :c].reset_index(drop=True) if c > 0 and h < rows else pd.DataFrame()

        # Data block: Starts at (h, c). Reset index for alignment.
        data_block = df_raw.iloc[h:, c:].reset_index(drop=True) if c < cols and h < rows else pd.DataFrame()

        return header_block, index_block, data_block

    def _build_multiindexed_dataframe(self, df_raw: pd.DataFrame, h: int, c: int) -> pd.DataFrame:
        """
        Constructs the core multi-indexed DataFrame before final cleaning/flattening.

        Uses the detected/provided h and c to define the row and column multi-indices.
        No extra text columns are identified or removed.

        Args:
            df_raw (pd.DataFrame): The raw DataFrame.
            h (int): Number of header rows.
            c (int): Number of base index columns.

        Returns:
            pd.DataFrame: A DataFrame with a (potentially multi-level) row index
                          and a (potentially multi-level) column index. Row index levels
                          are named 'idx_0', 'idx_1', ... Column index levels (if any)
                          are named 'lvl0', 'lvl1', ...
        """
        logging.debug("Building multi-indexed DataFrame")

        # --- Determine All Row Index Columns ---
        all_index_cols = list(range(c))

        # --- Create Row Index ---
        if not all_index_cols:
            first_measure_col = 0 # All columns are potentially measures
            # Use a simple RangeIndex reflecting original row position if no index cols found
            row_multi_index = pd.RangeIndex(start=h, stop=df_raw.shape[0], name="original_row")
        else:
            # First column *not* part of the index determines start of measures
            first_measure_col = max(all_index_cols) + 1
            # Extract index data, ensure string type, fill NaNs for robust multi-index creation
            row_index_df = df_raw.iloc[h:, all_index_cols].reset_index(drop=True)
            row_index_df = row_index_df.astype(str).fillna('') # Use empty string for NaN in index keys

            # Create index (single or multi-level), naming levels 'idx_0', 'idx_1', ...
            if row_index_df.shape[1] == 1:
                row_multi_index = pd.Index(row_index_df.iloc[:, 0], name="idx_0")
            else:
                row_multi_index = pd.MultiIndex.from_frame(row_index_df, names=[f"idx_{i}" for i in range(len(all_index_cols))])

        # --- Create Column Index ---
        if first_measure_col >= df_raw.shape[1]:
            logging.warning("No measure columns identified after accounting for all index columns.")
            # Return DataFrame with only the row index if no data columns remain
            return pd.DataFrame(index=row_multi_index)

        # Extract header data for the measure columns
        header_for_cols = df_raw.iloc[:h, first_measure_col:]
        # Fill merged header cells rightward before creating index
        header_for_cols = header_for_cols.ffill(axis=1)
        # Ensure strings and fill NaNs for consistent tuple creation
        header_for_cols = header_for_cols.astype(str).fillna('')

        # Create column index (RangeIndex, single Index, or MultiIndex)
        if h == 0: # No header rows
            col_index = pd.RangeIndex(start=first_measure_col, stop=df_raw.shape[1])
            # Provide a default name for stacking later if needed
            col_index = col_index.rename("header_level_0")
        elif h == 1: # Single header row
             # Name level 'lvl0' for consistency with multi-level and long format stack
            col_index = pd.Index(header_for_cols.iloc[0], name="lvl0")
        else: # Multi-level header
            # Create tuples from header rows, name levels 'lvl0', 'lvl1', ...
            tuples = list(zip(*(header_for_cols.iloc[r] for r in range(h))))
            col_index = pd.MultiIndex.from_tuples(tuples, names=[f"lvl{r}" for r in range(h)])

        # --- Extract Data and Build DataFrame ---
        body_data = df_raw.iloc[h:, first_measure_col:].reset_index(drop=True)

        # Create the main DataFrame structure
        body_df = pd.DataFrame(body_data.values, index=row_multi_index, columns=col_index)

        return body_df

    def _clean_and_coerce(self, body_df: pd.DataFrame) -> pd.DataFrame:
        """Cleans placeholder strings and coerces data columns to numeric."""
        if body_df.empty:
            return body_df

        logging.debug("Cleaning and coercing data block")
        df = body_df.copy() # Work on a copy

        # Define placeholders to replace with NaN
        # Regex for '-' or '--' potentially surrounded by whitespace
        placeholders_regex = r'^\s*--?\s*$'
        # Specific strings (Excel errors, etc.) - '' was handled in _load_raw_data
        specific_placeholders = ['#VALUE!', '#DIV/0!']

        # --- Suppress FutureWarning from replace potentially downcasting ---
        # This warning occurs because replacing strings with NaN might change
        # an 'object' column to 'float', which is considered downcasting.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)

            # Perform replacements
            df = df.replace(placeholders_regex, np.nan, regex=True)
            df = df.replace(specific_placeholders, np.nan)

            # Optional: Explicitly infer types after replace. Not strictly
            # necessary as to_numeric follows, but doesn't hurt.
            try:
                df = df.infer_objects(copy=False)
            except TypeError: # Handle older pandas versions
                df = df.infer_objects()
        # --- End warning suppression ---

        # Attempt numeric conversion for all columns in the body DataFrame.
        # Assumes these are intended measure columns. Non-numeric become NaN.
        df = df.apply(pd.to_numeric, errors='coerce', axis=0)

        return df

    def _flatten_result(self, cleaned_df: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Flattens the cleaned, multi-indexed DataFrame to 'wide' or 'long' format.

        Args:
            cleaned_df (pd.DataFrame): The DataFrame after cleaning and coercion.
            method (str): 'wide' or 'long'.

        Returns:
            pd.DataFrame: The final flattened DataFrame.
        """
        if cleaned_df.empty:
             # Return empty frame but preserve index names if they exist
             index_names = []
             if isinstance(cleaned_df.index, pd.MultiIndex):
                 index_names = [name for name in cleaned_df.index.names if name is not None]
             elif cleaned_df.index.name:
                  index_names = [cleaned_df.index.name]
             return pd.DataFrame(columns=index_names)

        logging.debug(f"Flattening result using method: {method}")

        if method == 'wide':
            # Wide format: Concatenate column level values, reset row index to columns.
            flat_df = cleaned_df.copy()
            if isinstance(flat_df.columns, pd.MultiIndex):
                # Join levels with '__', skipping empty/NA levels and stripping whitespace.
                flat_df.columns = [
                    '__'.join(str(level).strip() for level in col_tuple if pd.notna(level) and str(level).strip())
                    for col_tuple in flat_df.columns.values
                ]
            else: # Single level column index
                 flat_df.columns = [str(col).strip() for col in flat_df.columns]
            # Move row index levels (idx_0, idx_1, ...) into columns
            return flat_df.reset_index()

        elif method == 'long':
            # Long format: 'Stack' column levels into rows.
            if isinstance(cleaned_df.columns, pd.MultiIndex):
                # Stack all column levels (lvl0, lvl1, ...)
                stacked = cleaned_df.stack(level=list(range(cleaned_df.columns.nlevels)),
                                           future_stack=True) # dropna=False is implicit
            else: # Single level column index
                 level_name = cleaned_df.columns.name if cleaned_df.columns.name else 0
                 stacked = cleaned_df.stack(level=level_name, future_stack=True) # dropna=False is implicit

            # Stack results in a Series; rename the data part to 'Value'.
            long_df = stacked.rename('Value')
            # Move all index levels (original row idx_* + stacked column lvl*) into columns.
            long_df = long_df.reset_index()
            # Re-apply numeric coercion as stacking can sometimes change dtypes.
            long_df['Value'] = pd.to_numeric(long_df['Value'], errors='coerce')
            return long_df

        else:
            # Should not be reachable if validation in flatten() works, but good practice.
            raise ValueError(f"Invalid flatten method: '{method}'. Must be 'wide' or 'long'.")

    def flatten(
        self,
        h: Optional[int] = None,
        c: Optional[int] = 1,
        method: str = 'wide'
    ) -> pd.DataFrame:
        """
        Loads, processes, and flattens the specified Excel sheet or DataFrame.

        This is the main public method orchestrating the entire process.

        Args:
            h (Optional[int]): Manually specify the number of header rows (>=0).
                               If None, auto-detection is performed.
            c (Optional[int]): Manually specify the number of index columns (>=0).
                               If None, auto-detection is performed.
            method (str): Output format: 'wide' or 'long'. Defaults to 'wide'.

        Returns:
            pd.DataFrame: The processed and flattened DataFrame.

        Raises:
            FileNotFoundError: If the Excel file path is invalid.
            ValueError: If parameters (h, c, method) are invalid.
            Exception: For errors during Excel reading or processing steps.
        """
        # --- Parameter Validation ---
        if method not in ['wide', 'long']:
            raise ValueError(f"Invalid method '{method}'. Choose 'wide' or 'long'.")

        try:
            # --- Step 0: Load Data ---
            df_raw = self._load_raw_data()
           
            if df_raw.empty:
                logging.warning("Input data is empty.")
                return pd.DataFrame() # Return empty DataFrame

            # --- Step 1: Determine Header/Index Split ---
            if h is None or c is None:
                detected_h, detected_c = self._detect_header_index_split()
                h = h if h is not None else detected_h
                c = c if c is not None else detected_c
                logging.info(f"Using detected/default split: header_rows={h}, index_cols={c}")
            else:
                # Validate manually provided h and c
                logging.info(f"Using provided split: header_rows={h}, index_cols={c}")
                if not (isinstance(h, int) and h >= 0 and h < df_raw.shape[0]):
                     raise ValueError(f"Provided h={h} is invalid for data with {df_raw.shape[0]} rows.")
                # c can equal shape[1] if there are only index columns specified
                if not (isinstance(c, int) and c >= 0 and c <= df_raw.shape[1]):
                     raise ValueError(f"Provided c={c} is invalid for data with {df_raw.shape[1]} columns.")

            # --- Step 2: Extract Blocks ---
            header_block, index_block, data_block = self._extract_blocks(df_raw, h, c)
            
            # --- Step 3: Build Multi-indexed DataFrame ---
            body_df = self._build_multiindexed_dataframe(df_raw, h, c)
            
            # --- Step 4: Clean Data & Coerce Types ---
            cleaned_df = self._clean_and_coerce(body_df)
           
            # --- Step 5: Flatten to Final Format ---
            final_df = self._flatten_result(body_df, method)

            logging.info(f"Flattening complete. Result shape: {final_df.shape}")
            return final_df

        except Exception as e:
            # Log any exception during the process and re-raise
            logging.exception(f"Error during flattening process:")
            raise e
        
    @staticmethod
    def read_instructions(filepath: str, encoding: str = 'utf-8') -> str:
        """
        Utility method to read instruction files.
        
        Args:
            filepath (str): Path to the instruction file.
            encoding (str): File encoding, defaults to 'utf-8'.
            
        Returns:
            str: Contents of the instruction file.
        """
        with open(filepath, 'r', encoding=encoding) as f: 
            return f.read()