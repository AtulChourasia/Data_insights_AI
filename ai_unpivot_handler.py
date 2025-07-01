# ai_unpivot_handler.py
import time
import pandas as pd
import sqlite3
import streamlit as st
import logging
import json
from backend import preprocess_column_name, fetch_sql_result
# from frontend_copy_5 import refresh_displayed_data_after_unpivot


logger = logging.getLogger(__name__)

class AIUnpivotHandler:
    """AI-powered handler for intelligent unpivoting of tables"""
    
    def __init__(self, llm, db_path="data_store.db"):
        self.llm = llm
        self.db_path = db_path
    
    def analyze_table_for_unpivot(self, table_name, sample_df):
        """
        Use AI to analyze if a table should be unpivoted and how
        """
        # Get column information
        numeric_cols = sample_df.select_dtypes(include=['number']).columns.tolist()
        non_numeric_cols = sample_df.select_dtypes(exclude=['number']).columns.tolist()
        
        # Create a data sample for AI analysis
        sample_data = sample_df.head(5).to_string()
        
        prompt = f"""
        Analyze this table to determine if it's a pivot table that should be unpivoted:

        Table Name: {table_name}
        
        Columns:
        - Non-numeric columns: {', '.join(non_numeric_cols)}
        - Numeric columns: {', '.join(numeric_cols)}
        
        Sample Data:
        {sample_data}
        
        Please analyze and provide a JSON response with the following structure:
        {{
            "is_pivot_table": true/false,
            "confidence": 0-100,
            "reasoning": "explanation of why this is or isn't a pivot table",
            "suggested_index_columns": ["column1", "column2"],
            "suggested_value_columns": ["column3", "column4"],
            "suggested_variable_name": "suggested name for the variable column",
            "suggested_value_name": "suggested name for the value column",
            "transformation_benefits": "explanation of benefits if unpivoted"
        }}
        
        Consider:
        1. Are the numeric columns representing time periods, categories, or measurements that could be better as rows?
        2. Would unpivoting make the data easier to query and analyze?
        3. What columns uniquely identify each entity (good index columns)?
        4. What would be meaningful names for the unpivoted columns?
        
        Return ONLY the JSON object, no additional text.
        """
        
        try:
            response = self.llm.invoke(prompt).content.strip()
            
            # Clean response if it contains markdown
            if response.startswith("```json"):
                response = response.split("```json")[1]
            if response.startswith("```"):
                response = response.split("```")[1]
            if "```" in response:
                response = response.split("```")[0]
            
            analysis = json.loads(response)
            return analysis
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
            # Return default analysis
            return {
                "is_pivot_table": len(numeric_cols) > 3 and len(non_numeric_cols) <= 3,
                "confidence": 50,
                "reasoning": "Could not complete AI analysis, using heuristic detection",
                "suggested_index_columns": non_numeric_cols[:2] if non_numeric_cols else [],
                "suggested_value_columns": numeric_cols,
                "suggested_variable_name": "metric",
                "suggested_value_name": "value",
                "transformation_benefits": "May simplify querying if this is time-series or category data"
            }
    
    def refresh_displayed_data_after_unpivot():
        """
        Refresh the displayed data in session state to show unpivoted versions
        where applicable, while maintaining the original structure for non-unpivoted tables.
        """
        import streamlit as st
        import sqlite3
        import pandas as pd
        from backend import preprocess_column_name, fetch_sql_result, get_database_schema
        import logging
        
        logger = logging.getLogger(__name__)
        
        try:
            # Get all tables from the database
            conn = sqlite3.connect("data_store.db")
            cursor = conn.cursor()
            tables_in_db = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            table_names = [t[0] for t in tables_in_db]
            conn.close()
            
            # For each file in session state
            for file_name, file_info in st.session_state.files_dict.items():
                # For each sheet in the file
                for sheet_name in list(file_info["df_dict"].keys()):
                    # Get the table name that corresponds to this sheet
                    if "table_mapping" in file_info and sheet_name in file_info["table_mapping"]:
                        original_table_name = file_info["table_mapping"][sheet_name]
                        original_table_name = preprocess_column_name(original_table_name)
                        
                        # Check if an unpivoted version exists
                        unpivoted_table_name = f"{original_table_name}_unpivoted"
                        
                        if unpivoted_table_name in table_names:
                            # Load the unpivoted data
                            unpivoted_df = fetch_sql_result("data_store.db", f"SELECT * FROM {unpivoted_table_name}")
                            
                            # Update the displayed dataframe with the unpivoted version
                            st.session_state.files_dict[file_name]["df_dict"][sheet_name] = unpivoted_df
                            
                            # Also update the table samples
                            st.session_state.table_samples[unpivoted_table_name] = fetch_sql_result(
                                "data_store.db", 
                                f"SELECT * FROM {unpivoted_table_name} LIMIT 20"
                            )
                            
                            # Remove the old table from samples if it exists
                            if original_table_name in st.session_state.table_samples:
                                del st.session_state.table_samples[original_table_name]
                            
                            # Update the table mapping to point to the unpivoted table
                            st.session_state.files_dict[file_name]["table_mapping"][sheet_name] = unpivoted_table_name
                            
                            # Update the main table_dict to use unpivoted table
                            # Find the key in table_dict that matches this table
                            for key, value in list(st.session_state.table_dict.items()):
                                if value == original_table_name:
                                    st.session_state.table_dict[key] = unpivoted_table_name
                                    break
                        
            # Update schema info
            st.session_state.schema_info = get_database_schema()
            
        except Exception as e:
            logger.error(f"Error refreshing displayed data after unpivot: {str(e)}")
            st.error(f"Error refreshing data: {str(e)}")

    def get_optimized_unpivot_plan(self, table_name, sample_df, user_context=""):
        """
        Get AI-optimized unpivoting plan based on table structure and user context
        """
        # First, analyze if it's a pivot table
        analysis = self.analyze_table_for_unpivot(table_name, sample_df)
        
        # If not a pivot table with high confidence, return None
        if not analysis['is_pivot_table'] or analysis['confidence'] < 60:
            return None
        
        # Get more detailed unpivoting plan
        prompt = f"""
        Create an optimized unpivoting plan for this pivot table:

        Table: {table_name}
        Current Analysis: {json.dumps(analysis, indent=2)}
        User Context: {user_context if user_context else "General data analysis"}
        
        Sample Data:
        {sample_df.head(5).to_string()}
        
        Provide a detailed unpivoting plan as JSON:
        {{
            "index_columns": ["col1", "col2"],
            "value_columns": ["col3", "col4"],
            "variable_column_name": "descriptive_name",
            "value_column_name": "descriptive_name",
            "remove_nulls": true/false,
            "create_indexes": ["suggested_index_columns"],
            "expected_benefits": ["benefit1", "benefit2"],
            "sample_queries": ["example SQL query 1", "example SQL query 2"]
        }}
        
        Make column names SQL-friendly and descriptive.
        
        Return ONLY the JSON object.
        """
        
        try:
            response = self.llm.invoke(prompt).content.strip()
            
            # Clean response
            if response.startswith("```json"):
                response = response.split("```json")[1]
            if response.startswith("```"):
                response = response.split("```")[1]
            if "```" in response:
                response = response.split("```")[0]
            
            plan = json.loads(response)
            
            # Validate and clean the plan
            plan['index_columns'] = [col for col in plan.get('index_columns', []) if col in sample_df.columns]
            plan['value_columns'] = [col for col in plan.get('value_columns', []) if col in sample_df.columns]
            
            # Ensure we have valid columns
            if not plan['index_columns']:
                plan['index_columns'] = analysis['suggested_index_columns']
            if not plan['value_columns']:
                plan['value_columns'] = analysis['suggested_value_columns']
            
            # If still no valid columns, return None
            if not plan['index_columns'] or not plan['value_columns']:
                logger.warning(f"No valid columns found for unpivoting {table_name}")
                return None
            
            # Clean column names
            plan['variable_column_name'] = preprocess_column_name(plan.get('variable_column_name', 'variable'))
            plan['value_column_name'] = preprocess_column_name(plan.get('value_column_name', 'value'))
            
            plan['analysis'] = analysis
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating unpivot plan: {str(e)}")
            # Return basic plan based on initial analysis
            if analysis['suggested_index_columns'] and analysis['suggested_value_columns']:
                return {
                    "index_columns": analysis['suggested_index_columns'],
                    "value_columns": analysis['suggested_value_columns'],
                    "variable_column_name": preprocess_column_name(analysis.get('suggested_variable_name', 'variable')),
                    "value_column_name": preprocess_column_name(analysis.get('suggested_value_name', 'value')),
                    "remove_nulls": True,
                    "create_indexes": analysis['suggested_index_columns'][:1] if analysis['suggested_index_columns'] else [],
                    "expected_benefits": [analysis['transformation_benefits']],
                    "sample_queries": [],
                    "analysis": analysis
                }
            else:
                return None
    
    def execute_unpivot_with_plan(self, table_name, plan, progress_callback=None):
        """
        Execute the unpivoting based on the AI-generated plan
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Read the full table
            if progress_callback:
                progress_callback(0.1, "Reading table from database...")
            
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            original_shape = df.shape
            
            if progress_callback:
                progress_callback(0.3, "Applying AI-optimized unpivoting...")
            
            # Perform unpivot with AI-suggested parameters
            unpivoted_df = pd.melt(
                df,
                id_vars=plan['index_columns'],
                value_vars=plan['value_columns'],
                var_name=plan['variable_column_name'],
                value_name=plan['value_column_name']
            )
            
            # Remove nulls if suggested
            if plan.get('remove_nulls', True):
                unpivoted_df = unpivoted_df.dropna(subset=[plan['value_column_name']])
            
            final_shape = unpivoted_df.shape
            
            if progress_callback:
                progress_callback(0.6, "Creating backup...")
            
            # Create backup
            backup_table_name = f"{table_name}_pivot_backup"
            df.to_sql(backup_table_name, conn, if_exists='replace', index=False)
            
            if progress_callback:
                progress_callback(0.8, "Saving unpivoted table...")
            
            # Replace original table
            unpivoted_df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            # Create suggested indexes
            for idx_col in plan.get('create_indexes', []):
                if idx_col in unpivoted_df.columns:
                    clean_col = preprocess_column_name(idx_col)
                    try:
                        conn.execute(f"CREATE INDEX idx_{table_name}_{clean_col} ON {table_name}({clean_col})")
                    except:
                        pass
            
            conn.commit()
            conn.close()
            
            if progress_callback:
                progress_callback(1.0, "Unpivoting complete!")
            
            result_info = {
                'original_shape': original_shape,
                'final_shape': final_shape,
                'backup_table': backup_table_name,
                'plan_used': plan
            }
            
            return True, "Table unpivoted successfully with AI optimization", result_info
            
        except Exception as e:
            logger.error(f"Error executing unpivot: {str(e)}")
            return False, f"Error unpivoting table: {str(e)}", None



def render_ai_unpivot_phase(table_dict, table_samples, llm):
    """
    Simplified Phase 2: Show all tables with checkboxes for unpivot selection
    AI automatically determines best unpivot strategy and column names
    """
    import streamlit as st
    import pandas as pd
    import sqlite3
    import logging
    import json
    from backend import preprocess_column_name, fetch_sql_result
    
    logger = logging.getLogger(__name__)
    
    st.markdown("## 📊 Phase 2: Table Transformation")
    st.markdown("Select which tables need to be unpivoted (transformed from wide to long format)")
    
    # Initialize session state
    if 'unpivot_selections' not in st.session_state:
        st.session_state.unpivot_selections = {}
    
    # Initialize selections for new tables
    for table_name in table_dict.values():
        if table_name not in st.session_state.unpivot_selections:
            st.session_state.unpivot_selections[table_name] = {
                'selected': False
            }
    
    # Display all tables with selection checkboxes
    # st.markdown("### 📋 Available Tables")
    
    # Create a container for the table list
    table_container = st.container()
    
    with table_container:
        # Show summary metrics
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.metric("Total Tables", len(table_dict))
        # with col2:
        #     selected_count = sum(1 for t in st.session_state.unpivot_selections.values() if t.get('selected', False))
        #     st.metric("Selected for Unpivot", selected_count)
        
        st.markdown("---")
        # col1, col2 = st.columns([1,19])
        # with col1:
        #     if st.button("☑️", use_container_width=True):
        #         for table_name in st.session_state.unpivot_selections:
        #             st.session_state.unpivot_selections[table_name]['selected'] = True
        #         st.rerun()
        # with col2:
        #     pass
        # Display each table with checkbox and info
        for table_name in sorted(table_dict.values()):
            if table_name in table_samples:
                sample_df = table_samples[table_name]
                
                # Create columns for table display
                col1, col2, col3 = st.columns([4, 3, 1])
                
                with col1:
                    # Checkbox for selection
                    selected = st.checkbox(
                        f"**{table_name}**",
                        value=st.session_state.unpivot_selections[table_name].get('selected', False),
                        key=f"select_{table_name}",
                        help=f"Shape: {sample_df.shape[0]} rows × {sample_df.shape[1]} columns"
                    )
                    
                    # Update selection in session state
                    st.session_state.unpivot_selections[table_name]['selected'] = selected
                
                with col2:
                    # Show table dimensions
                    st.markdown(f"📊 **{sample_df.shape[0]} rows × {sample_df.shape[1]} columns**")
                
                with col3:
                    # Preview button
                    if st.button("👁️ Preview", key=f"preview_{table_name}"):
                        st.session_state[f"show_preview_{table_name}"] = not st.session_state.get(f"show_preview_{table_name}", False)
                
                # Show preview if requested
                if st.session_state.get(f"show_preview_{table_name}", False):
                    with st.expander(f"Preview: {table_name}", expanded=True):
                        st.dataframe(sample_df.head(10), use_container_width=True, height=200)
        
        st.markdown("---")
    
    # Action buttons
    col1, col2 = st.columns([2, 2])
    
    with col1:
        # Process button
        if st.button("🔄 **Unpivot Selected Tables**", type="primary", use_container_width=True):
            selected_tables = [
                table_name for table_name, info in st.session_state.unpivot_selections.items()
                if info.get('selected', False)
            ]
            
            if not selected_tables:
                st.warning("⚠️ Please select at least one table to unpivot")
            else:
                # Process selected tables
                with st.spinner(f"Unpivoting {len(selected_tables)} table(s)..."):
                    success_count = 0
                    error_count = 0
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, table_name in enumerate(selected_tables):
                        try:
                            status_text.text(f"🤖 AI analyzing {table_name}...")
                            
                            # Fetch full data from database
                            full_df = fetch_sql_result("data_store.db", f"SELECT * FROM {table_name}")
                            
                            # Get AI recommendations for unpivoting
                            unpivot_strategy = get_ai_unpivot_strategy(full_df, table_name, llm)
                            
                            if not unpivot_strategy['should_unpivot']:
                                logger.info(f"AI decided not to unpivot {table_name}: {unpivot_strategy['reason']}")
                                error_count += 1
                                continue
                            
                            status_text.text(f"🔄 Unpivoting {table_name}...")
                            
                            # Perform unpivot using AI-determined strategy
                            unpivoted_df = pd.melt(
                                full_df,
                                id_vars=unpivot_strategy['id_columns'],
                                value_vars=unpivot_strategy['value_columns'],
                                var_name=unpivot_strategy['var_name'],
                                value_name=unpivot_strategy['value_name']
                            )
                            
                            # Clean the unpivoted data
                            unpivoted_df = unpivoted_df.dropna(subset=[unpivot_strategy['value_name']])
                            
                            # Apply any column renaming suggested by AI
                            if 'column_renames' in unpivot_strategy:
                                unpivoted_df = unpivoted_df.rename(columns=unpivot_strategy['column_renames'])
                            
                            if not unpivoted_df.empty:
                                # Use AI-suggested table name
                                new_table_name = preprocess_column_name(unpivot_strategy['new_table_name'])
                                
                                # Store in SQLite (replacing the original)
                                conn = sqlite3.connect("data_store.db")
                                
                                # First, drop the original table
                                try:
                                    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                                    logger.info(f"Dropped original table: {table_name}")
                                except Exception as e:
                                    logger.warning(f"Could not drop original table {table_name}: {e}")
                                
                                # Store the unpivoted version with the original table name
                                unpivoted_df.to_sql(table_name, conn, if_exists='replace', index=False)
                                conn.close()
                                
                                # Update session state - replace the original table
                                st.session_state.table_samples[table_name] = unpivoted_df.head(20)
                                
                                success_count += 1
                                logger.info(f"Successfully unpivoted and replaced {table_name}")
                                logger.info(f"  Strategy: {unpivot_strategy['description']}")
                                logger.info(f"  Original shape: {full_df.shape} -> New shape: {unpivoted_df.shape}")
                            else:
                                error_count += 1
                                logger.error(f"Failed to unpivot {table_name}: Empty result after removing nulls")
                        
                        except Exception as e:
                            error_count += 1
                            logger.error(f"Error unpivoting {table_name}: {str(e)}")
                            st.error(f"❌ Error processing {table_name}: {str(e)}")
                        
                        # Update progress
                        progress = (idx + 1) / len(selected_tables)
                        progress_bar.progress(progress)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Show results
                    if success_count > 0:
                        st.success(f"✅ Successfully unpivoted {success_count} table(s)")
                        
                        # Show what was transformed
                        with st.expander("📋 Tables transformed", expanded=True):
                            st.write("The following tables were unpivoted and replaced:")
                            for table_name in selected_tables:
                                if table_name in [t for t in selected_tables if st.session_state.unpivot_selections[t].get('selected', False)]:
                                    st.write(f"• **{table_name}** (transformed in-place)")
                    
                    if error_count > 0:
                        st.error(f"❌ Failed to unpivot {error_count} table(s)")
                    
                    if success_count > 0:
                        # Update table_dict to reflect current database state
                        conn = sqlite3.connect("data_store.db")
                        cursor = conn.cursor()
                        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
                        conn.close()
                        
                        # Rebuild table_dict with current tables
                        st.session_state.table_dict = {table[0]: table[0] for table in tables}
                        
                        # Mark phase as complete
                        st.session_state.unpivot_phase_complete = True
                        
                        # Auto-refresh after 2 seconds
                        st.info("🚀 Moving to Phase 3: Query & Analysis...")
                        import time
                        time.sleep(2)
                        st.rerun()
    
    with col2:
        # Skip button
        if st.button("⏭️ **Skip Unpivoting**", use_container_width=True):
            st.session_state.unpivot_phase_complete = True
            st.info("Skipping unpivot phase...")
            import time
            time.sleep(1)
            st.rerun()
    
    
        # Select all/none buttons
        
        
        
    
    # Information box
    with st.expander("ℹ️ What is unpivoting?", expanded=False):
        st.markdown("""
        **Unpivoting** transforms wide-format tables (where data is spread across columns) into long-format tables 
        (where data is stacked in rows). This is useful for:
        
        - 📊 **Pivot tables** with metrics spread across date/category columns
        - 📈 **Cross-tabulations** that need to be normalized
        - 🔄 **Wide data** that's difficult to query with SQL
        
        The AI automatically:
        - Detects which columns should be ID columns vs value columns
        - Names the unpivoted columns appropriately
        - Creates SQL-friendly table and column names
        """)
    
    return st.session_state.get('unpivot_phase_complete', False)


def get_ai_unpivot_strategy(df, table_name, llm):
    """
    Use AI to determine the best unpivot strategy for a table
    """
    import numpy as np
    
    # Create a sample of the data for AI analysis
    sample_data = df.head(10).to_string()
    columns_info = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = int(df[col].nunique())  # Convert to Python int
        null_count = int(df[col].isnull().sum())  # Convert to Python int
        
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
        
        columns_info.append({
            'column': col,
            'dtype': dtype,
            'unique_count': unique_count,
            'null_count': null_count,
            'sample_values': str(converted_values)
        })
    
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
        "var_name": "metric_name",  // Name for the variable column (e.g., "month", "category", "metric_type")
        "value_name": "metric_value",  // Name for the value column (e.g., "amount", "value", "count")
        "new_table_name": "suggested_table_name",
        "column_renames": {{"old_name": "new_name"}},  // Optional: rename columns for clarity
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
    
    try:
        response = llm.invoke(prompt).content.strip()
        
        # Clean up response to ensure it's valid JSON
        if response.startswith("```json"):
            response = response.split("```json")[1]
        if response.startswith("```"):
            response = response.split("```")[1]
        if "```" in response:
            response = response.split("```")[0]
        
        strategy = json.loads(response)
        
        # Validate the response
        required_keys = ['should_unpivot', 'id_columns', 'value_columns', 'var_name', 'value_name', 'new_table_name']
        for key in required_keys:
            if key not in strategy:
                raise ValueError(f"Missing required key: {key}")
        
        # Ensure columns exist in the dataframe
        strategy['id_columns'] = [col for col in strategy['id_columns'] if col in df.columns]
        strategy['value_columns'] = [col for col in strategy['value_columns'] if col in df.columns]
        
        # If no valid columns found, don't unpivot
        if not strategy['id_columns'] or not strategy['value_columns']:
            strategy['should_unpivot'] = False
            strategy['reason'] = "Could not identify valid ID or value columns"
        
        return strategy
        
    except Exception as e:
        # Fallback strategy if AI fails
        logger.warning(f"AI unpivot strategy failed for {table_name}: {str(e)}")
        
        # Simple heuristic fallback
        id_columns = []
        value_columns = []
        
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() < len(df) * 0.1:
                id_columns.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                value_columns.append(col)
        
        if not id_columns and len(df.columns) > 0:
            id_columns = [df.columns[0]]
        
        return {
            'should_unpivot': len(value_columns) > 1,
            'reason': 'Fallback strategy used',
            'id_columns': id_columns,
            'value_columns': value_columns,
            'var_name': 'metric',
            'value_name': 'value',
            'new_table_name': f"{table_name}_unpivoted",
            'description': 'Standard unpivot transformation'
        }