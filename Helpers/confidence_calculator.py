
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import re
import json
from sqlalchemy import text
from database import engine  

class EnhancedConfidenceCalculator:
    """
    Multi-dimensional confidence scoring for text-to-SQL systems
    Compatible with your SQLAlchemy backend
    """
   
    def __init__(self, embeddings_model, llm):
        self.embeddings = embeddings_model
        self.llm = llm
       
    def calculate_confidence(
        self,
        user_query: str,
        sql_query: str,
        schema_info: Dict = None,
        table_samples: Dict = None
    ) -> Dict[str, float]:
        """Calculate multi-dimensional confidence score"""
        scores = {}
       
        # 1. Semantic Similarity
        scores['semantic'] = self._semantic_confidence(user_query, sql_query)
       
        # 2. Execution Validation (using your backend's approach)
        scores['execution'] = self._execution_confidence(sql_query)
       
        # 3. LLM-based comparison
        scores['llm_comparison'] = self._llm_comparison_confidence(user_query, sql_query)
       
        weights = {
            'semantic': 0.7,           # Intent matching
            'execution': 0.1,          # Execution validation
            'llm_comparison': 0.2,     # LLM-based comparison
                     
        }
       
        overall = sum(scores[key] * weights[key] for key in scores.keys())
        scores['overall'] = round(overall, 2)
       
        return scores
   
    def _semantic_confidence(self, user_query: str, sql_query: str) -> float:
        """Improved semantic similarity with multiple approaches"""
        try:
            # Method 1: Direct query-to-SQL embedding similarity
            user_embed = np.array(self.embeddings.embed_query(user_query))
            sql_embed = np.array(self.embeddings.embed_query(sql_query))
            direct_sim = np.dot(user_embed, sql_embed) / (
                np.linalg.norm(user_embed) * np.linalg.norm(sql_embed)
            )
           
            # Method 2: SQL-to-NL translation
            prompt = f"""Convert this SQL to a natural language question. Be concise.
            SQL: {sql_query}
            Question:"""
           
            sql_nl = self.llm.invoke(prompt).content.strip()
            sql_nl_embed = np.array(self.embeddings.embed_query(sql_nl))
            translation_sim = np.dot(user_embed, sql_nl_embed) / (
                np.linalg.norm(user_embed) * np.linalg.norm(sql_nl_embed)
            )
           
            # Combine both methods (weighted average)
            combined_sim = 0.6 * direct_sim + 0.4 * translation_sim
            confidence = round(((combined_sim + 1) / 2) * 100, 2)
           
            return min(100, max(0, confidence))
           
        except Exception as e:
            print(f"Semantic confidence error: {e}")
            return 50.0
   
    def _execution_confidence(self, sql_query: str) -> float:
        """Test SQL execution using the EXACT same method as  main backend"""
        try:
            # Use the exact same approach as your ask_question endpoint
            df_result = pd.read_sql_query(text(sql_query), engine)
            
            # Check if execution was successful
            if df_result is None:
                return 0.0
            
            # Check for error patterns in the results
            if len(df_result) == 1 and len(df_result.columns) == 1:
                cell_value = str(df_result.iloc[0, 0]).lower()
                if any(error_word in cell_value for error_word in ['error', 'failed', 'invalid']):
                    return 0.0
            
            # Check for suspicious SQL patterns (security)
            sql_lower = sql_query.lower().strip()
            dangerous_patterns = [
                'drop table', 'delete from', 'truncate table', 'alter table',
                'create table', 'insert into', 'update set', 'drop database'
            ]
            
            if any(pattern in sql_lower for pattern in dangerous_patterns):
                return 0.0
            
            # Successful execution
            return 100.0
                   
        except Exception as e:
            # Analyze the error type for better confidence scoring
            error_msg = str(e).lower()
            
            # SQL syntax errors - very low confidence
            if any(pattern in error_msg for pattern in ['syntax error', 'near', 'unexpected token']):
                return 0.0
            
            # Schema/table errors - low confidence but query structure might be valid
            elif any(pattern in error_msg for pattern in ['no such table', 'no such column', 'table', 'column']):
                return 15.0
            
            # Other database errors
            else:
                return 5.0
   
    def _llm_comparison_confidence(self, user_query: str, sql_query: str) -> float:
        """Compare user query and SQL query using LLM"""
        try:
            prompt = f"""
Compare the user query and SQL query to determine how well they match.
 
User Query: "{user_query}"
SQL Query: "{sql_query}"
 
Rate the match quality from 0-100 based on:
- Do the main entities/concepts align?
- Are the requested actions properly represented?
- Are filters and conditions correctly captured?
- Is the overall intent preserved?
 
Return only a numeric confidence score (0-100):
"""
           
            response = self.llm.invoke(prompt).content.strip()
           
            # Extract numeric score from response
            score_match = re.search(r'\b(\d{1,3})\b', response)
            if score_match:
                confidence = int(score_match.group(1))
                return min(100, max(0, confidence))
            else:
                return 50.0
               
        except Exception as e:
            print(f"LLM comparison error: {e}")
            return 50.0
    

# Usage function that matches your backend pattern
def calculate_query_confidence(user_query: str, sql_query: str, llm, embeddings) -> Dict[str, float]:
    """
    Convenience function that matches your backend's function signature
    """
    calculator = EnhancedConfidenceCalculator(embeddings, llm)
    return calculator.calculate_confidence(user_query, sql_query)

