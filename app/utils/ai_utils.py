import openai
import os
import logging
from typing import List, Dict, Any, Optional, Union
import json
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_text_with_ai(
    text: str,
    task: str = "analyze",
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> Dict[str, Any]:
    """
    Analyze text using OpenAI's API.
    
    Args:
        text: The text to analyze
        task: The type of analysis to perform
        model: The OpenAI model to use
        temperature: Controls randomness (0-2)
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Dictionary containing the analysis results
    """
    try:
        # Prepare the prompt based on the task
        if task == "summarize":
            prompt = f"Please summarize the following text:\n\n{text}"
        elif task == "sentiment":
            prompt = f"Analyze the sentiment of the following text (positive, negative, or neutral):\n\n{text}"
        elif task == "keywords":
            prompt = f"Extract the most important keywords from the following text:\n\n{text}"
        else:  # Default to general analysis
            prompt = f"Analyze the following text and provide insights:\n\n{text}"
        
        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract the response
        result = {
            "analysis": response.choices[0].message['content'],
            "model": response.model,
            "usage": dict(response.usage),
            "task": task
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_text_with_ai: {e}")
        raise


def generate_sql_from_natural_language(
    question: str,
    schema_info: Dict[str, Any],
    model: str = "gpt-4",
    temperature: float = 0.3
) -> str:
    """
    Generate a SQL query from a natural language question.
    
    Args:
        question: The natural language question
        schema_info: Database schema information
        model: The OpenAI model to use
        temperature: Controls randomness (0-2)
        
    Returns:
        Generated SQL query
    """
    try:
        # Prepare the schema information
        schema_str = json.dumps(schema_info, indent=2)
        
        # Create the prompt
        prompt = f"""Given the following database schema:
        
        {schema_str}
        
        Please generate a SQL query to answer the following question:
        
        {question}
        
        Return ONLY the SQL query, without any additional text or explanation.
        """
        
        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a SQL expert. Generate SQL queries based on the given schema and question."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=500
        )
        
        # Extract the SQL query
        sql_query = response.choices[0].message['content'].strip()
        
        # Clean up the SQL query
        if sql_query.startswith("```sql"):
            sql_query = sql_query[7:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        
        return sql_query.strip()
        
    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        raise


def analyze_dataframe(
    df: pd.DataFrame,
    question: str = None,
    model: str = "gpt-4",
    max_rows: int = 100
) -> Dict[str, Any]:
    """
    Analyze a pandas DataFrame using AI.
    
    Args:
        df: The DataFrame to analyze
        question: Optional specific question to answer about the data
        model: The OpenAI model to use
        max_rows: Maximum number of rows to include in the analysis
        
    Returns:
        Dictionary containing the analysis results
    """
    try:
        # Sample the DataFrame if it's too large
        if len(df) > max_rows:
            df_sample = df.sample(min(max_rows, len(df)))
        else:
            df_sample = df.copy()
        
        # Convert DataFrame to a string representation
        data_str = df_sample.to_string()
        
        # Prepare the prompt
        if question:
            prompt = f"""Given the following dataset:
            
            {data_str}
            
            Please answer the following question: {question}
            
            Provide a detailed analysis and any relevant insights.
            """
        else:
            prompt = f"""Analyze the following dataset and provide insights:
            
            {data_str}
            
            Please include:
            1. Summary statistics
            2. Data quality issues
            3. Interesting patterns or trends
            4. Any recommendations for further analysis
            """
        
        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data analyst. Analyze the given data and provide clear, actionable insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1500
        )
        
        # Extract the analysis
        analysis = response.choices[0].message['content']
        
        # Prepare the result
        result = {
            "analysis": analysis,
            "model": response.model,
            "tokens_used": response.usage.total_tokens,
            "data_summary": {
                "rows": len(df),
                "columns": len(df.columns),
                "columns_list": df.columns.tolist(),
                "sample_size": len(df_sample)
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing DataFrame: {e}")
        raise


def detect_sensitive_data(column_data: List[Any]) -> Dict[str, Any]:
    """
    Detect if a column contains sensitive data (PII, etc.).
    
    Args:
        column_data: List of values from a column
        
    Returns:
        Dictionary with detection results
    """
    try:
        # Convert to strings and take a sample
        sample = [str(x) for x in column_data[:1000]]  # Limit to first 1000 values
        sample_str = "\n".join(sample[:10])  # Use first 10 for the prompt
        
        prompt = f"""Analyze the following sample data and determine if it contains any sensitive information:
        
        {sample_str}
        
        Please respond with a JSON object containing:
        - is_sensitive (boolean): Whether the data appears to be sensitive
        - data_type (string): The type of data (e.g., 'email', 'phone', 'ssn', 'name', 'address', 'other')
        - confidence (float): Your confidence level (0-1)
        - reason (string): Brief explanation of your assessment
        """
        
        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data security expert. Analyze the data for sensitive information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        # Parse the response
        try:
            result = json.loads(response.choices[0].message['content'])
            return result
        except json.JSONDecodeError:
            # If the response isn't valid JSON, return a default result
            return {
                "is_sensitive": False,
                "data_type": "unknown",
                "confidence": 0.0,
                "reason": "Unable to determine if data is sensitive"
            }
            
    except Exception as e:
        logger.error(f"Error detecting sensitive data: {e}")
        return {
            "is_sensitive": False,
            "data_type": "error",
            "confidence": 0.0,
            "reason": f"Error during detection: {str(e)}"
        }
