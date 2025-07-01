import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import pandas as pd
import openai
import os
from dotenv import load_dotenv
import json

from ..schemas.ai_schemas import AIResponse, AIAnalysisResult
from ..config.database import SessionLocal

logger = logging.getLogger(__name__)
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

class AIService:
    def __init__(self, db: Session):
        self.db = db
        self.default_model = "gpt-4"
    
    async def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None, model: str = None) -> AIResponse:
        """Generate a response using the AI model."""
        try:
            model = model or self.default_model
            
            # Prepare messages
            messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
            
            # Add context if provided
            if context:
                messages.append({"role": "system", "content": f"Context: {json.dumps(context, indent=2)}"})
            
            # Add user prompt
            messages.append({"role": "user", "content": prompt})
            
            # Call OpenAI API
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract the response
            answer = response.choices[0].message['content']
            
            return AIResponse(
                answer=answer,
                metadata={
                    "model": model,
                    "tokens_used": response.usage.total_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            raise ValueError(f"Failed to generate response: {e}")
    
    async def analyze_data(self, data: List[Dict[str, Any]], question: str) -> AIAnalysisResult:
        """Analyze data and provide insights based on a question."""
        try:
            # Convert data to string for the prompt
            data_str = "\n".join([str(item) for item in data[:10]])  # Limit to first 10 rows
            
            prompt = f"""Analyze the following data and answer the question.
            
            Data (first 10 rows):
            {data_str}
            
            Question: {question}
            
            Please provide a detailed analysis and answer the question based on the data.
            """
            
            response = await self.generate_response(prompt)
            
            return AIAnalysisResult(
                success=True,
                result={"answer": response.answer},
                execution_time=0.0,  # TODO: Track execution time
                model_used=self.default_model,
                tokens_used=response.metadata.get("tokens_used", 0)
            )
            
        except Exception as e:
            logger.error(f"Error in analyze_data: {e}")
            return AIAnalysisResult(
                success=False,
                error=str(e),
                execution_time=0.0,
                model_used=self.default_model,
                tokens_used=0
            )
    
    async def generate_sql(self, question: str, schema_info: Dict[str, Any]) -> str:
        """Generate SQL query based on natural language question and schema information."""
        try:
            prompt = f"""Given the following database schema information:
            
            {json.dumps(schema_info, indent=2)}
            
            Generate a SQL query to answer the following question:
            {question}
            
            Return ONLY the SQL query, without any additional text or explanation.
            """
            
            response = await self.generate_response(prompt)
            return response.answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise ValueError(f"Failed to generate SQL: {e}")

# Factory function to get AI service
def get_ai_service():
    db = SessionLocal()
    try:
        yield AIService(db)
    finally:
        db.close()
