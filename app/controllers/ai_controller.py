from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional

from ..services.ai_service import AIService, get_ai_service
from ..schemas.ai_schemas import (
    QuestionRequest,
    IterativeQuestionRequest,
    InsightsRequest,
    AIResponse,
    AIAnalysisResult,
    UnpivotTablesRequest
)

router = APIRouter()

@router.post("/ask/", response_model=AIResponse)
async def ask_question(
    request: QuestionRequest,
    ai_service: AIService = Depends(get_ai_service)
):
    """
    Ask a question and get an AI-generated response.
    """
    try:
        return await ai_service.generate_response(
            prompt=request.question,
            context=request.context,
            model=request.model
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )

@router.post("/ask/iterative/", response_model=AIResponse)
async def ask_question_iterative(
    request: IterativeQuestionRequest,
    ai_service: AIService = Depends(get_ai_service)
):
    """
    Ask a question with iterative refinement based on previous attempts.
    """
    try:
        # Add previous attempts to the context
        context = request.context or {}
        if request.previous_attempts:
            context["previous_attempts"] = request.previous_attempts
        
        return await ai_service.generate_response(
            prompt=request.question,
            context=context,
            model=request.model
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in iterative question answering: {str(e)}"
        )

@router.post("/analyze/", response_model=AIAnalysisResult)
async def analyze_data(
    request: InsightsRequest,
    ai_service: AIService = Depends(get_ai_service)
):
    """
    Analyze data and provide insights based on a question.
    """
    try:
        return await ai_service.analyze_data(
            data=request.data,
            question=request.question
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing data: {str(e)}"
        )

@router.post("/unpivot/", response_model=AIAnalysisResult)
async def unpivot_tables(
    request: UnpivotTablesRequest,
    ai_service: AIService = Depends(get_ai_service)
):
    """
    Unpivot tables using AI to determine the best strategy.
    """
    try:
        # This would typically interact with the AI service to determine
        # the best unpivoting strategy and apply it
        return {
            "success": True,
            "result": {
                "message": "Unpivoting functionality will be implemented here",
                "tables": request.table_names
            },
            "model_used": "gpt-4",
            "tokens_used": 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in unpivoting tables: {str(e)}"
        )

@router.post("/generate-sql/", response_model=Dict[str, str])
async def generate_sql(
    question: str,
    schema_info: Dict[str, Any],
    ai_service: AIService = Depends(get_ai_service)
):
    """
    Generate SQL from natural language question and schema information.
    """
    try:
        sql = await ai_service.generate_sql(question, schema_info)
        return {"sql": sql}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error generating SQL: {str(e)}"
        )

@router.get("/health/", response_model=Dict[str, str])
async def health_check():
    """
    Check if the AI service is healthy.
    """
    return {"status": "healthy"}
