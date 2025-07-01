from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

from .config.config import init_app
from .config.database import Base, engine
from .routers import api as api_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Data Insights AI API",
    description="API for Data Insights AI application",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize application
init_app(app)

# Include API router
app.include_router(api_router.api_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    """Initialize services when the application starts."""
    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint that provides API information."""
    return {
        "message": "Welcome to Data Insights AI API",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# For development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
