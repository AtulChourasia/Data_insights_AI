from fastapi import APIRouter
from ..controllers import file_controller, query_controller, ai_controller, table_controller

# Create API router
api_router = APIRouter()

# Include all routers
api_router.include_router(
    file_controller.router,
    prefix="/files",
    tags=["files"]
)

api_router.include_router(
    query_controller.router,
    prefix="/queries",
    tags=["queries"]
)

api_router.include_router(
    ai_controller.router,
    prefix="/ai",
    tags=["ai"]
)

api_router.include_router(
    table_controller.router,
    prefix="/tables",
    tags=["tables"]
)
