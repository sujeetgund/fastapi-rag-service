from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from core.config import settings
from api.routes import qa_router
from services.document_service import DocumentService
from services.qa_service import QAService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global services
document_service = None
qa_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global document_service, qa_service
    
    logger.info("Initializing services...")
    try:
        document_service = DocumentService()
        qa_service = QAService()
        await document_service.initialize()
        await qa_service.initialize()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    logger.info("Shutting down services...")
    if document_service:
        await document_service.cleanup()
    if qa_service:
        await qa_service.cleanup()

app = FastAPI(
    title="Document Q&A API with LangChain",
    description="API for answering questions based on document content using Google Generative AI",
    version="1.0.0",
    lifespan=lifespan,
    root_path="/api/v1"
)

# Include routers
app.include_router(qa_router)

@app.get("/")
async def root():
    return {
        "message": "RAG API for Hackrx",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )