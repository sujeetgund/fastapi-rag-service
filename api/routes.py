from fastapi import APIRouter, Depends, HTTPException
import logging
from models.schemas import QuestionAnswerRequest, QuestionAnswerResponse
from core.security import verify_token
from services.document_service import DocumentService
from services.qa_service import QAService
from core.exceptions import DocumentProcessingError, QAServiceError

logger = logging.getLogger(__name__)

qa_router = APIRouter(prefix="/hackrx", tags=["hackrx"])

# Dependency to get services
async def get_document_service() -> DocumentService:
    from main import document_service
    if not document_service:
        raise HTTPException(status_code=500, detail="Document service not initialized")
    return document_service

async def get_qa_service() -> QAService:
    from main import qa_service
    if not qa_service:
        raise HTTPException(status_code=500, detail="QA service not initialized")
    return qa_service

@qa_router.post("/run", response_model=QuestionAnswerResponse)
async def run_qa(
    request: QuestionAnswerRequest,
    token: str = Depends(verify_token),
    document_service: DocumentService = Depends(get_document_service),
    qa_service: QAService = Depends(get_qa_service)
):
    """
    Process a document and answer questions based on its content using Google Generative AI.
    
    - **documents**: URL to the PDF document to process
    - **questions**: List of questions to answer based on the document
    """
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        
        if not request.questions:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        # Process document
        vectorstore = await document_service.process_document(request.documents)
        
        # Answer questions
        answers = await qa_service.answer_questions(vectorstore, request.questions)
        
        logger.info("Request processed successfully")
        return QuestionAnswerResponse(answers=answers)
        
    except (DocumentProcessingError, QAServiceError) as e:
        logger.error(f"Service error: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
