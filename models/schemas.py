from pydantic import BaseModel
from typing import List

class QuestionAnswerRequest(BaseModel):
    documents: str  
    questions: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": "https://example.com/document.pdf",
                "questions": [
                    "What is the main topic of this document?",
                    "What are the key findings?"
                ]
            }
        }

class QuestionAnswerResponse(BaseModel):
    answers: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "answers": [
                    "The main topic is...",
                    "The key findings are..."
                ]
            }
        }

class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None