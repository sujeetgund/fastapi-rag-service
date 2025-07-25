import httpx
import PyPDF2
import io
from typing import List, Tuple, Optional
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
from core.config import settings
from core.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self):
        self.embeddings = None
        self.text_splitter = None
        self.document_cache = {}
        
    async def initialize(self):
        """Initialize the document service"""
        try:
            # Initialize embeddings
            if not settings.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required")
                
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.GOOGLE_EMBEDDING_MODEL,
                google_api_key=settings.GOOGLE_API_KEY
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            
            logger.info("Document service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize document service: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        self.document_cache.clear()
        logger.info("Document service cleaned up")
    
    async def download_pdf(self, url: str) -> bytes:
        """Download PDF from URL"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.content
            except httpx.RequestError as e:
                raise DocumentProcessingError(f"Error downloading document: {str(e)}")
            except httpx.HTTPStatusError as e:
                raise DocumentProcessingError(f"HTTP error downloading document: {str(e)}")
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            if not text.strip():
                raise DocumentProcessingError("No text content found in the PDF")
                
            return text.strip()
            
        except Exception as e:
            raise DocumentProcessingError(f"Error processing PDF: {str(e)}")
    
    async def process_document(self, doc_url: str) -> FAISS:
        """Process document and create vector store"""
        
        # Check cache first
        if settings.ENABLE_CACHE and doc_url in self.document_cache:
            logger.info("Using cached document")
            return self.document_cache[doc_url]
        
        try:
            # Download and extract text
            logger.info(f"Downloading document from {doc_url}")
            pdf_content = await self.download_pdf(doc_url)
            
            logger.info("Extracting text from PDF")
            document_text = self.extract_text_from_pdf(pdf_content)
            
            # Split text into chunks
            logger.info("Creating text chunks")
            text_chunks = self.text_splitter.split_text(document_text)
            
            if not text_chunks:
                raise DocumentProcessingError("No text chunks created from document")
            
            # Create documents
            documents = [
                Document(page_content=chunk, metadata={"source": doc_url, "chunk_id": i})
                for i, chunk in enumerate(text_chunks)
            ]
            
            # Create vector store
            logger.info("Creating vector store")
            vectorstore = FAISS.from_documents(documents=documents, embedding=self.embeddings, docstore=InMemoryDocstore())
            
            # Cache the result
            if settings.ENABLE_CACHE:
                self.document_cache[doc_url] = vectorstore
                logger.info("Document processed and cached")
            
            logger.info("Document processing completed successfully")
            return vectorstore
            
        except DocumentProcessingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing document: {e}")
            raise DocumentProcessingError(f"Unexpected error: {str(e)}")
