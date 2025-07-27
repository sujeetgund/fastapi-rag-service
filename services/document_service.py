from typing import List
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from core.config import settings
from core.exceptions import DocumentProcessingError

from uuid import uuid4
import tempfile
import requests
import time


logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self):
        self.embeddings = None
        self.text_splitter = None
        self.document_loader = None
        self.document_cache = {}

    async def initialize(self):
        """Initialize the document service"""
        try:
            # Initialize embeddings
            if not settings.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required")

            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.GOOGLE_EMBEDDING_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                task_type="QUESTION_ANSWERING",
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

    def extract_documents_from_pdf(self, pdf_url: str) -> List[Document]:
        """Extract documents from PDF URL"""

        try:
            # Try to load PDF directly from URL
            logger.info(f"Loading PDF directly from URL: {pdf_url}")
            self.document_loader = PyPDFLoader(pdf_url)
            documents = self.document_loader.load()
            if not documents:
                raise DocumentProcessingError("No documents loaded from PDF")
            return documents
        except Exception as direct_load_error:
            # Fallback: download PDF and load from local file
            logger.warning(f"Direct loading failed! Attempting to download PDF...")

            start_time = time.time()
            resp = requests.get(pdf_url)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(resp.content)
                tmp_path = f.name
            print(
                "=" * 60,
                f"\nTime taken to download PDF: {time.time() - start_time:.2f} seconds\n",
                "=" * 60,
            )

            self.document_loader = PyPDFLoader(tmp_path)
            documents = self.document_loader.load()

            if not documents:
                raise DocumentProcessingError("No documents loaded from PDF (fallback)")

            return documents
        except Exception as e:
            raise DocumentProcessingError(f"Error processing PDF: {str(e)}")

    async def process_document(self, doc_url: str) -> FAISS:
        """Process document and create vector store"""

        # Check cache first
        if settings.ENABLE_CACHE and doc_url in self.document_cache:
            logger.info("Using cached document")
            vectorstore_name = self.document_cache[doc_url]
            logger.info(f"Loading vector store from cache: {vectorstore_name}")
            return FAISS.load_local(
                vectorstore_name, self.embeddings, allow_dangerous_deserialization=True
            )

        try:
            # Extract documents from PDF
            logger.info("Extracting documents from PDF")
            start_time = time.time()
            documents = self.extract_documents_from_pdf(doc_url)
            print(
                "=" * 60,
                f"\nTime taken to extract documents: {time.time() - start_time:.2f} seconds\n",
                "=" * 60,
            )

            # Split text into chunks
            logger.info("Creating text chunks")
            start_time = time.time()
            text_chunks = self.text_splitter.split_documents(documents)
            print(
                "=" * 60,
                f"\nTime taken to create text chunks: {time.time() - start_time:.2f} seconds\n",
                "=" * 60,
            )

            if not text_chunks:
                raise DocumentProcessingError("No text chunks created from document")

            # Create vector store
            logger.info("Creating vector store")
            start_time = time.time()
            vectorstore = FAISS.from_documents(
                documents=documents, embedding=self.embeddings
            )
            print(
                "=" * 60,
                f"\nTime taken to create vector store: {time.time() - start_time:.2f} seconds\n",
                "=" * 60,
            )

            # Cache the result
            if settings.ENABLE_CACHE:
                logger.info("Saving vector store to cache")
                start_time = time.time()
                vectorstore_name = f"vectorstore_{uuid4()}"
                vectorstore.save_local(vectorstore_name)
                self.document_cache[doc_url] = vectorstore_name
                print(
                    "=" * 60,
                    f"\nTime taken to cache vector store: {time.time() - start_time:.2f} seconds\n",
                    "=" * 60,
                )

            logger.info("Document processing completed successfully")
            return vectorstore

        except DocumentProcessingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing document: {e}")
            raise DocumentProcessingError(f"Unexpected error: {str(e)}")
