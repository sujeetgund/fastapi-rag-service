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
import fitz  # PyMuPDF for PDF processing
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
        start_time = time.time()

        try:
            # Try to load PDF directly from URL
            logger.info(f"Loading PDF directly from URL: {pdf_url}")
            direct_load_time = time.time()

            self.document_loader = PyPDFLoader(pdf_url)
            documents = self.document_loader.load()

            logger.info(
                f"Directly loaded {len(documents)} pages in {time.time() - direct_load_time:.2f} seconds"
            )

            if not documents:
                raise DocumentProcessingError("No documents loaded from PDF")
            return documents
        except Exception as direct_load_error:
            # Fallback: download PDF and load from local file
            logger.warning(f"Direct loading failed! Attempting to download PDF...")
            download_time = time.time()

            resp = requests.get(pdf_url)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(resp.content)
                tmp_path = f.name
            logger.info(f"Downloaded PDF in {time.time() - download_time:.2f} seconds")

            logger.info(f"Loading PDF from temporary file: {tmp_path}")
            tmp_load_time = time.time()

            self.document_loader = PyPDFLoader(tmp_path)
            documents = self.document_loader.load()

            logger.info(
                f"Loaded {len(documents)} pages from temporary file in {time.time() - tmp_load_time:.2f} seconds"
            )

            if not documents:
                raise DocumentProcessingError("No documents loaded from PDF (fallback)")

            return documents
        except Exception as e:
            raise DocumentProcessingError(f"Error processing PDF: {str(e)}")

    def extract_documents_from_pdf_fast(self, pdf_url: str) -> List[Document]:
        """Extract documents from PDF URL using PyMuPDF (faster)"""

        start_time = time.time()

        try:
            # Download PDF content
            response = requests.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()

            # Load PDF directly from bytes (no temp file needed)
            pdf_content = response.content
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")

            documents = []
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text = page.get_text()

                if text.strip():  # Only add pages with content
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": pdf_url,
                            "page": page_num + 1,
                            "total_pages": pdf_document.page_count,
                        },
                    )
                    documents.append(doc)

            pdf_document.close()

            logger.info(
                f"Extracted {len(documents)} pages in {time.time() - start_time:.2f} seconds"
            )
            return documents

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {str(e)}")
            raise DocumentProcessingError(f"Error processing PDF: {str(e)}")

    async def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Async vector store creation"""

        start_time = time.time()

        # Extract texts and metadata from documents
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        logger.info(
            f"Extracted {len(texts)} texts and {len(metadatas)} metadata entries from documents"
        )

        # Generate embeddings asynchronously
        embeddings = await self.embeddings.aembed_documents(
            texts=texts, task_type="QUESTION_ANSWERING"
        )

        logger.info(
            f"Generated embeddings with length of single embedding: {len(embeddings[0])}"
        )

        # Create FAISS vector store
        vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=self.embeddings,
            metadatas=metadatas,
        )

        total_time = time.time() - start_time
        logger.info(f"Created vector store in {total_time:.2f} seconds")

        return vectorstore

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
            documents = self.extract_documents_from_pdf_fast(doc_url)

            # Split text into chunks
            logger.info("Creating text chunks")
            start_time = time.time()
            text_chunks = self.text_splitter.split_documents(documents)
            logger.info(
                f"Created text chunks in {time.time() - start_time:.2f} seconds"
            )

            if not text_chunks:
                raise DocumentProcessingError("No text chunks created from document")

            # Create vector store
            logger.info("Creating vector store")
            vectorstore = await self.create_vector_store(text_chunks)

            # Cache the vector store if enabled
            if settings.ENABLE_CACHE:
                logger.info("Saving vector store to cache")
                start_time = time.time()

                vectorstore_name = f"vectorstore_{uuid4()}"
                vectorstore.save_local(vectorstore_name)
                self.document_cache[doc_url] = vectorstore_name

                logger.info(
                    f"Saved vector store as {vectorstore_name} in {time.time() - start_time:.2f} seconds"
                )

            logger.info("Document processing completed successfully")
            return vectorstore

        except DocumentProcessingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing document: {e}")
            raise DocumentProcessingError(f"Unexpected error: {str(e)}")
