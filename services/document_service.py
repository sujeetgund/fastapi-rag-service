from typing import List, Dict, Any
import logging
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
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

from langchain_community.document_loaders import PyMuPDFLoader


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

    def _format_table_content(
        self, table_lines: List[str], table_format: str = "markdown"
    ) -> str:
        """
        Format table content for better RAG retrieval.

        Args:
            table_lines (List[str]): Lines that appear to be table content
            table_format (str): Desired format for the table

        Returns:
            str: Formatted table content
        """

        if table_format == "markdown":
            # Convert to markdown table format
            if len(table_lines) < 2:
                return "\n".join(table_lines)

            # Try to create a proper markdown table
            formatted_lines = []
            for i, line in enumerate(table_lines):
                # Replace multiple spaces with | for markdown
                cells = [cell.strip() for cell in line.split() if cell.strip()]
                if cells:
                    formatted_line = "| " + " | ".join(cells) + " |"
                    formatted_lines.append(formatted_line)

                    # Add header separator after first row
                    if i == 0:
                        separator = "|" + "|".join([" --- " for _ in cells]) + "|"
                        formatted_lines.append(separator)

            return "\n".join(formatted_lines)

        elif table_format == "structured":
            # Create structured representation
            structured = ["STRUCTURED TABLE DATA:"]
            for i, line in enumerate(table_lines):
                structured.append(f"Row {i+1}: {line}")
            return "\n".join(structured)

        else:  # csv or default
            return "\n".join(table_lines)

    def _process_mixed_content(
        self, content: str, page_num: int, table_format: str = "markdown"
    ) -> str:
        """
        Process content that may contain both text and tables for optimal RAG performance.

        Args:
            content (str): Raw content from the page
            page_num (int): Page number for context
            table_format (str): Format for table representation

        Returns:
            str: Processed content optimized for RAG
        """

        lines = content.split("\n")
        processed_lines = []
        current_section = []
        in_table = False

        for line in lines:
            line = line.strip()
            if not line:
                if current_section:
                    processed_lines.append("\n".join(current_section))
                    current_section = []
                continue

            # Heuristic to detect table-like content
            # Tables often have multiple columns separated by spaces or tabs
            tab_count = line.count("\t")
            space_groups = len([x for x in line.split("  ") if x.strip()])

            is_table_row = (
                tab_count >= 2
                or space_groups >= 3
                or ("|" in line and line.count("|") >= 2)
            )

            if is_table_row and not in_table:
                # Starting a table section
                if current_section:
                    processed_lines.append("\n".join(current_section))
                    current_section = []
                processed_lines.append(f"\n[TABLE CONTENT - Page {page_num}]")
                in_table = True
                current_section = [line]
            elif is_table_row and in_table:
                # Continue table
                current_section.append(line)
            elif not is_table_row and in_table:
                # End of table
                if current_section:
                    table_content = self._format_table_content(
                        current_section, table_format
                    )
                    processed_lines.append(table_content)
                    processed_lines.append("[END TABLE]\n")
                    current_section = []
                in_table = False
                current_section = [line]
            else:
                # Regular text
                current_section.append(line)

        # Handle any remaining content
        if current_section:
            if in_table:
                table_content = self._format_table_content(
                    current_section, table_format
                )
                processed_lines.append(table_content)
                processed_lines.append("[END TABLE]")
            else:
                processed_lines.append("\n".join(current_section))

        return "\n".join(processed_lines)

    def _create_overlap_content(
        prev_content: str, current_content: str, page_num: int
    ) -> str:
        """
        Create overlap content between adjacent pages for better context in RAG.

        Args:
            prev_content (str): Content from previous page
            current_content (str): Content from current page
            page_num (int): Current page number

        Returns:
            str: Overlap content or empty string if not beneficial
        """

        # Take last few sentences from previous page and first few from current
        prev_sentences = prev_content.split(".")[-3:]  # Last 3 sentences
        current_sentences = current_content.split(".")[:3]  # First 3 sentences

        # Clean and filter
        prev_clean = [
            s.strip() for s in prev_sentences if s.strip() and len(s.strip()) > 10
        ]
        current_clean = [
            s.strip() for s in current_sentences if s.strip() and len(s.strip()) > 10
        ]

        if prev_clean and current_clean:
            overlap_content = (
                f"[CONTEXT OVERLAP - Pages {page_num-1}-{page_num}]\n"
                f"...{'. '.join(prev_clean)}.\n"
                f"{'. '.join(current_clean)}..."
            )
            return overlap_content

        return ""

    def load_pdf_for_rag(
        self,
        file_path: str,
        table_extraction_kwargs: Dict[str, Any] = None,
        chunk_overlap_pages: bool = False,
        include_page_numbers: bool = True,
        table_format: str = "markdown",
    ) -> List[Document]:

        start_time = time.time()
        
        try:
            # Initialize the PDFPlumberLoader with table extraction enabled
            loader_kwargs = {
                "extract_images": False,  # Focus on text and tables for RAG
            }

            # Add table extraction parameters if provided
            if table_extraction_kwargs:
                loader_kwargs.update(table_extraction_kwargs)

            # Load the PDF
            logger.info(f"Loading PDF from: {file_path}")
            loader = PDFPlumberLoader(file_path, **loader_kwargs)

            # Extract documents
            raw_documents = loader.load()
            logger.info(f"Extracted {len(raw_documents)} pages from PDF")
            logger.info(
                f"PDF loading completed in {time.time() - start_time:.2f} seconds"
            )

            # Process documents for RAG optimization
            processed_documents = []
            
            start_time = time.time()
            
            for i, doc in enumerate(raw_documents):
                page_num = i + 1

                # Extract and process the content
                content = doc.page_content.strip()

                if not content:
                    logger.warning(f"Page {page_num} appears to be empty, skipping...")
                    continue

                # Enhanced metadata for better retrieval
                metadata = {
                    "source": file_path,
                    "page": page_num,
                    "total_pages": len(raw_documents),
                    "content_type": "mixed",  # Indicates both text and potential tables
                }

                # Add original metadata if it exists
                if hasattr(doc, "metadata") and doc.metadata:
                    metadata.update(doc.metadata)

                # Process content to separate text from tables
                processed_content = self._process_mixed_content(
                    content, page_num, table_format=table_format
                )

                # Create enhanced document
                enhanced_doc = Document(
                    page_content=processed_content, metadata=metadata
                )

                processed_documents.append(enhanced_doc)

                # Add context overlap from adjacent pages if requested
                if chunk_overlap_pages and i > 0:
                    overlap_content = self._create_overlap_content(
                        raw_documents[i - 1].page_content, content, page_num
                    )
                    if overlap_content:
                        overlap_doc = Document(
                            page_content=overlap_content,
                            metadata={
                                **metadata,
                                "content_type": "overlap",
                                "overlap_pages": f"{page_num-1}-{page_num}",
                            },
                        )
                        processed_documents.append(overlap_doc)

            logger.info(
                f"Created {len(processed_documents)} processed documents in {time.time() - start_time:.2f} seconds"
            )
            return processed_documents

        except FileNotFoundError:
            logger.error(f"PDF file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}", exc_info=True)
            raise

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

            pdf_doc = PyMuPDFLoader(extract_tables="markdown")

            documents = []
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text = page.get_text()
                tables = page.find_tables()

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
            
            if(settings.ENV == "local"):
                # Download the PDF to a temporary file
                resp = requests.get(doc_url)
                resp.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    f.write(resp.content)
                    tmp_path = f.name
                    documents = self.load_pdf_for_rag(tmp_path)
            else:
                # Use the fast extraction method for production
                documents = self.load_pdf_for_rag(doc_url)

            # Split text into chunks
            # logger.info("Creating text chunks")
            # start_time = time.time()
            # text_chunks = self.text_splitter.split_documents(documents)
            # logger.info(
            #     f"Created text chunks in {time.time() - start_time:.2f} seconds"
            # )

            # if not text_chunks:
            #     raise DocumentProcessingError("No text chunks created from document")

            # Create vector store
            logger.info("Creating vector store")
            vectorstore = await self.create_vector_store(documents)

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
