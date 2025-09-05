import time
from typing import List
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from core.config import settings
from core.exceptions import QAServiceError

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

from langsmith import traceable, Client

logger = logging.getLogger(__name__)


def format_docs(retrieved_docs):
    context_text = "\n---\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


class QAService:
    def __init__(self):
        self.llm = None
        self.qa_prompt = None

    async def initialize(self):
        """Initialize the QA service"""
        try:
            # Initialize LLM
            if not settings.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required")

            self.llm = ChatGoogleGenerativeAI(
                model=settings.GOOGLE_MODEL_NAME,
                google_api_key=settings.GOOGLE_API_KEY,
                verbose=True,
            )

            # Create QA prompt template
            self.qa_prompt = PromptTemplate(
                template="""
                You are a document reader. Your ONLY job is to find answers in the text below. You cannot use any other knowledge.

                CRITICAL CONSTRAINTS:
                - Answer ONLY from the context provided
                - If not found in context: "Information not found in documents"
                - No external knowledge allowed
                - No assumptions or inferences

                Context from documents:
                {context}

                User question: {question}

                Document-based answer:
                """,
                input_variables=["context", "question"],
            )

            logger.info("QA service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize QA service: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("QA service cleaned up")

    @traceable(run_type="chain", project_name=settings.LANGSMITH_PROJECT)
    async def answer_questions(
        self, vectorstore: FAISS, questions: List[str]
    ) -> List[str]:
        """Answer questions using the vector store"""
        try:
            start_time = time.time()

            # Create output parser
            parser = StrOutputParser()

            # Create retriever from vector store
            retriever = vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 5}
            )

            # Create retrieval QA chain
            parallel_chain = RunnableParallel(
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
            )
            qa_chain = parallel_chain | self.qa_prompt | self.llm | parser

            start_time = time.time()

            answers = await qa_chain.abatch(questions)
            logger.info(
                f"Time taken to answer questions: {time.time() - start_time:.2f} seconds"
            )

            return answers

        except Exception as e:
            logger.error(f"Error in answer_questions: {e}")
            raise QAServiceError(f"Failed to answer questions: {str(e)}")
