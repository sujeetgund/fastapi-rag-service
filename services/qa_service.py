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
                model=settings.GOOGLE_MODEL_NAME, google_api_key=settings.GOOGLE_API_KEY
            )

            # Create QA prompt template
            self.qa_prompt = PromptTemplate(
                template="""
                You are an expert AI assistant for answering questions about company policies. Your task is to answer the user's question using ONLY the provided policy document excerpts.

                Follow these rules STRICTLY:
                1. Base your answer entirely on the text provided in the "Context" section.
                2. Do not make assumptions or infer information that is not explicitly stated in the context.
                3. If the context does not contain the answer to the question, you MUST state: "The provided policy documents do not contain information on this topic."
                4. Be concise and direct in your answer.
                5. Don't include any additional information or explanations beyond the answer to the question.

                Context:
                ---
                {context}
                ---

                Question: {question}

                Answer:
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

    async def answer_questions(
        self, vectorstore: FAISS, questions: List[str]
    ) -> List[str]:
        """Answer questions using the vector store"""
        try:
            # qa_chain = RetrievalQA.from_chain_type(
            #     llm=self.llm,
            #     chain_type="stuff",
            #     retriever=vectorstore.as_retriever(
            #         search_kwargs={"k": settings.MAX_RELEVANT_CHUNKS}
            #     ),
            #     chain_type_kwargs={"prompt": self.qa_prompt},
            #     return_source_documents=False,
            # )

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

            print(
                "=" * 60,
                f"\nTime taken to create QA chain: {time.time() - start_time:.2f} seconds\n",
                "=" * 60,
            )

            answers = []
            for question in questions:
                logger.info(f"Processing question: {question[:50]}...")

                try:
                    # Get answer from QA chain
                    start_time = time.time()
                    result = await qa_chain.ainvoke(question)
                    answer = result

                    # Clean up the answer
                    if not answer:
                        answer = "The information requested is not available in the provided document."

                    answers.append(answer)

                except Exception as e:
                    logger.error(f"Error answering question '{question}': {e}")
                    answers.append(
                        "Sorry, I encountered an error while processing this question."
                    )
                finally:
                    print(
                        "=" * 60,
                        f"\nTime taken to answer question '{question[:50]}...': {time.time() - start_time:.2f} seconds\n",
                        "=" * 60,
                    )

            return answers

        except Exception as e:
            logger.error(f"Error in answer_questions: {e}")
            raise QAServiceError(f"Failed to answer questions: {str(e)}")
