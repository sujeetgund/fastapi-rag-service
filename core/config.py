import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Document Q&A API"
    ENV: str = "local"

    # Authentication
    BEARER_TOKEN: str = (
        "53c5fa96c744c52cab517ea967decca429e501a34ae7e56f70da36a3fa71baf6"
    )

    # Google AI Configuration
    GOOGLE_API_KEY: str
    GOOGLE_MODEL_NAME: str
    GOOGLE_EMBEDDING_MODEL: str
    
    # Langsmith Configuration
    LANGSMITH_API_KEY: str
    LANGSMITH_TRACING: str
    LANGSMITH_ENDPOINT: str
    LANGSMITH_PROJECT: str

    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 400
    MAX_RELEVANT_CHUNKS: int = 5

    # Cache Configuration
    ENABLE_CACHE: bool = False
    CACHE_TTL: int = 600  # 10 minutes

    class Config:
        env_file = ".env"


settings = Settings()
