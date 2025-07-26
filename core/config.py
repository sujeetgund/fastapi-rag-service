from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Document Q&A API"

    # Authentication
    BEARER_TOKEN: str = (
        "53c5fa96c744c52cab517ea967decca429e501a34ae7e56f70da36a3fa71baf6"
    )

    # Google AI Configuration
    GOOGLE_API_KEY: str
    GOOGLE_MODEL_NAME: str
    GOOGLE_EMBEDDING_MODEL: str

    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 400
    MAX_RELEVANT_CHUNKS: int = 5

    # Cache Configuration
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 600  # 10 minutes

    class Config:
        env_file = ".env"


settings = Settings()
