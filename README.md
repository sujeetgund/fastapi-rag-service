# FastAPI RAG Service

A high-performance document question-answering service built with FastAPI and LangChain, implementing Retrieval-Augmented Generation (RAG) for intelligent document analysis.

## 🚀 Live Demo

- **API Docs**: https://hackrx-rag-app.onrender.com/docs
- **GitHub Repository**: https://github.com/sujeetgund/fastapi-rag-service.git

## 📋 Overview

This service allows users to upload documents via URL and ask multiple questions about the content. Using advanced RAG techniques with LangChain, it provides accurate, context-aware answers by retrieving relevant document sections and generating responses.

## ✨ Features

- **Document Processing**: Support for various document formats (PDF, DOC, TXT, etc.)
- **Multi-Question Support**: Process multiple questions in a single request
- **RESTful API**: Clean, well-documented FastAPI endpoints
- **RAG Pipeline**: Advanced retrieval-augmented generation for accurate answers
- **Dockerized**: Easy deployment with Docker support
- **Type Safety**: Full Pydantic model validation
- **Error Handling**: Comprehensive exception handling and logging

## 🛠️ Tech Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **LangChain**: Framework for developing applications with language models
- **Pydantic**: Data validation and settings management
- **Docker**: Containerization for easy deployment
- **UV**: Fast Python package installer and resolver

## 📁 Project Structure

```
fastapi-rag-service/
├── api/
│   └── routes.py              # API endpoint definitions
├── core/
│   ├── config.py              # Configuration settings
│   ├── exceptions.py          # Custom exception classes
│   └── security.py            # Security and authentication
├── models/
│   └── schemas.py             # Pydantic models and schemas
├── services/
│   ├── document_service.py    # Document processing logic
│   └── qa_service.py          # Question-answering service
├── .dockerignore
├── .gitignore
├── .python-version
├── Dockerfile                 # Docker configuration
├── main.py                    # FastAPI application entry point
├── pyproject.toml             # Project metadata and dependencies
├── README.md
├── requirements.txt           # Python dependencies
└── uv.lock                    # UV lock file
```

## 📊 API Models

### QuestionAnswerRequest Model

The input structure for the primary `/hackrx/run` endpoint.

| Field     | Type       | Description                              | Example                                    |
|-----------|------------|------------------------------------------|--------------------------------------------|
| documents | str        | URL of the document to process           | "https://example.com/document.pdf"        |
| questions | List[str]  | List of questions to answer              | ["What is the main topic?", "What are the key findings?"] |

### QuestionAnswerResponse Model

The output structure from the RAG pipeline.

| Field   | Type       | Description                                      | Example                                    |
|---------|------------|--------------------------------------------------|--------------------------------------------|
| answers | List[str]  | Generated answers in the same order as questions| ["The main topic is...", "The key findings are..."] |

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional)
- UV package manager (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/sujeetgund/fastapi-rag-service.git
cd fastapi-rag-service
```

2. **Install dependencies**

Using UV (recommended):
```bash
uv sync
```

Using pip:
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Docker Deployment

1. **Build the Docker image**
```bash
docker build -t fastapi-rag-service .
```

2. **Run the container**
```bash
docker run -p 8000:8000 fastapi-rag-service
```

## 📖 Usage

### API Endpoint

**POST** `/hackrx/run`

Process a document and answer questions about its content.

#### Request Example

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/research-paper.pdf",
    "questions": [
      "What is the main research question?",
      "What methodology was used?",
      "What are the key conclusions?"
    ]
  }'
```

#### Response Example

```json
{
  "answers": [
    "The main research question focuses on investigating the impact of machine learning algorithms on data processing efficiency.",
    "The study employed a mixed-methods approach combining quantitative analysis with qualitative interviews.",
    "The key conclusions indicate that ML algorithms can improve processing efficiency by up to 40% while maintaining data accuracy."
  ]
}
```

### Interactive Documentation

Visit `http://localhost:8000/docs` to access the interactive Swagger UI documentation.

## ⚙️ Configuration

Configuration settings can be modified in `core/config.py`:

- API settings
- Model parameters
- Document processing options
- Security configurations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.