# Stage 1: install dependencies
FROM python:3.12-slim AS builder
WORKDIR /app

# Upgrade pip and install your dependencies
RUN python -m pip install --upgrade pip

# Copy only requirements to leverage Docker cache
COPY requirements.txt . 

# Install dependencies to a specific directory
RUN pip install --prefix=/install -r requirements.txt

# Stage 2: build the final image
FROM python:3.12-slim
WORKDIR /app

# Copy installed libs from the builder
COPY --from=builder /install /usr/local

# Copy your app code
COPY . .

# Ensure FastAPI logs appear in real‑time
ENV PYTHONUNBUFFERED=1

# Cloud Run expects PORT environment variable; default to 8080
ENV PORT=8080

EXPOSE 8080

# Launch via Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
