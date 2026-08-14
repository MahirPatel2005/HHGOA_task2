FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY frontend ./frontend
COPY data ./data
COPY scripts ./scripts
COPY .env.example ./.env.example

RUN pip install --no-cache-dir \
    "qdrant-client>=1.12" \
    "sentence-transformers>=3.0" \
    "datasets>=2.20" \
    "groq>=0.11"

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
