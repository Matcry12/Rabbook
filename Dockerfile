FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Chromium for crawl4ai web scraping
RUN playwright install chromium --with-deps

# Pre-download embedding and reranking models so first query is instant
RUN python -c "\
from langchain_huggingface import HuggingFaceEmbeddings; \
from sentence_transformers import CrossEncoder; \
HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .

RUN mkdir -p data/uploads/urls data/chroma_db

EXPOSE 6001

CMD ["python", "main.py"]
