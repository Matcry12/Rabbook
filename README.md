# Rabbook

A simple FastAPI RAG app for chatting with your own PDF and TXT documents.

## Stack

- FastAPI for the web UI
- Chroma for vector storage
- Hugging Face embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Google Generative AI via LangChain for answer generation

## Project Structure

```text
rabbook/
├── main.py
├── config.py
├── ingest.py
├── retrieve.py
├── prompt.py
├── templates/
├── static/
├── tests/
├── data/
├── .env.example
├── .gitignore
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your Google API key to `.env`.

## Run

```bash
python main.py
```

Open `http://127.0.0.1:6001`.

## Notes

- Browsers commonly block port `6000`, so the app defaults to `6001`.
- Documents in `data/` and generated Chroma files are ignored by Git.
- Uploaded files go into `data/uploads/`.
