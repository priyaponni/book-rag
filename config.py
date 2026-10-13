"""Shared settings for ingest, retrieval, and the API. Loaded from environment / .env."""

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://bookrag:bookrag@localhost:5432/bookrag")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")

# Answer-generation provider for api/main.py - see llm.py for the AnswerGenerator
# interface. "anthropic" (default, hosted Claude) or "ollama" (local, no API key).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

TABLE_NAME = "books"

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
DOCUMENTS_PATH = os.path.join(PROCESSED_DATA_DIR, "documents.jsonl")
