"""Embed documents (data/processed/documents.jsonl) and upsert into pgvector.

Usage:
    python -m ingest.embed
"""

import json
from abc import ABC, abstractmethod

import psycopg2.extras

from config import DOCUMENTS_PATH, EMBEDDING_DIM, EMBEDDING_MODEL, TABLE_NAME
from db import ensure_vector_index, get_connection, init_db


class Embedder(ABC):
    """Swap in a different embedding provider (OpenAI, Voyage, ...) by implementing this."""

    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbedder(Embedder):
    """Free, local embeddings via sentence-transformers. Default for this project."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()


# To swap in a hosted API instead of a local model, implement the same interface, e.g.:
#
# class OpenAIEmbedder(Embedder):
#     def __init__(self, model_name: str = "text-embedding-3-small"):
#         from openai import OpenAI
#         self.client = OpenAI()
#         self.model_name = model_name
#         self.dim = 1536
#
#     def embed(self, texts: list[str]) -> list[list[float]]:
#         response = self.client.embeddings.create(model=self.model_name, input=texts)
#         return [d.embedding for d in response.data]
#
# Then update EMBEDDING_DIM in .env to match, and pass OpenAIEmbedder() to embed_and_upsert().


def load_documents() -> list[dict]:
    documents = []
    with open(DOCUMENTS_PATH) as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))
    return documents


def embed_and_upsert(documents: list[dict] | None = None, embedder: Embedder | None = None):
    documents = documents if documents is not None else load_documents()
    embedder = embedder or SentenceTransformerEmbedder()

    if embedder.dim != EMBEDDING_DIM:
        raise ValueError(
            f"Embedder dim {embedder.dim} != EMBEDDING_DIM {EMBEDDING_DIM} in .env. "
            "Update EMBEDDING_DIM (and recreate the table) to match."
        )

    init_db()

    texts = [doc["text"] for doc in documents]
    vectors = embedder.embed(texts)

    conn = get_connection()
    with conn.cursor() as cur:
        rows = [
            (
                doc["id"],
                doc["title"],
                doc["author"],
                doc["text"],
                json.dumps(doc["metadata"]),
                vector,
            )
            for doc, vector in zip(documents, vectors)
        ]
        psycopg2.extras.execute_values(
            cur,
            f"""
            INSERT INTO {TABLE_NAME} (id, title, author, text, metadata, embedding)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                author = EXCLUDED.author,
                text = EXCLUDED.text,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding;
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s)",
        )
    conn.commit()
    conn.close()
    ensure_vector_index()
    print(f"Upserted {len(rows)} books into '{TABLE_NAME}'")


def main():
    embed_and_upsert()


if __name__ == "__main__":
    main()
