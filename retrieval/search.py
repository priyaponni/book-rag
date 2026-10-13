"""Vector similarity search over the books table, with a light re-ranking pass.

Usage:
    python -m retrieval.search "space opera with political intrigue"
"""

import sys

from config import TABLE_NAME
from db import get_connection
from ingest.embed import SentenceTransformerEmbedder

_embedder = None


def get_embedder() -> SentenceTransformerEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder()
    return _embedder


def search(query: str, top_k: int = 5, candidate_pool: int = 20) -> list[dict]:
    """Embed the query, run cosine similarity search, then re-rank by a blend of
    vector similarity and average rating so well-regarded books edge out otherwise-
    similar ones. Returns top_k results with both scores."""
    embedder = get_embedder()
    query_vector = embedder.embed([query])[0]

    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, title, author, text, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM {TABLE_NAME}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_vector, query_vector, candidate_pool),
        )
        rows = cur.fetchall()
    conn.close()

    results = []
    for id_, title, author, text, metadata, similarity in rows:
        rating = (metadata or {}).get("average_rating") or 0
        # Blend: similarity dominates, rating (0-5 scale, normalized) nudges the order.
        rerank_score = 0.85 * similarity + 0.15 * (rating / 5)
        results.append(
            {
                "id": id_,
                "title": title,
                "author": author,
                "text": text,
                "metadata": metadata,
                "similarity": float(similarity),
                "rerank_score": float(rerank_score),
            }
        )

    results.sort(key=lambda r: r["rerank_score"], reverse=True)
    return results[:top_k]


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m retrieval.search "your query"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    for rank, result in enumerate(search(query), start=1):
        print(f"{rank}. {result['title']} by {result['author']} "
              f"(similarity={result['similarity']:.3f})")


if __name__ == "__main__":
    main()
