"""Postgres/pgvector connection and schema management."""

import psycopg2
from pgvector.psycopg2 import register_vector

from config import DATABASE_URL, EMBEDDING_DIM, TABLE_NAME


def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    return conn


def init_db():
    """Create the books table if it doesn't exist yet."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT,
                text TEXT NOT NULL,
                metadata JSONB,
                embedding VECTOR({EMBEDDING_DIM})
            );
            """
        )
    conn.commit()
    conn.close()


def ensure_vector_index():
    """Create the ivfflat index if it's missing. Must run after rows exist -
    ivfflat trains its clusters on the data present at creation time, so calling
    this before any rows are loaded would build a useless (or failing) index.
    `lists = 10` is tuned for a few-hundred-row demo dataset, not production scale.
    """
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
        (row_count,) = cur.fetchone()
        if row_count == 0:
            conn.close()
            return
        cur.execute(
            f"""
            SELECT 1 FROM pg_indexes WHERE indexname = '{TABLE_NAME}_embedding_idx';
            """
        )
        if cur.fetchone() is None:
            cur.execute(
                f"""
                CREATE INDEX {TABLE_NAME}_embedding_idx
                ON {TABLE_NAME} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);
                """
            )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialized table '{TABLE_NAME}' in {DATABASE_URL}")
