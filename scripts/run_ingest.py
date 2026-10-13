"""End-to-end ingest of a starter set of ~50 books: fetch -> chunk -> embed.

Usage:
    python -m scripts.run_ingest
"""

from ingest.chunk import main as chunk_main
from ingest.embed import main as embed_main
from ingest.fetch_books import fetch_books, save_raw_books

# A handful of topics that together pull ~50 varied books from Google Books.
# Plain free-text queries, not subject=True: Google's `subject:` filter only matches
# exact BISAC taxonomy strings (e.g. "science fiction" happens to be one, but
# "fantasy"/"mystery"/"history"/"cooking" return zero results through that filter).
SUBJECTS = [
    ("science fiction", 15),
    ("fantasy", 10),
    ("mystery", 10),
    ("history", 8),
    ("cooking", 7),
]


def main():
    total = 0
    for subject, count in SUBJECTS:
        items = fetch_books(subject, max_results=count, subject=False)
        saved = save_raw_books(items)
        print(f"[{subject}] fetched {len(items)}, saved {len(saved)}")
        total += len(saved)

    print(f"Fetched {total} raw books total. Chunking...")
    chunk_main()

    print("Embedding and upserting into pgvector...")
    embed_main()

    print("Done. Try: curl -X POST localhost:8000/ask -H 'content-type: application/json' "
          "-d '{\"query\": \"a book about time travel\"}'")


if __name__ == "__main__":
    main()
