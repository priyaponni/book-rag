"""Transform raw Google Books JSON (data/raw/*.json) into flat, embeddable documents.

Each book becomes exactly one document: {id, title, author, text, metadata}.
No sub-chunking - book descriptions are short enough (a paragraph or two) that
splitting them would lose more context than it gains. See README Decisions.

Usage:
    python -m ingest.chunk
"""

import glob
import json
import os

from config import DOCUMENTS_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR


def build_document(raw: dict) -> dict | None:
    volume_info = raw.get("volumeInfo", {})
    title = volume_info.get("title")
    if not title:
        return None

    authors = volume_info.get("authors", [])
    author = ", ".join(authors) if authors else "Unknown"

    description = volume_info.get("description", "")
    categories = volume_info.get("categories", [])
    # searchInfo.textSnippet is Google Books' matched-text snippet for the query -
    # the closest thing the free API gives us to a "review snippet".
    snippet = raw.get("searchInfo", {}).get("textSnippet", "")

    text_parts = [
        description,
        f"Categories: {', '.join(categories)}" if categories else "",
        snippet,
    ]
    text = "\n\n".join(part for part in text_parts if part).strip()
    if not text:
        return None

    return {
        "id": raw["id"],
        "title": title,
        "author": author,
        "text": text,
        "metadata": {
            "published_date": volume_info.get("publishedDate"),
            "categories": categories,
            "page_count": volume_info.get("pageCount"),
            "average_rating": volume_info.get("averageRating"),
            "ratings_count": volume_info.get("ratingsCount"),
            "language": volume_info.get("language"),
            "thumbnail": volume_info.get("imageLinks", {}).get("thumbnail"),
            "preview_link": volume_info.get("previewLink"),
            "info_link": volume_info.get("infoLink"),
        },
    }


def chunk_all() -> list[dict]:
    documents = []
    for path in sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.json"))):
        with open(path) as f:
            raw = json.load(f)
        doc = build_document(raw)
        if doc:
            documents.append(doc)
    return documents


def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    documents = chunk_all()
    with open(DOCUMENTS_PATH, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")
    print(f"Wrote {len(documents)} documents to {DOCUMENTS_PATH}")


if __name__ == "__main__":
    main()
