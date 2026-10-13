"""Pull book records from the Google Books API and save raw JSON to data/raw/.

Usage:
    python -m ingest.fetch_books --query "science fiction" --max-results 40
    python -m ingest.fetch_books --subject "cooking" --max-results 20

The Google Books "volumes" search endpoint works without an API key for basic
search traffic; set GOOGLE_BOOKS_API_KEY in .env if you start hitting rate limits.
"""

import argparse
import json
import os
import time

import requests

from config import GOOGLE_BOOKS_API_KEY, RAW_DATA_DIR

GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
PAGE_SIZE = 40  # Google Books API max per request


MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2  # 2, 4, 8, 16, 32s on successive 429s


def _get_with_retry(params: dict) -> requests.Response:
    """GET with exponential backoff on 429 - the free, unauthenticated Google Books
    endpoint rate-limits fairly aggressively and a single retry is often not enough."""
    for attempt in range(MAX_RETRIES):
        response = requests.get(GOOGLE_BOOKS_URL, params=params, timeout=30)
        if response.status_code != 429:
            response.raise_for_status()
            return response
        if attempt == MAX_RETRIES - 1:
            response.raise_for_status()  # exhausted retries, surface the 429
        wait = BACKOFF_BASE_SECONDS * (2 ** attempt)
        print(f"  429 from Google Books API, retrying in {wait}s "
              f"(attempt {attempt + 1}/{MAX_RETRIES})...")
        time.sleep(wait)
    raise RuntimeError("unreachable")  # loop always returns or raises


def fetch_books(query: str, max_results: int = 40, subject: bool = False) -> list[dict]:
    """Fetch up to max_results book volumes matching a query or subject."""
    search_term = f"subject:{query}" if subject else query
    items: list[dict] = []
    start_index = 0

    while len(items) < max_results:
        page_size = min(PAGE_SIZE, max_results - len(items))
        params = {"q": search_term, "startIndex": start_index, "maxResults": page_size}
        if GOOGLE_BOOKS_API_KEY:
            params["key"] = GOOGLE_BOOKS_API_KEY

        response = _get_with_retry(params)
        payload = response.json()

        page_items = payload.get("items", [])
        if not page_items:
            break

        items.extend(page_items)
        start_index += len(page_items)
        time.sleep(0.5)  # be polite to the free, unauthenticated endpoint

    return items[:max_results]


def save_raw_books(items: list[dict]) -> list[str]:
    """Write one JSON file per book to data/raw/, keyed by Google Books volume id."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    saved_paths = []
    for item in items:
        volume_id = item.get("id")
        if not volume_id:
            continue
        path = os.path.join(RAW_DATA_DIR, f"{volume_id}.json")
        with open(path, "w") as f:
            json.dump(item, f, indent=2)
        saved_paths.append(path)
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Fetch books from the Google Books API")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help="Free-text search query, e.g. 'science fiction'")
    group.add_argument("--subject", help="Subject/category search, e.g. 'cooking'")
    parser.add_argument("--max-results", type=int, default=40)
    args = parser.parse_args()

    if args.subject:
        items = fetch_books(args.subject, max_results=args.max_results, subject=True)
    else:
        items = fetch_books(args.query, max_results=args.max_results, subject=False)

    saved = save_raw_books(items)
    print(f"Fetched {len(items)} books, saved {len(saved)} to {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
