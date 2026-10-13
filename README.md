# book-rag

A small RAG (retrieval-augmented generation) app over a corpus of books pulled from
the Google Books API. Ask a natural-language question, get back an answer grounded
in a handful of retrieved books, with citations.

## Architecture

```
Google Books API --> ingest/fetch_books.py --> data/raw/*.json
                                                      |
                                                      v
                                          ingest/chunk.py --> data/processed/documents.jsonl
                                                      |
                                                      v
                                          ingest/embed.py --> pgvector (Postgres)
                                                      |
                                                      v
User question --> api/main.py --> retrieval/search.py --> top-k books
                        |                                       |
                        +--------------- Claude API <-----------+
                        |          (grounded answer + citations)
                        v
                  {answer, citations}
```

- **Ingest** (`ingest/`): fetch raw book data, flatten each book into one
  embeddable document, embed and upsert into Postgres/pgvector. Structured as three
  separate scripts (fetch -> chunk -> embed) so each step is independently rerunnable
  and so a future `ingest/fetch_podcasts.py` can follow the same fetch -> chunk ->
  embed pattern without touching the rest of the pipeline.
- **Retrieval** (`retrieval/search.py`): embeds the query, does cosine similarity
  search in pgvector, re-ranks the candidate pool with a light rating-aware blend.
- **API** (`api/main.py`): one FastAPI endpoint, `POST /ask`, that retrieves context
  and asks Claude to answer using only that context, citing which books it used.
- **Evals** (`evals/eval_set.json`): a stub list of test queries to fill in once
  there's a real corpus to evaluate against.

## Setup

Requires Python 3.10+ and Docker.

```bash
cp .env.example .env          # fill in ANTHROPIC_API_KEY
docker-compose up -d          # postgres + pgvector on localhost:5432
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m scripts.run_ingest  # fetch ~50 books, chunk, embed - first run downloads
                               # the MiniLM model (~90MB) and takes a minute or two

uvicorn api.main:app --reload
```

Then:

```bash
curl -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"query": "a science fiction book about time travel"}'
```

### Re-running individual ingest steps

```bash
python -m ingest.fetch_books --subject "poetry" --max-results 20
python -m ingest.chunk
python -m ingest.embed
```

## Decisions

Architecture decisions and why, logged as they're made.

- **pgvector over a dedicated vector DB (Pinecone, Weaviate, etc.)** - the corpus is
  small (dozens to low thousands of books) and we already need a relational store for
  book metadata. One database, one docker-compose service, no extra infra to run
  locally. `ivfflat` cosine index is enough at this scale; would revisit (HNSW, or a
  dedicated vector store) if the corpus grew into the millions.
- **One document per book, no sub-chunking** - Google Books descriptions are a
  paragraph or two at most. Splitting them into smaller chunks would fragment context
  without meaningfully improving retrieval granularity, and would multiply row count
  for no benefit at this corpus size.
- **`all-MiniLM-L6-v2` (sentence-transformers) for embeddings** - free, runs locally,
  no API key or network dependency for the embedding step, 384 dimensions keeps the
  pgvector index small. `ingest/embed.py` defines an `Embedder` interface so swapping
  in a hosted embedding API later (OpenAI, Voyage) is a one-class change - see the
  commented `OpenAIEmbedder` stub in that file.
- **Re-ranking is a simple similarity + rating blend, not a cross-encoder** - a
  learned re-ranker is overkill for a few dozen candidates from a small corpus.
  Revisit if the corpus grows or if book ratings turn out to be a bad proxy for
  relevance.
- **Claude for answer generation, called directly (not via LangChain or similar)** -
  one prompt, one call, no multi-step agent behavior needed yet. The system prompt
  instructs Claude to answer only from the retrieved context and to say so plainly
  when nothing in the corpus is a good match, to avoid hallucinated recommendations.
- **Answer generation is pluggable (`llm.py`)** - `AnswerGenerator` is an interface
  like `Embedder` in `ingest/embed.py`; `ClaudeGenerator` (default) and
  `OllamaGenerator` (local, free, no API key) both implement it. Switch via
  `LLM_PROVIDER` in `.env`. Local models trade answer quality for zero cost/key -
  useful for development or offline use.
