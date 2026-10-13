"""FastAPI app: POST /ask {query} -> retrieves relevant books, asks Claude to
answer using only that context, and returns the answer plus which books it cited.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm import LLMError, get_generator
from retrieval.search import search

import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="book-rag")

SYSTEM_PROMPT = """You are a book recommendation assistant. You are given a user \
question and a list of candidate books (title, author, description). Answer the \
question using ONLY the information in the candidate books - do not use outside \
knowledge about books not listed.

Cite every book you rely on by title in your answer text (e.g. "In Dune, ..."). \
If none of the candidates actually answer the question, say so plainly instead of \
guessing. Keep the answer concise - a short paragraph is enough."""


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


class Citation(BaseModel):
    id: str
    title: str
    author: str
    similarity: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


def build_context(results: list[dict]) -> str:
    blocks = []
    for r in results:
        blocks.append(
            f"Title: {r['title']}\nAuthor: {r['author']}\nDescription: {r['text']}"
        )
    return "\n\n---\n\n".join(blocks)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    results = search(request.query, top_k=request.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No books found in the index yet")

    context = build_context(results)
    user_content = f"Candidate books:\n\n{context}\n\nQuestion: {request.query}"

    try:
        generator = get_generator()
        answer = generator.generate(SYSTEM_PROMPT, user_content)
    except LLMError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))

    return AskResponse(
        answer=answer,
        citations=[
            Citation(id=r["id"], title=r["title"], author=r["author"], similarity=r["similarity"])
            for r in results
        ],
    )
