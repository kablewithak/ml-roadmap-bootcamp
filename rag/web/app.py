"""FastAPI web application for RAG Assistant.

Provides a simple web interface using HTMX for interactions.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag.config import Config
from rag.ingest import DocumentIngester
from rag.pipeline import PipelineError, RAGPipeline

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Assistant",
    description="Question answering over local knowledge bases",
    version="0.1.0",
)

# Set up templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Global pipeline instance
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Get or create the pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page."""
    # Check health status
    pipeline = get_pipeline()
    health = pipeline.check_health()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "health": health,
            "config": {
                "model": Config.OLLAMA_MODEL,
                "top_k": Config.RETRIEVER_TOP_K,
            },
        },
    )


@app.post("/ask", response_class=HTMLResponse)
async def ask(
    request: Request,
    question: str = Form(...),
    top_k: Optional[int] = Form(None),
):
    """Process a question and return the answer."""
    if not question.strip():
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "error": "Please enter a question."},
        )

    pipeline = get_pipeline()

    try:
        answer = pipeline.ask(
            question=question,
            top_k=top_k,
        )
        sources = pipeline.get_sources()

        return templates.TemplateResponse(
            "partials/answer.html",
            {
                "request": request,
                "question": question,
                "answer": answer,
                "sources": sources,
            },
        )

    except PipelineError as e:
        logger.error(f"Pipeline error: {e}")
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "error": str(e)},
        )


@app.post("/ingest", response_class=HTMLResponse)
async def ingest(request: Request):
    """Trigger knowledge base ingestion."""
    try:
        ingester = DocumentIngester()
        num_chunks = ingester.ingest()

        return templates.TemplateResponse(
            "partials/ingest_result.html",
            {
                "request": request,
                "num_chunks": num_chunks,
                "success": True,
            },
        )

    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "error": f"Ingestion failed: {e}"},
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    pipeline = get_pipeline()
    return pipeline.check_health()


@app.get("/sources", response_class=HTMLResponse)
async def sources(request: Request):
    """Get sources from last query."""
    pipeline = get_pipeline()
    sources = pipeline.get_sources()

    return templates.TemplateResponse(
        "partials/sources.html",
        {"request": request, "sources": sources},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "rag.web.app:app",
        host=Config.WEB_HOST,
        port=Config.WEB_PORT,
        reload=True,
    )
