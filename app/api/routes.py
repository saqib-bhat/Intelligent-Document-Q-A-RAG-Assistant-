"""FastAPI route definitions."""

import logging
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.database.connection import get_db
from app.models.document import Document
from app.schemas.document import DocumentResponse, UploadResponse
from app.schemas.query import QueryRequest, QueryResponse, CitationSchema
from app.services.generation.llm_client import LLMClient

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

# ---------------------------------------------------------------------------
# Lazy dependency injection for the singleton pipeline/services
# ---------------------------------------------------------------------------
_pipeline = None
_generator = None


def get_pipeline():
    """Return the singleton IngestionPipeline (lazy init)."""
    global _pipeline
    if _pipeline is None:
        from app.services.ingestion.pipeline import IngestionPipeline
        _pipeline = IngestionPipeline()
    return _pipeline


def get_generator():
    """Return the singleton GeneratorService (lazy init)."""
    global _generator
    if _generator is None:
        from app.services.generation.generator import GeneratorService
        from app.services.retrieval.retriever import RetrieverService

        pipeline = get_pipeline()
        retriever = RetrieverService(
            embedder=pipeline.embedder,
            vector_store=pipeline.vector_store,
        )
        llm = LLMClient()
        _generator = GeneratorService(retriever=retriever, llm_client=llm)
    return _generator


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@router.get("/health", tags=["System"])
def health_check():
    """Liveness probe — returns 200 if the API is running."""
    return {"status": "healthy", "version": settings.app_version}


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_FILE_SIZE_MB = 50


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    summary="Upload a document (PDF, DOCX, TXT)",
)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a document to the RAG system.

    - Validates file type and size.
    - Extracts text, creates chunks, generates embeddings, builds FAISS index.
    - Persists document metadata to PostgreSQL.
    """
    # Validate extension
    ext = Path(file.filename or "").suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Save to disk
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    dest = upload_dir / unique_name

    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {exc}",
        )

    # Validate size
    size_mb = dest.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit.",
        )

    # Run ingestion pipeline
    pipeline = get_pipeline()
    try:
        doc = pipeline.ingest(
            file_path=dest,
            original_filename=file.filename or unique_name,
            db=db,
        )
    except Exception as exc:
        logger.exception(f"Ingestion error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )

    return UploadResponse(
        message="Document uploaded and indexed successfully.",
        document=DocumentResponse.model_validate(doc),
    )


@router.get(
    "/documents",
    response_model=List[DocumentResponse],
    tags=["Documents"],
    summary="List all uploaded documents",
)
def list_documents(db: Session = Depends(get_db)):
    """Return metadata for all documents in the system."""
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return [DocumentResponse.model_validate(d) for d in docs]


# ---------------------------------------------------------------------------
# Delete document
# ---------------------------------------------------------------------------
@router.delete(
    "/documents/{document_id}",
    tags=["Documents"],
    summary="Delete a document and rebuild the FAISS index",
)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    """
    Delete a document by ID.
    - Removes the record from PostgreSQL.
    - Rebuilds the FAISS index without the deleted document's chunks.
    """
    import uuid as _uuid
    try:
        doc_uuid = _uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid document ID format: {document_id}",
        )
    doc = db.query(Document).filter(Document.id == doc_uuid).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found.",
        )

    pipeline = get_pipeline()
    filename = doc.original_filename

    # Remove from DB
    db.delete(doc)
    db.commit()

    # Rebuild FAISS index excluding deleted document's chunks
    remaining_chunks = [
        c for c in pipeline.vector_store._chunks
        if c.source != filename
    ]

    if remaining_chunks:
        import numpy as np
        pipeline.vector_store.create_index()
        embeddings = pipeline.embedder.embed_chunks(remaining_chunks)
        pipeline.vector_store.add_chunks(remaining_chunks, embeddings)
        pipeline.vector_store.save()
    else:
        pipeline.vector_store.reset()

    # Reset generator so it picks up the updated vector store
    global _generator
    _generator = None

    logger.info(f"Document '{filename}' deleted and index rebuilt.")
    return {"message": f"Document '{filename}' deleted successfully."}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["Query"],
    summary="Ask a question about the uploaded documents",
)
def query_documents(
    request: QueryRequest,
    db: Session = Depends(get_db),
):
    """
    Answer a natural-language question using RAG.

    - Embeds the question and retrieves the most relevant chunks from FAISS.
    - Constructs a grounded prompt and calls the NVIDIA NIM LLM.
    - Returns the answer with source citations (document name + page number).
    """
    generator = get_generator()
    try:
        result = generator.answer(
            question=request.question,
            session_id=request.session_id,
            db=db,
            top_k=request.top_k,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception(f"Query error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {exc}",
        )

    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationSchema(
                document_name=c.document_name,
                page_number=c.page_number,
                relevance_score=c.relevance_score,
            )
            for c in result.citations
        ],
        retrieval_latency_seconds=result.retrieval_latency_seconds,
        response_latency_seconds=result.response_latency_seconds,
        model_used=result.model_used,
        session_id=request.session_id,
    )
