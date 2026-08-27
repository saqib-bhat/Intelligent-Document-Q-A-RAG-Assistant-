"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config.settings import get_settings
from app.database.connection import create_tables
from app.utils.logger import configure_logging

settings = get_settings()
configure_logging(debug=settings.debug)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup and shutdown logic."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    create_tables()
    logger.info("Database tables verified.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-ready Retrieval-Augmented Generation (RAG) API.\n\n"
        "Upload documents (PDF, DOCX, TXT) and ask natural-language questions. "
        "Answers are grounded in the uploaded content and include source citations."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
