"""
FastAPI application entry-point for the LLM Hallucination Tester service.

Endpoints
---------
GET  /health                — Liveness probe (used by Cloud Run).
POST /check-hallucination   — Run the full hallucination-checking pipeline.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.hallucination_checker import HallucinationChecker
from app.models import (
    HealthResponse,
    HallucinationRequest,
    HallucinationResponse,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
MAX_QUESTIONS: int = int(os.getenv("MAX_QUESTIONS", "10"))

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.

    Validates the presence of the OpenAI API key at startup to fail fast
    instead of surfacing the error on the first request.
    """
    # TODO: maybe check if the api key is valid by making a dummy call? But this is fine for now
    if not OPENAI_API_KEY:
        logger.warning(
            "OPENAI_API_KEY is not set in env. Calls will fail unless you pass it."
        )
    else:
        logger.info("OPENAI_API_KEY detected — service ready to roll.")

    yield  # Application runs here

    logger.info("LLM Hallucination Tester shutting down.")


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        A fully configured :class:`FastAPI` instance.
    """
    application = FastAPI(
        title="LLM Hallucination Tester",
        description=(
            "A service that uses OpenAI (via LangChain) "
            "to generate factual questions about a topic, answer them, and then "
            "automatically fact-check each answer for hallucinations."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # -----------------------------------------------------------------------
    # CORS Middleware
    # -----------------------------------------------------------------------
    allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
    if allowed_origins_env == "*":
        origins = ["*"]
    else:
        origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.mount("/static", StaticFiles(directory="app/static"), name="static")

    return application


app: FastAPI = create_app()

# ---------------------------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all exception handler that prevents stack traces leaking to clients.

    Args:
        request: The incoming HTTP request.
        exc:     The unhandled exception.

    Returns:
        A JSON 500 response with a generic error message.
    """
    logger.exception("Unhandled exception for %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please try again."},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get(
    "/",
    summary="Serve Web UI",
    tags=["UI"],
)
async def serve_index() -> FileResponse:
    """
    Serve the premium HTML/CSS dashboard interface for testing hallucinations.
    """
    return FileResponse("app/static/index.html")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Operations"],
)
async def health_check() -> HealthResponse:
    """
    Liveness probe endpoint.

    Returns basic service metadata so that Cloud Run health checks and
    monitoring tools can verify the service is running.
    """
    return HealthResponse()


@app.post(
    "/check-hallucination",
    response_model=HallucinationResponse,
    summary="Run hallucination check",
    tags=["Hallucination Testing"],
    status_code=status.HTTP_200_OK,
)
async def check_hallucination(payload: HallucinationRequest) -> HallucinationResponse:
    """
    Execute the full three-stage hallucination-checking pipeline.

    **Pipeline stages:**

    1. **Question Generation** — OpenAI generates ``num_questions`` factual
       questions about ``topic``.
    2. **Answer Generation** — OpenAI answers each question concurrently.
    3. **Fact-Checking** — A second OpenAI call scores each answer as
       ACCURATE, HALLUCINATED, or UNCERTAIN with a confidence score.

    Args:
        payload: Validated :class:`HallucinationRequest` from the request body.

    Returns:
        :class:`HallucinationResponse` with per-question verdicts and an
        aggregate hallucination rate.

    Raises:
        HTTPException 400: If ``num_questions`` exceeds the server-side cap.
        HTTPException 401: If the OpenAI API key is missing.
        HTTPException 502: If the upstream OpenAI API call fails.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OPENAI_API_KEY is not configured on the server.",
        )

    if payload.num_questions > MAX_QUESTIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"num_questions exceeds the server maximum of {MAX_QUESTIONS}.",
        )

    logger.info(
        "Received /check-hallucination request — topic=%r  num_questions=%d  model=%s  reference_mode=%s",
        payload.topic,
        payload.num_questions,
        payload.model,
        "Yes" if payload.reference else "No",
    )

    try:
        checker = HallucinationChecker(
            model_name=payload.model,
            api_key=OPENAI_API_KEY,
        )
        result = await checker.run(
            topic=payload.topic,
            num_questions=payload.num_questions,
            reference=payload.reference,
        )
    except ValueError as exc:
        logger.error("Validation / parsing error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Upstream LLM call failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to communicate with the upstream LLM provider.",
        ) from exc

    return result
