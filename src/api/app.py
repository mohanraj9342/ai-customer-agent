"""
src/api/app.py
==============
Phase 15 — FastAPI Application Factory & Lifecycle Management.

Configures:
- Application lifespan pre-warming of model and retrieval pipelines.
- CORS middleware with wildcard regex support for Vercel preview environments.
- Global exception handlers preventing secret and traceback leakage.
- Health, readiness, and chat routes.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.config import ApiConfig, load_api_config
from src.api.routes import router
from src.api.schemas import ErrorResponse
from src.evaluation.agent_reviewer import AgentReviewer
from src.generation.agent_orchestrator import GroundedSupportAgent, IntentPredictor
from src.retrieval.historical_response_retriever import HistoricalResponseRetriever

logger = logging.getLogger(__name__)


def create_app(
    agent: Optional[GroundedSupportAgent] = None,
    reviewer: Optional[AgentReviewer] = None,
    api_config: Optional[ApiConfig] = None,
) -> FastAPI:
    """
    Constructs and configures the FastAPI application.

    Args:
        agent: Optional pre-configured GroundedSupportAgent (e.g. for testing).
        reviewer: Optional pre-configured AgentReviewer (e.g. for testing).
        api_config: Optional API configuration override.

    Returns:
        Configured FastAPI application ready for ASGI serving.
    """
    config = api_config or load_api_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: pre-warm pipeline assets if not injected
        if getattr(app.state, "agent", None) is None:
            logger.info("Pre-warming Phase 10-13 support pipelines...")
            retriever = HistoricalResponseRetriever()
            intent_predictor = IntentPredictor()
            app.state.agent = GroundedSupportAgent(
                retriever=retriever,
                intent_predictor=intent_predictor,
            )
        if getattr(app.state, "reviewer", None) is None:
            app.state.reviewer = AgentReviewer()
        app.state.api_config = config
        logger.info("Pipeline pre-warming complete. Ready for requests.")
        yield
        logger.info("Application shutting down.")

    app = FastAPI(
        title="AppleSupport AI Customer Agent API",
        description=(
            "Production backend service for the Grounded AI Customer Support Agent. "
            "Integrates Phase 10 Intent Classification, Phase 11 Historical Retrieval, "
            "Phase 12 Grounded Generation, and Phase 13 Independent Quality Review."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # If explicit instances were provided, store them immediately
    if agent is not None:
        app.state.agent = agent
    if reviewer is not None:
        app.state.reviewer = reviewer
    app.state.api_config = config

    # Configure CORS for local development and future Vercel frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_origin_regex=config.cors_origin_regex,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Global Exception Handlers (Safe RFC-Compliant Errors)
    # -----------------------------------------------------------------------
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        details = []
        for err in exc.errors():
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            msg = err.get("msg", "Invalid value")
            details.append({"location": loc, "issue": msg})
        return JSONResponse(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            content=ErrorResponse(
                status="error",
                code="VALIDATION_ERROR",
                message="Request validation failed. Please inspect input parameters.",
                details=details,
            ).model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                status="error",
                code=f"HTTP_{exc.status_code}",
                message=str(exc.detail),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled server exception: %s", str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                status="error",
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected server error occurred. Please try again later.",
            ).model_dump(),
        )

    # Include routes
    app.include_router(router)

    return app


# Default application instance for Uvicorn ASGI server
app = create_app()
