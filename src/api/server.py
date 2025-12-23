# src/api/server.py
import os
import sys
import logging
import uvicorn
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import AsyncGenerator

# --- Environment Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Project Modules
from src.api.routes import router as api_router
from src.api.services import InferenceEngine, inference_engine
from src.api.config import settings  # <--- Import Settings here

# Setup Logger
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG_MODE else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("PINN_Server")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan Context Manager.
    Initializes the Inference Engine using paths from Configuration.
    """
    logger.info(f"--- Starting {settings.APP_NAME} v{settings.APP_VERSION} ---")

    # Determine Device
    device_str = settings.DEVICE
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Compute Device: {device_str}")

    # Initialize Global Inference Engine
    import src.api.services as services

    try:
        services.inference_engine = InferenceEngine(
            call_model_dir=settings.CALL_MODEL_DIR,
            put_model_dir=settings.PUT_MODEL_DIR,
            device_str=device_str,
        )
        logger.info("Inference Engine successfully loaded.")
    except Exception as e:
        logger.critical(f"Startup Failed: {e}")
        # In production, you might want to exit here, but for now we let it run (unhealthy state)

    yield

    logger.info("--- Shutting down ---")
    services.inference_engine = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    @app.get("/", tags=["System"])
    async def root():
        return {
            "app": settings.APP_NAME,
            "status": "online",
            "version": settings.APP_VERSION,
        }

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "src.api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG_MODE,
    )
