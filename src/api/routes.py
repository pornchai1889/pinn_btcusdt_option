# src/api/routes.py
import logging
from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any

# Project Modules
from src.api.schemas import OptionPricingRequest, PricingResponse, OptionType
from src.api.services import InferenceEngine, inference_engine

# Setup Logger
logger = logging.getLogger("PINN_API_Routes")

# Initialize Router
# Tags are used for grouping operations in the Swagger UI documentation
router = APIRouter(prefix="/v1", tags=["Option Pricing Inference"])


def get_inference_engine() -> InferenceEngine:
    """
    Dependency Injection for the Inference Engine.
    Ensures that the server refuses traffic if the models are not properly loaded (Health Check).

    Raises:
        HTTPException (503): If the inference engine singleton is not initialized.
    """
    if inference_engine is None:
        logger.critical("Inference Engine accessed but not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference Engine is not ready. Models are loading or failed to initialize.",
        )
    return inference_engine


@router.post(
    "/predict",
    response_model=PricingResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute Option Price & Greeks",
    response_description="Returns the calculated price and sensitivity metrics (Greeks).",
)
async def predict(
    request: OptionPricingRequest,
    engine: InferenceEngine = Depends(get_inference_engine),
) -> PricingResponse:
    """
    **Main Inference Endpoint**

    Performs a forward pass on the Physics-Informed Neural Network (PINN) to compute:
    1.  **Option Price ($V$):** Theoretical value based on market parameters.
    2.  **Greeks:** First and second-order derivatives ($\Delta, \Gamma, \Theta, \nu, \rho$)
        calculated via exact Automatic Differentiation (Autograd).

    **Error Handling:**
    - Validates input boundaries via Pydantic schemas.
    - Catches numerical instabilities (NaN/Inf) during Autograd execution.
    """
    try:
        # Delegate logic to the Service Layer
        response = engine.predict(request)
        return response

    except ValueError as ve:
        # Handle mathematical domain errors (e.g., negative volatility leaked through)
        logger.error(f"Validation/Math Error: {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Mathematical constraints violated: {str(ve)}",
        )

    except Exception as e:
        # Handle unexpected runtime errors (e.g., CUDA OOM, Autograd failure)
        logger.error(f"Internal Inference Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal calculation error. Please contact support.",
        )


@router.get(
    "/health", status_code=status.HTTP_200_OK, summary="Liveness Probe", tags=["System"]
)
async def health_check() -> Dict[str, str]:
    """
    **Kubernetes/LoadBalancer Health Check**

    Verifies that:
    1. The API Server is running.
    2. The Inference Engine (Models) are loaded in memory.

    Returns 200 OK if healthy, otherwise 503.
    """
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System starting up...",
        )
    return {"status": "healthy", "engine": "active"}


@router.get(
    "/metadata",
    status_code=status.HTTP_200_OK,
    summary="Model Metadata",
    tags=["System"],
)
async def get_model_metadata(
    engine: InferenceEngine = Depends(get_inference_engine),
) -> Dict[str, Any]:
    """
    **Research Metadata Endpoint**

    Exposes technical details about the currently loaded models.
    Useful for experiment tracking and ensuring client-side parameter alignment.
    """
    return {
        "call_model": {
            "version": engine.call_bundle.run_dir.name,
            "training_range": {
                "S": [engine.call_bundle.S_min, engine.call_bundle.S_max],
                "K": [engine.call_bundle.K_min, engine.call_bundle.K_max],
                "T": [engine.call_bundle.t_min, engine.call_bundle.t_max],
            },
        },
        "put_model": {
            "version": engine.put_bundle.run_dir.name,
            "training_range": {
                "S": [engine.put_bundle.S_min, engine.put_bundle.S_max],
                "K": [engine.put_bundle.K_min, engine.put_bundle.K_max],
                "T": [engine.put_bundle.t_min, engine.put_bundle.t_max],
            },
        },
        "device": str(engine.device),
    }
