# src/api/routes.py
import logging
from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any, List  # Added List

# Project Modules
from src.api.schemas import OptionPricingRequest, PricingResponse

# Dependency Injection Setup
import src.api.services as services
from src.api.services import InferenceEngine

# Setup Logger
logger = logging.getLogger("PINN_API_Routes")

# Initialize Router
router = APIRouter(prefix="/v1", tags=["Option Pricing Inference"])


def get_inference_engine() -> InferenceEngine:
    """
    Dependency Injection for the Inference Engine.
    Ensures that the server refuses traffic if the models are not properly loaded (Health Check).
    """
    if services.inference_engine is None:
        logger.critical("Inference Engine accessed but not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference Engine is not ready. Models are loading or failed to initialize.",
        )
    return services.inference_engine


@router.post(
    "/predict",
    response_model=List[PricingResponse],  # Changed to List
    status_code=status.HTTP_200_OK,
    summary="Compute Option Price & Greeks (Batch Support)",
    response_description="Returns a list of calculated prices and sensitivity metrics.",
)
async def predict_batch(
    requests: List[OptionPricingRequest],  # Changed to List input
    engine: InferenceEngine = Depends(get_inference_engine),
) -> List[PricingResponse]:  # Changed return type hint
    """
    **Batch Inference Endpoint**

    Performs a forward pass on the Physics-Informed Neural Network (PINN) for a list of requests.
    This allows processing multiple option scenarios in a single HTTP transaction,
    significantly reducing network overhead.

    **Features:**
    - **Batch Processing:** Accepts an array of option parameters.
    - **Atomic Validation:** Validates all inputs before processing.
    - **Autograd:** Computes exact Greeks (Delta, Gamma, etc.) for each item.

    **Error Handling:**
    - Returns 422 if any item in the batch violates domain constraints.
    - Returns 500 for internal calculation errors.
    """
    try:
        # Delegate batch logic to the Service Layer
        responses = engine.predict_batch(requests)
        return responses

    except ValueError as ve:
        logger.error(f"Validation/Math Error in batch: {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Mathematical constraints violated: {str(ve)}",
        )

    except Exception as e:
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
    """
    if services.inference_engine is None:
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
