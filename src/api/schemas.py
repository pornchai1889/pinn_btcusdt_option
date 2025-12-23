# src/api/schemas.py
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class OptionType(str, Enum):
    """
    Enumeration for supported option types.
    Strictly differentiates between 'call' and 'put' contracts.
    """

    CALL = "call"
    PUT = "put"


class OptionPricingRequest(BaseModel):
    """
    Data Transfer Object (DTO) for Option Pricing Request.

    Implements strict boundary validation to ensure numerical stability
    within the PINN inference engine.
    """

    # 1. Market Parameters
    spot_price: float = Field(
        ...,
        gt=0.0,
        description="Current market price of the underlying asset (S). Must be strictly positive (>0).",
    )
    strike_price: float = Field(
        ...,
        gt=0.0,
        description="Strike price of the option (K). Must be strictly positive (>0).",
    )
    time_to_maturity: float = Field(
        ...,
        ge=0.0,
        description="Time to expiration in years (T). E.g., 0.5 for 6 months. Must be non-negative.",
    )
    risk_free_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Annualized risk-free interest rate (r). Range [0.0, 1.0] (0% to 100%).",
    )
    volatility: float = Field(
        ...,
        gt=0.0,
        le=5.0,
        description="Annualized volatility (sigma). Range (0.0, 5.0]. Capped at 500% for model stability.",
    )

    # 2. Option Configuration
    option_type: OptionType = Field(
        ..., description="Contract type specification: 'call' or 'put'."
    )

    # 3. Advanced Configuration (Optional)
    request_id: Optional[str] = Field(
        None,
        description="Optional unique identifier for request tracing and log correlation.",
    )

    # Pydantic V2 Configuration
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "spot_price": 100.0,
                "strike_price": 100.0,
                "time_to_maturity": 0.25,
                "risk_free_rate": 0.05,
                "volatility": 0.5,
                "option_type": "call",
                "request_id": "exp-2025-batch1",
            }
        }
    )


class GreeksResponse(BaseModel):
    """
    Encapsulates the financial sensitivity metrics (The Greeks).
    Computed via Automatic Differentiation (Autograd) w.r.t input tensors.

    Note: Descriptions use standard notation (dV/dx) for universal compatibility.
    """

    delta: float = Field(
        ..., description="Delta: First-order sensitivity to Spot Price (dV/dS)."
    )
    gamma: float = Field(
        ...,
        description="Gamma: Second-order sensitivity to Spot Price (d2V/dS2). Convexity.",
    )
    theta: float = Field(..., description="Theta: Sensitivity to Time decay (dV/dt).")
    vega: float = Field(..., description="Vega: Sensitivity to Volatility (dV/dSigma).")
    rho: float = Field(..., description="Rho: Sensitivity to Risk-free Rate (dV/dr).")


class PricingResponse(BaseModel):
    """
    Standardized response format for the PINN Inference Service.
    Delivers the predicted price along with risk metrics and execution metadata.
    """

    price: float = Field(
        ..., description="Predicted Option Price (V) from the PINN model."
    )
    greeks: GreeksResponse = Field(
        ..., description="Associated Greek risk metrics derived via Autograd."
    )

    # Metadata for System Monitoring & Reproducibility
    model_version: str = Field(
        ..., description="Checkpoint identifier of the loaded model."
    )
    inference_time_ms: float = Field(
        ..., description="Total inference latency in milliseconds (ms)."
    )
    device: str = Field(
        ..., description="Compute unit used for inference (e.g., 'cpu', 'cuda:0')."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "price": 10.4502,
                "greeks": {
                    "delta": 0.6368,
                    "gamma": 0.0187,
                    "theta": -6.4105,
                    "vega": 37.5210,
                    "rho": 53.2100,
                },
                "model_version": "pinn_put_v1",
                "inference_time_ms": 1.25,
                "device": "cpu",
            }
        }
    )
