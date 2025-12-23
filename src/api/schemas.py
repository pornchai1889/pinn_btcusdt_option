# src/api/schemas.py
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class OptionType(str, Enum):
    """
    Enumeration for supported option types.
    Using String Enum ensures strict validation against 'call' or 'put'.
    """
    CALL = "call"
    PUT = "put"


class OptionPricingRequest(BaseModel):
    """
    Data Transfer Object (DTO) for Option Pricing Request.
    
    Validates financial parameters to ensure they fall within 
    meaningful mathematical bounds before reaching the inference engine.
    """
    
    # 1. Market Parameters
    spot_price: float = Field(
        ..., 
        gt=0.0, 
        description="Current market price of the underlying asset (S). Must be positive."
    )
    strike_price: float = Field(
        ..., 
        gt=0.0, 
        description="Strike price of the option (K). Must be positive."
    )
    time_to_maturity: float = Field(
        ..., 
        ge=0.0, 
        description="Time to expiration in years (T). E.g., 0.5 for 6 months."
    )
    risk_free_rate: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Annualized risk-free interest rate (r). E.g., 0.05 for 5%."
    )
    volatility: float = Field(
        ..., 
        gt=0.0, 
        le=5.0, 
        description="Annualized volatility (sigma). Must be positive. Capped at 500% for stability."
    )
    
    # 2. Option Configuration
    option_type: OptionType = Field(
        ..., 
        description="Type of the option contract: 'call' or 'put'."
    )

    # 3. Advanced Configuration (Optional)
    # Allows researchers to tag requests for A/B testing or logging
    request_id: Optional[str] = Field(
        None, 
        description="Optional unique identifier for tracing this request in logs."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "spot_price": 100.0,
                "strike_price": 100.0,
                "time_to_maturity": 1.0,
                "risk_free_rate": 0.05,
                "volatility": 0.2,
                "option_type": "call",
                "request_id": "exp-001"
            }
        }
    )

    @field_validator('time_to_maturity')
    @classmethod
    def prevent_negative_time(cls, v: float) -> float:
        """
        Explicit validator to handle edge cases for Time.
        While T=0 is theoretically allowed (Payoff), negative time is physically impossible.
        """
        if v < 0:
            raise ValueError("Time to maturity cannot be negative.")
        return v


class GreeksResponse(BaseModel):
    """
    Encapsulates the sensitivity metrics (The Greeks).
    Calculated via Automatic Differentiation (Autograd) in the PINN model.
    """
    delta: float = Field(..., description="Sensitivity to Spot Price (dV/dS).")
    gamma: float = Field(..., description="Sensitivity to Delta (d2V/dS2). Convexity.")
    theta: float = Field(..., description="Sensitivity to Time (dV/dt). Time decay.")
    vega: float = Field(..., description="Sensitivity to Volatility (dV/dSigma).")
    rho: float = Field(..., description="Sensitivity to Interest Rate (dV/dr).")


class PricingResponse(BaseModel):
    """
    Standardized response format for the Inference API.
    Includes the calculated price, full Greek metrics, and performance metadata.
    """
    price: float = Field(..., description="Predicted Option Price (V).")
    greeks: GreeksResponse = Field(..., description="Associated Greek risk metrics.")
    
    # Metadata for Research & Monitoring
    model_version: str = Field(..., description="Identifier of the loaded model checkpoint.")
    inference_time_ms: float = Field(..., description="Total execution time in milliseconds.")
    device: str = Field(..., description="Hardware used for inference (CPU/CUDA).")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "price": 10.4502,
                "greeks": {
                    "delta": 0.6368,
                    "gamma": 0.0187,
                    "theta": -6.41,
                    "vega": 37.52,
                    "rho": 53.21
                },
                "model_version": "pinn_v1_call",
                "inference_time_ms": 1.25,
                "device": "cpu"
            }
        }
    )