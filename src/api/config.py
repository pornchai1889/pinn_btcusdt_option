# src/api/config.py
import os
import logging
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Setup Logger specifically for Config
logger = logging.getLogger("PINN_Config")


class Settings(BaseSettings):
    """
    Application Configuration Management.

    Reads configuration from environment variables (priority) or .env file.
    Follows the 12-Factor App methodology for configuration.
    """

    # --- Server Configuration ---
    APP_NAME: str = "PINN Bitcoin Option Pricing API"
    APP_VERSION: str = "1.0.0"
    API_HOST: str = Field("0.0.0.0", description="Host to bind the server to")
    API_PORT: int = Field(8000, description="Port to bind the server to")
    DEBUG_MODE: bool = Field(False, description="Enable debug logs and reload")

    # --- Model Paths ---
    # In production, these should be absolute paths provided by the deployment environment
    CALL_MODEL_DIR: str = Field(
        ..., description="Path to the trained Call Option model directory"
    )
    PUT_MODEL_DIR: str = Field(
        ..., description="Path to the trained Put Option model directory"
    )

    # --- Hardware ---
    DEVICE: str = Field(
        "auto",
        description="Force specific device (cpu, cuda). 'auto' detects available hardware.",
    )

    # --- Security (CORS) ---
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="List of allowed origins for CORS. Use ['*'] for development only.",
    )

    # Pydantic Settings Config
    # This tells Pydantic to read from a file named '.env' automatically
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields in .env
    )


# Singleton Instance
try:
    settings = Settings()
    logger.info("Configuration loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load configuration: {e}")
    logger.critical(
        "Please ensure environment variables are set or a .env file exists."
    )
    # We re-raise to crash early if config is invalid (Fail Fast Principle)
    raise e
