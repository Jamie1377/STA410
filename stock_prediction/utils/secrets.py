"""
Secrets management for STA410 Stock Prediction System.

Provides secure loading of API keys and sensitive credentials from environment variables,
with proper error handling and fallback strategies.
"""

import os
from typing import Optional, Tuple
from stock_prediction.utils.logger import setup_logger
from stock_prediction.utils.exceptions import SecretsManagementError

logger = setup_logger(__name__)


def get_alpaca_credentials() -> Tuple[str, str]:
    """
    Retrieve Alpaca API credentials from environment variables.
    
    Returns:
        Tuple of (api_key, secret_key)
    
    Raises:
        SecretsManagementError: If credentials not found in environment
    
    Note:
        Credentials should be set via:
        - Environment variables: ALPACA_API_KEY, ALPACA_SECRET_KEY
        - .env file (loaded automatically)
        - Secrets manager (in production)
    """
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        logger.error(
            "Alpaca API credentials not found. "
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
        )
        raise SecretsManagementError(
            "Alpaca credentials missing. Set environment variables: "
            "ALPACA_API_KEY, ALPACA_SECRET_KEY"
        )
    
    logger.debug("Alpaca credentials loaded from environment")
    return api_key, secret_key


def get_alpaca_base_url() -> str:
    """
    Get Alpaca API base URL based on environment.
    
    Returns:
        API base URL for paper or live trading
    """
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    # Security: warn if using live trading
    if "paper" not in base_url.lower():
        logger.warning(
            "Using LIVE trading API endpoint. This will execute real trades. "
            "Ensure PAPER_TRADING=false is intentional!"
        )
    
    return base_url


def validate_api_credentials(api_key: str, secret_key: str) -> bool:
    """
    Perform basic validation of API credentials format.
    
    Args:
        api_key: API key to validate
        secret_key: Secret key to validate
    
    Returns:
        True if credentials appear valid (basic checks only)
    
    Raises:
        SecretsManagementError: If credentials are clearly invalid
    """
    if not api_key or len(api_key) < 10:
        raise SecretsManagementError("API key appears invalid (too short)")
    
    if not secret_key or len(secret_key) < 10:
        raise SecretsManagementError("Secret key appears invalid (too short)")
    
    logger.debug("API credentials passed basic validation")
    return True


def load_secrets_from_file(filepath: str) -> dict:
    """
    DEPRECATED: Load secrets from JSON file (legacy compatibility).
    
    This method is provided for backward compatibility only.
    Use environment variables instead in new code.
    
    Args:
        filepath: Path to JSON file with secrets
    
    Returns:
        Dictionary with secrets
    
    Raises:
        SecretsManagementError: If file not found or invalid
    """
    logger.warning(
        "Loading secrets from file is deprecated. "
        "Use environment variables (ALPACA_API_KEY, etc.) instead."
    )
    
    import json
    from pathlib import Path
    
    try:
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"Secrets file not found: {filepath}")
        
        with open(file_path) as f:
            secrets = json.load(f)
        
        logger.warning(f"Loaded secrets from file {filepath} - ENSURE THIS IS NOT COMMITTED")
        return secrets
    
    except FileNotFoundError as e:
        raise SecretsManagementError(f"Secrets file not found: {filepath}") from e
    except json.JSONDecodeError as e:
        raise SecretsManagementError(f"Invalid JSON in secrets file: {filepath}") from e
    except Exception as e:
        raise SecretsManagementError(f"Error loading secrets: {str(e)}") from e


def mask_secret(secret: str, visible_chars: int = 4) -> str:
    """
    Mask a secret for logging purposes.
    
    Args:
        secret: Secret to mask
        visible_chars: Number of characters to show at the end
    
    Returns:
        Masked string (e.g., "****ABCD")
    
    Example:
        >>> mask_secret("my-secret-key-12345", 4)
        '***-secret-key-12345'
    """
    if len(secret) <= visible_chars:
        return "*" * len(secret)
    
    return "*" * (len(secret) - visible_chars) + secret[-visible_chars:]
