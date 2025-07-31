from enum import Enum
import os


class FeatureFlags(Enum):
    """
    Feature flags for the app.
    """

    # Enable/disable the mcp feature
    MCP = "FEATURE_MCP"


def is_feature_enabled(feature: FeatureFlags) -> bool:
    """
    Check if a feature is enabled.

    Args:
        feature (FeatureFlags): The feature to check.

    Returns:
        bool: True if the feature is enabled, False otherwise.
    """
    value = os.getenv(feature.value, "false").lower()
    return value in ("true", "1", "yes", "y", "on")
