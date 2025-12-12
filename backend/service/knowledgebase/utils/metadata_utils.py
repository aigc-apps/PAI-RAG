from typing import Tuple
from db.models.knowledgebase.metadata import MetadataValueType
from loguru import logger


def validate_metadata_value(
    value: str | int | float, value_type: str
) -> Tuple[bool, str | int | float]:
    """
    Validate and convert metadata value based on value_type.

    Args:
        value: The value to validate
        value_type: The expected type (string, number, datetime)

    Returns:
        Tuple of (is_valid, converted_value)
    """
    if value_type == MetadataValueType.STRING:
        return True, str(value)
    elif value_type == MetadataValueType.NUMBER:
        try:
            if isinstance(value, (int, float)):
                return True, value
            elif isinstance(value, str):
                # Try to convert to float first (handles both int and float)
                num_value = float(value)
                # If it's a whole number, return int
                if num_value.is_integer():
                    return True, int(num_value)
                return True, num_value
            else:
                return False, value
        except (ValueError, TypeError):
            return False, value
    elif value_type == MetadataValueType.DATETIME:
        try:
            if isinstance(value, (int, float)):
                # Assume it's a timestamp
                return True, float(value)
            elif isinstance(value, str):
                # Try to parse as float timestamp
                return True, float(value)
            else:
                return False, value
        except (ValueError, TypeError):
            return False, value
    else:
        logger.warning(f"Invalid metadata value type: {value_type}")
        return False, value
