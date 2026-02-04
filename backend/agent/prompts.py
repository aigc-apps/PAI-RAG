import yaml
from pathlib import Path
from loguru import logger

# Load prompts from YAML file
_PROMPTS_CACHE = None

def _load_prompts():
    logger.info("Loading prompts...")
    """Load prompts from YAML file"""
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE

    # Get the project root directory (assuming this file is in backend/agent/)
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    prompts_file = project_root / "resources" / "prompts" / "prompts.yaml"

    if not prompts_file.exists():
        error_msg = f"Prompts file not found at {prompts_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts_data = yaml.safe_load(f)

        if not prompts_data:
            error_msg = f"Prompts file is empty or invalid: {prompts_file}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        required_keys = ['react_prompt']
        missing_keys = [key for key in required_keys if key not in prompts_data or not prompts_data[key]]

        if missing_keys:
            error_msg = f"Missing required prompt keys in YAML file: {missing_keys}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        _PROMPTS_CACHE = {
            "react_prompt": prompts_data['react_prompt'],
        }

        logger.info(f"Loaded prompts from {prompts_file}")
        return _PROMPTS_CACHE
    except yaml.YAMLError as e:
        error_msg = f"Failed to parse YAML file {prompts_file}: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Failed to load prompts from YAML file {prompts_file}: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

# Load prompts on module import
_prompts = _load_prompts()

REACT_PROMPT = _prompts['react_prompt']
