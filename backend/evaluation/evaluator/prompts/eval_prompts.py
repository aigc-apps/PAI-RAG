import yaml
from pathlib import Path
from loguru import logger

# Load eval prompts from YAML file
_EVAL_PROMPTS_CACHE = None

def _load_eval_prompts():
    """Load evaluation prompts from YAML file"""
    global _EVAL_PROMPTS_CACHE
    if _EVAL_PROMPTS_CACHE is not None:
        return _EVAL_PROMPTS_CACHE

    # Get the project root directory (assuming this file is in backend/evaluation/evaluator/prompts/)
    current_file = Path(__file__)
    # Go up 5 levels: prompts -> evaluator -> evaluation -> backend -> project_root
    project_root = current_file.parent.parent.parent.parent.parent
    eval_prompts_file = project_root / "resources" / "prompts" / "eval_prompts.yaml"

    if not eval_prompts_file.exists():
        error_msg = f"Eval prompts file not found at {eval_prompts_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with open(eval_prompts_file, 'r', encoding='utf-8') as f:
            prompts_data = yaml.safe_load(f)

        if not prompts_data:
            error_msg = f"Eval prompts file is empty or invalid: {eval_prompts_file}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if 'llm_judge_prompt' not in prompts_data or not prompts_data['llm_judge_prompt']:
            error_msg = "Missing required prompt key 'llm_judge_prompt' in YAML file"
            logger.error(error_msg)
            raise ValueError(error_msg)

        _EVAL_PROMPTS_CACHE = {
            'llm_judge_prompt': prompts_data['llm_judge_prompt'],
        }

        logger.info(f"Loaded eval prompts from {eval_prompts_file}")
        return _EVAL_PROMPTS_CACHE
    except yaml.YAMLError as e:
        error_msg = f"Failed to parse YAML file {eval_prompts_file}: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Failed to load eval prompts from {eval_prompts_file}: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

# Load prompts on module import
_eval_prompts = _load_eval_prompts()

LLM_JUDGE_PROMPT = _eval_prompts['llm_judge_prompt']
