from typing import List, Any, Dict


def components_to_dict(components: List[Any]) -> Dict[str, Any]:
    return {c.elem_id: c for c in components}


def check_variables_in_string(text, variables):
    missing_variables = [var for var in variables if f"{{{var}}}" not in text]
    if missing_variables:
        raise ValueError(f"以下变量名缺失: {', '.join(missing_variables)}")
