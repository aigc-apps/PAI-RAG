from typing import List, Any, Dict


def components_to_dict(components: List[Any]) -> Dict[str, Any]:
    return {c.elem_id: c for c in components}
