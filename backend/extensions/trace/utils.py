from pydantic import BaseModel


def pydantic_to_dict(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode='json')
    elif isinstance(obj, dict):
        return {k: pydantic_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [pydantic_to_dict(item) for item in obj]
    else:
        return obj
