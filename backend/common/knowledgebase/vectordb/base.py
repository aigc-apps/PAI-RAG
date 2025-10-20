from common.encrypt_utils import encrypt_key
from common.knowledgebase.types import VectorDbType
from pydantic import BaseModel, ConfigDict


class BaseVectorDbConnection(BaseModel):
    type: VectorDbType = VectorDbType.LOCAL
    model_config = ConfigDict(coerce_numbers_to_str=True)

    @classmethod
    def from_dict(cls, config: dict):
        if config.get("password"):
            config["encrypted_password"] = encrypt_key(config["password"])
            del config["password"]
        if config.get("sk"):
            config["encrypted_sk"] = encrypt_key(config["sk"])
            del config["sk"]

        return cls(**config)
