from utils.http_session import HttpSessionShared
from db.models.knowledgebase.reranker import RerankerModelEntity
from db.models.knowledgebase.embedding import EmbeddingModelEntity
from db.models.llm import LlmModelEntity
from common.llm.models import llm_url_to_model_provider_id_map
from common.encrypt_utils import encrypt_key
import os
from loguru import logger


def format_endpoint(endpoint: str) -> str:
    """
    Format the endpoint to the correct format.
    """
    if not endpoint:
        return endpoint

    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint = endpoint + "/v1"
    return endpoint

class BailianModelService:
    def __init__(self):
        self.endpoint = os.environ.get("BAILIAN_CONSOLE_ENDPOINT", "").rstrip("/")
        assert self.endpoint, "BAILIAN_CONSOLE_ENDPOINT is not set."
        logger.info(f"Bailian model service initialized with endpoint: {self.endpoint}")

    async def _get_bailian_model_by_provider_model_id(self, provider_name: str, model_id: str, tenant_id: str) -> dict:
        """
        Get a Bailian model entity by provider and model id.

        Args:
            provider_name: Bailian provider name
            model_id: Bailian model_id
            tenant_id: Tenant id

        Returns:
            dict of Bailian model metadata if found, None otherwise
        """
        logger.info(f"Getting Bailian model {model_id} by provider {provider_name} and tenant {tenant_id}.")
        session = await HttpSessionShared.ensure_session()

        # Bailian provider name is not consistent with the model provider name, so we need to map it.
        if provider_name.lower() == "dashscope":
            provider_name = "Tongyi"

        params = {
            "model_id": model_id,
            "provider": provider_name,
        }
        headers = {
            "X-TENANT-ID": tenant_id,
        }
        async with session.get(f"{self.endpoint}/infra/v1/models/metadata", params=params, headers=headers) as response:
            if response.status == 200:
                response_data = await response.json()
                data = response_data["data"]
                if not data:
                    logger.error(f"Bailian model {model_id} by provider {provider_name} and tenant {tenant_id} not found.")
                    raise ValueError(f"Bailian model {model_id} by provider {provider_name} and tenant {tenant_id} not found.")
                logger.info(f"Bailian model {model_id} by provider {provider_name} and tenant {tenant_id} found: {data}")
                return data
            else:
                logger.error(f"Failed to get Bailian model {model_id} by provider {provider_name} and tenant {tenant_id}: {response.status} {await response.text()}")
                raise Exception(f"Failed to get Bailian model {model_id} by provider {provider_name} and tenant {tenant_id}: {response.status} {await response.text()}")

    async def get_reranker_model_by_provider_model_id(self, provider_name: str, model_id: str, tenant_id: str) -> dict:
        """
        Get a Reranker model entity by provider and model id.

        Args:
            provider_name: Reranker provider name
            model_id: Reranker model_id
            tenant_id: Tenant id
        """
        model_dict = await self._get_bailian_model_by_provider_model_id(provider_name, model_id, tenant_id)
        assert model_dict["type"] == "rerank", f"Model {model_id} is not a reranker model."

        rerank_type = "dashscope" if provider_name in ["dashscope", "Tongyi"] else "openai_like"

        reranker_model = RerankerModelEntity.model_validate({
            "tenant_id": tenant_id,
            "model_id": model_id,
            "model_name": model_dict["model_name"],
            "base_url": model_dict["endpoint"],
            "encrypted_api_key": encrypt_key(model_dict["api_key"]),
            "type": rerank_type,
            "provider_name": provider_name,
        })
        logger.info(f"Reranker model {model_id} created: {reranker_model}")
        return reranker_model

    async def get_embedding_model_by_provider_model_id(self, provider_name: str, model_id: str, tenant_id: str) -> dict:
        """
        Get an Embedding model entity by provider and model id.

        Args:
            provider_name: Embedding provider name
            model_id: Embedding model_id
            tenant_id: Tenant id
        """
        model_dict = await self._get_bailian_model_by_provider_model_id(provider_name, model_id, tenant_id)
        assert model_dict["type"] == "text_embedding", f"Model {model_id} is not an embedding model."
        embedding_model = EmbeddingModelEntity.model_validate({
            "tenant_id": tenant_id,
            "model_id": model_id,
            "model_name": model_dict["model_name"],
            "endpoint": format_endpoint(model_dict["endpoint"]),
            "encrypted_api_key": encrypt_key(model_dict["api_key"]),
            "provider_name": provider_name,
            "is_default": False,
            "is_ready": True,
            "embed_batch_size": 10,
            "type": "openai_like"
        })
        logger.info(f"Embedding model {model_id} created: {embedding_model}")
        return embedding_model

    async def get_llm_model_by_provider_model_id(self, provider_name: str, model_id: str, tenant_id: str) -> dict:
        """
        Get a LLM model entity by provider and model id.

        Args:
            provider_name: LLM provider name
            model_id: LLM model_id
            tenant_id: Tenant id
        """
        model_dict = await self._get_bailian_model_by_provider_model_id(provider_name, model_id, tenant_id)
        assert model_dict["type"] == "llm", f"Model {model_id} is not an LLM model."

        vision_support = "vision" in model_dict.get("tags", [])
        enable_thinking = "reasoning" in model_dict.get("tags", [])
        llm_model = LlmModelEntity.model_validate({
            "tenant_id": tenant_id,
            "model_id": model_id,
            "model_name": model_dict["model_name"],
            "model": model_dict["model_name"],
            "base_url": format_endpoint(model_dict["endpoint"]),
            "encrypted_api_key": encrypt_key(model_dict["api_key"]),
            "type": llm_url_to_model_provider_id_map.get(model_dict["endpoint"], "openai_like"),
            "provider_name": provider_name,
            "vision_support": vision_support,
            "enable_thinking": enable_thinking,
            "temperature": model_dict.get("temperature", 0.1),
            "context_window": model_dict.get("context_window", 32000),
            "max_tokens": model_dict.get("max_tokens", 4000),
        })
        logger.info(f"LLM model {model_id} created: {llm_model}")
        return llm_model
