import logging
import httpx
import asyncio
import tempfile
import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.schema import BaseNode, MetadataMode
from typing import Dict, List, Sequence

logger = logging.getLogger(__name__)


async def download_url(url, temp_dir):
    if not url:
        return None

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code == 200:
            # Create a temporary file in the temporary directory
            temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir.name)
            temp_file.write(response.content)
            temp_file.close()
            return temp_file.name
        else:
            logger.error(
                f"Failed to download {url}: Status code {response.status_code}"
            )
            return None


async def load_images_from_nodes(nodes: Sequence[BaseNode]) -> List[str]:
    temp_dir = tempfile.TemporaryDirectory()

    tasks = [
        download_url(node.metadata.get("image_url", None), temp_dir) for node in nodes
    ]
    results = await asyncio.gather(*tasks)

    return temp_dir, results


def merge_embeddings_sum_normalize(emb1, emb2):
    merged_emb = np.array(emb1) + np.array(emb2)
    return merged_emb


async def async_embed_nodes(
    nodes: Sequence[BaseNode], embed_model: BaseEmbedding, show_progress: bool = False
) -> Dict[str, List[float]]:
    """Async get embeddings of the given nodes, run embedding model if necessary.

    Args:
        nodes (Sequence[BaseNode]): The nodes to embed.
        embed_model (BaseEmbedding): The embedding model to use.
        show_progress (bool): Whether to show progress bar.

    Returns:
        Dict[str, List[float]]: A map from node id to embedding.
    """
    id_to_embed_map: Dict[str, List[float]] = {}

    texts_to_embed = []
    ids_to_embed = []
    for node in nodes:
        if node.embedding is None:
            ids_to_embed.append(node.node_id)
            texts_to_embed.append(node.get_content(metadata_mode=MetadataMode.EMBED))
        else:
            id_to_embed_map[node.node_id] = node.embedding

    new_embeddings = await embed_model.aget_text_embedding_batch(
        texts_to_embed, show_progress=show_progress
    )

    if isinstance(embed_model, MultiModalEmbedding):
        temp_dir, image_paths = await load_images_from_nodes(nodes)
        image_embeddings = await embed_model.aget_image_embedding_batch(
            image_paths, show_progress=show_progress
        )

        for i in range(len(new_embeddings)):
            if image_embeddings[i]:
                new_embeddings[i] = list(
                    merge_embeddings_sum_normalize(
                        new_embeddings[i], image_embeddings[i]
                    )
                )
        temp_dir.cleanup()

    for new_id, text_embedding in zip(ids_to_embed, new_embeddings):
        id_to_embed_map[new_id] = text_embedding

    return id_to_embed_map
