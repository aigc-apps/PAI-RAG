from io import BytesIO
from loguru import logger
import httpx
import asyncio
import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.schema import BaseNode, MetadataMode, ImageNode
from typing import Dict, List, Sequence


async def download_url(url):
    if not url:
        return None

    image_stream = BytesIO()

    async with httpx.AsyncClient() as client:
        timeout = httpx.Timeout(
            connect=5.0,  # 连接超时时间为5秒
            read=10.0,  # 读取超时时间为10秒
            write=10.0,  # 写入超时时间为10秒
            pool=20.0,  # 连接池超时时间为20秒
        )
        logger.info("download_url url", url)
        try:
            response = await client.get(url, timeout=timeout)
            logger.info(response.text)
        except httpx.RequestError as exc:
            logger.info(f"An error occurred while requesting {exc.request.url!r}.")
        except httpx.TimeoutException as exc:
            logger.info(f"Request timed out: {exc.request.url!r}.")
        except httpx.HTTPStatusError as exc:
            logger.info(f"HTTP error: {exc}")

        if response.status_code == 200:
            # Create a temporary file in the temporary directory
            image_stream.write(response.content)
            image_stream.seek(0)
            return image_stream
        else:
            logger.error(
                f"Failed to download {url}: Status code {response.status_code}"
            )
            return None


async def load_images_from_nodes(nodes: Sequence[ImageNode]) -> List[BytesIO]:
    tasks = [download_url(node.image_url) for node in nodes]
    results = await asyncio.gather(*tasks)
    return results


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
    text_nodes = [node for node in nodes if not isinstance(node, ImageNode)]
    image_nodes = [node for node in nodes if isinstance(node, ImageNode)]

    id_to_embed_map: Dict[str, List[float]] = {}

    texts_to_embed = []
    ids_to_embed = []
    for node in text_nodes:
        if node.embedding is None:
            ids_to_embed.append(node.node_id)
            texts_to_embed.append(node.get_content(metadata_mode=MetadataMode.EMBED))
        else:
            id_to_embed_map[node.node_id] = node.embedding

    new_embeddings = await embed_model.aget_text_embedding_batch(
        texts_to_embed, show_progress=show_progress
    )

    if isinstance(embed_model, MultiModalEmbedding):
        image_list = await load_images_from_nodes(image_nodes)
        active_image_nodes, active_image_list = [], []
        for i, image in enumerate(image_list):
            if image:
                active_image_nodes.append(image_nodes[i])
                active_image_list.append(image_list[i])

        image_embeddings = await embed_model.aget_image_embedding_batch(
            active_image_list, show_progress=show_progress
        )
        for i in range(len(image_embeddings)):
            id_to_embed_map[active_image_nodes[i].node_id] = image_embeddings[i]

    for new_id, text_embedding in zip(ids_to_embed, new_embeddings):
        id_to_embed_map[new_id] = text_embedding

    return id_to_embed_map
