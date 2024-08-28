from llama_index.core.bridge.pydantic import Field, PrivateAttr
import torch
import os
from typing import List, Any
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.schema import ImageType
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from pai_rag.utils.constants import DEFAULT_MODEL_DIR

DEFAULT_CNCLIP_MODEL_DIR = os.path.join(DEFAULT_MODEL_DIR, "cn_clip")
DEFAULT_CNCLIP_MODEL = "ViT-L-14"


class CnClipEmbedding(MultiModalEmbedding):
    """ChineseCLIP embedding models for encoding text and image for Multi-Modal purpose."""

    embed_batch_size: int = Field(default=DEFAULT_EMBED_BATCH_SIZE, gt=0)
    _device: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _preprocess: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = DEFAULT_CNCLIP_MODEL,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )

        if model_name not in available_models():
            raise ValueError(f"Unknown ChineseClip model: {model_name}.")

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model, self._preprocess = load_from_name(
            self.model_name, device=self._device, download_root=DEFAULT_CNCLIP_MODEL_DIR
        )
        self._model.eval()

    @classmethod
    def class_name(cls) -> str:
        return "CnClipEmbedding"

    # TEXT EMBEDDINGS
    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        embedding = self._get_text_embeddings([text])[0]
        return embedding

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        results = []
        for text in texts:
            text_embedding = self._model.encode_text(
                clip.tokenize(text, context_length=512).to(self._device)
            )
            normed_embeddings = text_embedding / text_embedding.norm(
                dim=1, keepdim=True
            )
            results.append(normed_embeddings.tolist()[0])

        return results

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_text_embedding(query)

    # IMAGE EMBEDDINGS

    async def _aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        return self._get_image_embedding(img_file_path)

    def _get_image_embedding(self, img_file_path: ImageType) -> Embedding:
        if not img_file_path:
            return None
        with torch.no_grad():
            image = (
                self._preprocess(Image.open(img_file_path))
                .unsqueeze(0)
                .to(self._device)
            )

            embeddings = self._model.encode_image(image)
            normed_embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            return normed_embeddings.tolist()[0]


if __name__ == "__main__":
    clip_embedding = CnClipEmbedding()

    image_embedding = clip_embedding.get_image_embedding(
        "example_data/cn_clip/pokemon.jpeg"
    )
    texts_embedding = clip_embedding.get_text_embedding_batch(
        ["杰尼龟", "妙蛙种子", "葫芦娃", "皮卡丘"]
    )
    import numpy as np

    with torch.no_grad():
        from torch import nn

        image_features = np.reshape(image_embedding, (1, len(image_embedding)))
        text_features = np.array(texts_embedding)
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        logits_per_image = torch.tensor(
            100 * np.matmul(image_features, text_features.transpose())
        )
        print(logits_per_image)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print(probs)
