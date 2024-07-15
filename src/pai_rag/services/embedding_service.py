from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch

from pai_rag.modules.embedding.my_ort_embedding import MyORTModelForFeatureExtraction
from transformers import AutoTokenizer
from starlette.concurrency import run_in_threadpool

onnx_path = "/huggingface/sentence_transformers/bge-small-zh-v1.5-onnx"
model = MyORTModelForFeatureExtraction.from_pretrained(
    onnx_path, file_name="model_optimized.onnx", provider="CUDAExecutionProvider"
)
tokenizer = AutoTokenizer.from_pretrained(onnx_path)
max_length = model.config.max_position_embeddings


class EmbeddingInput(BaseModel):
    text: str = None


app = FastAPI()


def embed(text):
    encoded_input = tokenizer(
        text,
        padding=True,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    model_output = model(**encoded_input)
    embeddings = model_output[0][:, 0]

    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()


@app.post("/embedding")
async def get_embedding(input: EmbeddingInput):
    embeddings = await run_in_threadpool(embed, input.text)
    return {"embedding": embeddings}


uvicorn.run(app, host="0.0.0.0", port=9233)
