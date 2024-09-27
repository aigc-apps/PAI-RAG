import gradio as gr
from typing import Any, List
from pai_rag.app.web.rag_client import RagApiError, rag_client
import datetime
from pai_rag.app.web.ui_constants import (
    DEFAULT_EMBED_SIZE,
    EMBEDDING_DIM_DICT,
    LLM_MODEL_KEY_DICT,
    MLLM_MODEL_KEY_DICT,
    EMBEDDING_TYPE_DICT,
    EMBEDDING_MODEL_LINK_DICT,
)


def change_emb_source(source, model):
    return [
        gr.update(visible=(source == "HuggingFace")),
        EMBEDDING_DIM_DICT.get(source, DEFAULT_EMBED_SIZE)
        if source == "HuggingFace"
        else DEFAULT_EMBED_SIZE,
        EMBEDDING_TYPE_DICT.get(model, "Default")
        if source == "HuggingFace"
        else "Default",
        gr.update(
            value=f"Model Introduction: [{model}]({EMBEDDING_MODEL_LINK_DICT[model]})"
            if source == "HuggingFace"
            else ""
        ),
    ]


def change_emb_model(source, model):
    return (
        EMBEDDING_DIM_DICT.get(model, DEFAULT_EMBED_SIZE)
        if source == "HuggingFace"
        else DEFAULT_EMBED_SIZE,
        EMBEDDING_TYPE_DICT.get(model, "Default")
        if source == "HuggingFace"
        else "Default",
        gr.update(
            value=f"Model Introduction: [{model}]({EMBEDDING_MODEL_LINK_DICT[model]})"
            if source == "HuggingFace"
            else ""
        ),
    )


def change_use_oss(use_oss):
    if use_oss:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def choose_use_mllm(value):
    if value:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def change_llm(value):
    eas_visible = value == "PaiEas"
    api_visible = value != "PaiEas"
    model_options = LLM_MODEL_KEY_DICT.get(value, [])

    cur_model = model_options[0] if model_options else ""

    return [
        gr.update(visible=eas_visible),
        gr.update(visible=eas_visible),
        gr.update(visible=eas_visible),
        gr.update(choices=model_options, value=cur_model, visible=api_visible),
    ]


def change_mllm(value):
    eas_visible = value == "PaiEas"
    api_visible = value != "PaiEas"
    model_options = MLLM_MODEL_KEY_DICT.get(value, [])
    cur_model = model_options[0] if model_options else ""
    return [
        gr.update(visible=eas_visible),
        gr.update(visible=api_visible),
        gr.update(choices=model_options, value=cur_model),
    ]


def change_vectordb_conn(vectordb_type):
    adb_visible = False
    hologres_visible = False
    faiss_visible = False
    es_visible = False
    milvus_visible = False
    opensearch_visible = False
    postgresql_visible = False
    if vectordb_type.lower() == "analyticdb":
        adb_visible = True
    elif vectordb_type.lower() == "hologres":
        hologres_visible = True
    elif vectordb_type.lower() == "elasticsearch":
        es_visible = True
    elif vectordb_type.lower() == "milvus":
        milvus_visible = True
    elif vectordb_type.lower() == "faiss":
        faiss_visible = True
    elif vectordb_type.lower() == "opensearch":
        opensearch_visible = True
    elif vectordb_type.lower() == "postgresql":
        postgresql_visible = True

    return [
        gr.update(visible=adb_visible),
        gr.update(visible=hologres_visible),
        gr.update(visible=es_visible),
        gr.update(visible=faiss_visible),
        gr.update(visible=milvus_visible),
        gr.update(visible=opensearch_visible),
        gr.update(visible=postgresql_visible),
    ]


def save_config(input_elements: List[Any]):
    try:
        update_dict = {}
        for element, value in input_elements.items():
            if element.elem_id == "oss_ak":
                value_ak = value
            if element.elem_id == "oss_sk":
                value_sk = value
            update_dict[element.elem_id] = value
        rag_client.patch_config(update_dict)
        return [
            gr.update(
                value=input_oss_ak_sk(value_ak), type="text" if value_ak else "password"
            ),
            gr.update(
                value=input_oss_ak_sk(value_sk), type="text" if value_sk else "password"
            ),
            gr.update(
                value=f"[{datetime.datetime.now()}] Snapshot configuration saved successfully!"
            ),
        ]
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")


def input_oss_ak_sk(input):
    return (input[:2] + "*" * (len(input) - 4) + input[-2:]) if input else input
