import gradio as gr
import os
from typing import Any, List
from pai_rag.app.web.index_utils import components_to_index, index_to_components
from pai_rag.app.web.rag_local_client import RagApiError, rag_client
from pai_rag.core.rag_service import rag_service
from pai_rag.app.web.index_utils import index_related_component_keys
from pai_rag.app.web.tabs.model.index_info import get_index_map
import datetime
from pai_rag.app.web.ui_constants import (
    DEFAULT_EMBED_SIZE,
    DEFAULT_HF_EMBED_MODEL,
    EMBEDDING_DIM_DICT,
    EMBEDDING_TYPE_DICT,
)
from pai_rag.core.rag_index_manager import RagIndexEntry
from pai_rag.integrations.embeddings.pai.pai_embedding_config import (
    HuggingFaceEmbeddingConfig,
)
from pai_rag.integrations.index.pai.vector_store_config import FaissVectorStoreConfig
from pai_rag.utils.constants import DEFAULT_KNOWLEDGE_PATH
from pai_rag.integrations.llms.pai.llm_config import (
    PaiBaseLlmConfig,
)
from loguru import logger


def add_index(*components):
    component_args = dict(zip(index_related_component_keys, components))
    index_entry = components_to_index(**component_args)
    rag_client.add_index(index_entry)
    index_map = get_index_map()
    logger.info(f"Add index {index_entry.index_name} successfully")
    return [
        gr.update(
            choices=list(index_map.indexes.keys()) + ["NEW"],
            value=index_entry.index_name,
        ),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
    ]


def update_index(*components):
    component_args = dict(zip(index_related_component_keys, components))
    index_entry = components_to_index(**component_args)
    rag_client.update_index(index_entry)
    index_map = get_index_map()
    logger.info(f"Update index {index_entry.index_name} successfully")
    return [
        gr.update(
            choices=list(index_map.indexes.keys()) + ["NEW"],
            value=index_entry.index_name,
        ),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
    ]


def update_llms(selected_model):
    rag_config = rag_client.get_config()
    is_new = selected_model == "NEW"

    # Extract the relevant LLM configuration based on the selected model
    llm_config = next(
        (
            llm
            for llm in rag_config.llms
            if llm.model_id == selected_model
            or (not llm.model_id and llm.model == selected_model)
        ),
        None,
    )

    initial_values = {
        "base_url": llm_config.base_url if llm_config else "",
        "api_key": llm_config.api_key if llm_config else "",
        "model_name": llm_config.model if llm_config and not is_new else "",
        "model_id": llm_config.model_id if llm_config else "",
        "vision_support": llm_config.vision_support if llm_config else False,
    }

    # Update UI components based on the configuration
    return [
        gr.update(visible=True),
        gr.update(visible=not is_new),
        gr.update(value=initial_values["base_url"]),
        gr.update(value=initial_values["api_key"]),
        gr.update(value=initial_values["model_name"]),
        gr.update(value=initial_values["model_id"]),
        gr.update(value=initial_values["vision_support"]),
    ]


def save_new_llm(model_name, base_url, api_key, model_id, vision_support):
    rag_config = rag_client.get_config()
    if not all([base_url, api_key, model_name]):
        raise gr.Error("please fill in all fields")

    existing_model = next(
        (
            llm
            for llm in rag_config.llms
            if llm.model_id == model_id or (not llm.model_id and llm.model == model_id)
        ),
        None,
    )

    if existing_model:
        existing_model.base_url = base_url
        existing_model.api_key = api_key
        existing_model.model = model_name
        existing_model.model_id = model_id
        existing_model.vision_support = vision_support
    else:
        new_llm_config = {
            "source": "openai_compatible",
            "model_id": model_id,
            "base_url": base_url,
            "api_key": api_key,
            "model": model_name,
            "vision_support": vision_support,
        }
        new_llm = PaiBaseLlmConfig(**new_llm_config)

        rag_config.llms.append(new_llm)
        config = rag_service.get_config()
        update_dict = {}
        config["llms"].append(new_llm_config)
        update_dict["llms"] = config["llms"]
        rag_client.patch_config(update_dict)

    new_choices = ["NEW"] + [
        llm.model_id if llm.model_id else llm.model for llm in rag_config.llms
    ]
    return [
        gr.update(choices=new_choices, value=model_id or model_name),
        gr.update(visible=True),
    ]


def delete_llm(selected_model):
    rag_config = rag_client.get_config()
    # Find the LLM configuration by model_id
    rag_config.llms = [
        llm
        for llm in rag_config.llms
        if llm.model != selected_model and llm.model_id != selected_model
    ]

    config = rag_service.get_config()
    update_dict = {}
    update_dict["llms"] = [
        config_llm
        for config_llm in config["llms"]
        if config_llm.get("model") != selected_model
        and config_llm.get("model_id") != selected_model
    ]
    rag_client.patch_config(update_dict)

    new_choices = ["NEW"] + [
        llm.model_id if llm.model_id else llm.model for llm in rag_config.llms
    ]
    return [
        gr.update(choices=new_choices, value="NEW"),
        gr.update(visible=False),
        gr.update(visible=False),
    ]


def change_emb_source(source, model):
    if source.lower() == "huggingface" and not model:
        model = DEFAULT_HF_EMBED_MODEL
    return [
        gr.update(visible=(source.lower() == "huggingface"), value=model),
        EMBEDDING_DIM_DICT.get(source, DEFAULT_EMBED_SIZE)
        if source.lower() == "huggingface"
        else DEFAULT_EMBED_SIZE,
        gr.update(
            value=EMBEDDING_TYPE_DICT.get(model, "Default")
            if source.lower() == "huggingface"
            else "Default",
            visible=True if source.lower() == "huggingface" else False,
        ),
        gr.update(visible=True if source.lower() != "huggingface" else False),
    ]


def change_emb_model(source, model):
    if source.lower() == "huggingface" and not model:
        model = DEFAULT_HF_EMBED_MODEL

    return (
        EMBEDDING_DIM_DICT.get(model, DEFAULT_EMBED_SIZE)
        if source.lower() == "huggingface"
        else DEFAULT_EMBED_SIZE,
        EMBEDDING_TYPE_DICT.get(model, "Default")
        if source.lower() == "huggingface"
        else "Default",
    )


def change_use_oss(use_oss):
    if use_oss:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def change_enable_guardrail(enable_guardrail):
    if enable_guardrail:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def choose_use_mllm(value):
    if value:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def get_default_index_entry(index_map):
    index_name = f"INDEX_{len(index_map.indexes)}"
    return RagIndexEntry(
        index_name=index_name,
        embedding_config=HuggingFaceEmbeddingConfig(),
        vector_store_config=FaissVectorStoreConfig(
            persist_path=os.path.join(
                DEFAULT_KNOWLEDGE_PATH, index_name, ".index", ".faiss"
            )
        ),
    )


def change_vector_index(index_name):
    index_map = get_index_map()
    index_list = [index.index_name for index in index_map.indexes.values()]
    if index_name.lower() == "new":
        is_new = True
        index_entry = get_default_index_entry(index_map)
    else:
        is_new = False
        index_entry = index_map.indexes[index_name]
    return index_to_components(index_entry, index_list, is_new_index=is_new)


def change_vectordb_conn(vectordb_type):
    adb_visible = False
    hologres_visible = False
    faiss_visible = False
    es_visible = False
    milvus_visible = False
    opensearch_visible = False
    postgresql_visible = False
    tablestore_visible = False
    dashvector_visible = False
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
    elif vectordb_type.lower() == "tablestore":
        tablestore_visible = True
    elif vectordb_type.lower() == "dashvector":
        dashvector_visible = True
    return [
        gr.update(visible=adb_visible),
        gr.update(visible=hologres_visible),
        gr.update(visible=es_visible),
        gr.update(visible=faiss_visible),
        gr.update(visible=milvus_visible),
        gr.update(visible=opensearch_visible),
        gr.update(visible=postgresql_visible),
        gr.update(visible=tablestore_visible),
        gr.update(visible=dashvector_visible),
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
                value=f"[{datetime.datetime.now()}] Snapshot configuration saved successfully!",
                visible=True,
            ),
        ]
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")


def input_oss_ak_sk(input):
    return (input[:2] + "*" * (len(input) - 4) + input[-2:]) if input else input
