import json
import gradio as gr
import os
from typing import Any, List
from pairag.web.index_utils import components_to_index, index_to_components
from pairag.web.rag_local_client import RagApiError, rag_client
from pairag.web.index_utils import index_related_component_keys
from pairag.web.tabs.model.index_info import get_index_map
import datetime
from pairag.web.ui_constants import (
    DEFAULT_EMBED_SIZE,
    DEFAULT_HF_EMBED_MODEL,
    EMBEDDING_DIM_DICT,
    EMBEDDING_TYPE_DICT,
)
from pairag.integrations.embeddings.pai.pai_embedding_config import (
    HuggingFaceEmbeddingConfig,
)
from pairag.knowledgebase.index.pai.vector_store_config import FaissVectorStoreConfig

from pairag.integrations.llms.pai.llm_config import (
    OpenAICompatibleLlmConfig,
    DashScopeGenerationModels,
    DASHSCOPE_MODEL_META,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_TOKENS,
)
from pairag.knowledgebase.rag_knowledgebase import KnowledgeBase
from pairag.utils.constants import DEFAULT_KNOWLEDGEBASE_PATH
from loguru import logger


def add_index(*components):
    component_args = dict(zip(index_related_component_keys, components))
    index_entry = components_to_index(**component_args)
    rag_client.add_index(index_entry)
    index_map = get_index_map()
    logger.info(f"Add index {index_entry.name} successfully")
    return [
        gr.update(
            choices=list(index_map.knowledgebases.keys()) + ["NEW"],
            value=index_entry.name,
        ),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
    ]


def update_index(*components):
    component_args = dict(zip(index_related_component_keys, components))
    index_entry = components_to_index(**component_args)
    rag_client.update_index(index_entry)
    index_map = get_index_map()
    logger.info(f"Update index {index_entry.name} successfully")
    gr.Success(f"Update index {index_entry.name} successfully.", duration=1)

    return [
        gr.update(
            choices=list(index_map.knowledgebases.keys()) + ["NEW"],
            value=index_entry.name,
        ),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
    ]


def delete_index(vector_index):
    try:
        rag_client.delete_index(vector_index)
        logger.info(f"Delete index {vector_index} successfully")
    except Exception as e:
        raise gr.Error(f"Failed to delete index: {e}")
    index_map = get_index_map()
    return [
        gr.update(
            choices=list(index_map.knowledgebases.keys()) + ["NEW"],
            value=list(index_map.knowledgebases.keys())[0],
        ),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
    ]


def fill_llm_tokens(model_name: str, selected_model_id: str):
    """
    如果是新增model，自动根据model_name填充tokens配置
    如果是已有model，不用
    """
    if selected_model_id == "NEW":
        model_name_lower_case = model_name.lower()
        for attr_name in dir(DashScopeGenerationModels):
            if not attr_name.startswith("__"):
                preserved_model_name = getattr(DashScopeGenerationModels, attr_name)
                if model_name_lower_case == preserved_model_name:
                    meta = DASHSCOPE_MODEL_META.get(model_name_lower_case)
                    if meta:
                        return meta["context_window"], meta["num_output"]
        return DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS

    else:
        rag_config = rag_client.get_config()
        # Extract the relevant LLM configuration based on the selected model
        llm_config = next(
            (
                llm
                for llm in rag_config.llms
                if llm.model_id == selected_model_id
                or (not llm.model_id and llm.model == selected_model_id)
            ),
            None,
        )
        if llm_config:
            return llm_config.context_window, llm_config.max_tokens
        else:
            # If the selected model is not found in the LLM configurations,
            # return the default values
            return DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS


def update_llms(selected_model_id):
    rag_config = rag_client.get_config()
    is_new = selected_model_id == "NEW"

    # Extract the relevant LLM configuration based on the selected model
    llm_config = next(
        (
            llm
            for llm in rag_config.llms
            if llm.model_id == selected_model_id
            or (not llm.model_id and llm.model == selected_model_id)
        ),
        None,
    )

    initial_values = {
        "base_url": llm_config.base_url if llm_config else "",
        "api_key": llm_config.api_key if llm_config else "",
        "model_name": llm_config.model if llm_config and not is_new else "",
        "model_id": llm_config.model_id if llm_config else "",
        "context_window": llm_config.context_window
        if llm_config
        else DEFAULT_CONTEXT_WINDOW,
        "max_tokens": llm_config.max_tokens if llm_config else DEFAULT_MAX_TOKENS,
        "vision_support": llm_config.vision_support if llm_config else False,
        "is_reasoning_model": llm_config.is_reasoning_model if llm_config else False,
        "extra_body_str": llm_config.extra_body_str if llm_config else "{}",
    }

    # Update UI components based on the configuration
    return [
        gr.update(visible=True),
        gr.update(visible=not is_new),
        gr.update(value=initial_values["base_url"]),
        gr.update(value=initial_values["api_key"]),
        gr.update(value=initial_values["model_name"]),
        gr.update(value=initial_values["model_id"]),
        gr.update(value=initial_values["context_window"]),
        gr.update(value=initial_values["max_tokens"]),
        gr.update(value=initial_values["vision_support"]),
        gr.update(value=initial_values["is_reasoning_model"]),
        gr.update(value=initial_values["extra_body_str"]),
    ]


def save_new_llm(
    selected_model_id,
    model_name,
    base_url,
    api_key,
    model_id,
    context_window,
    max_tokens,
    vision_support,
    is_reasoning_model,
    extra_body_str,
):
    try:
        extra_body_str = extra_body_str or "{}"
        _ = json.loads(extra_body_str)
    except json.JSONDecodeError:
        raise gr.Error(f"Invalid JSON format in extra_body '{extra_body_str}'.")

    if context_window <= max_tokens:
        raise ValueError("context_window should be greater than max_tokens")

    rag_config = rag_client.get_config()
    is_new = selected_model_id == "NEW"
    if not all([base_url, api_key, model_name]):
        raise gr.Error("please fill in all fields")

    if is_new:
        model_index, existing_model = next(
            (
                (index, llm)
                for index, llm in enumerate(rag_config.llms)
                if llm.model_id == model_id
                or (not llm.model_id and llm.model == model_id)
            ),
            (-1, None),
        )
    else:
        model_index, existing_model = next(
            (
                (index, llm)
                for index, llm in enumerate(rag_config.llms)
                if llm.model_id == selected_model_id
                or (not llm.model_id and llm.model == selected_model_id)
            ),
            (-1, None),
        )

    if existing_model:
        existing_model.base_url = base_url
        existing_model.api_key = api_key
        existing_model.model = model_name
        existing_model.model_id = model_id
        existing_model.context_window = context_window
        existing_model.max_tokens = max_tokens
        existing_model.vision_support = vision_support
        existing_model.is_reasoning_model = is_reasoning_model
        existing_model.extra_body_str = extra_body_str
        rag_config.llms[model_index] = existing_model

    else:
        new_llm_config = {
            "source": "openai_compatible",
            "model_id": model_id,
            "base_url": base_url,
            "api_key": api_key,
            "model": model_name,
            "context_window": context_window,
            "max_tokens": max_tokens,
            "vision_support": vision_support,
            "is_reasoning_model": is_reasoning_model,
            "extra_body_str": extra_body_str,
        }
        new_llm = OpenAICompatibleLlmConfig(**new_llm_config)

        rag_config.llms.append(new_llm)

    update_dict = {}
    update_dict["llms"] = rag_config.llms
    rag_client.patch_config(update_dict)

    new_choices = [
        llm.model_id if llm.model_id else llm.model for llm in rag_config.llms
    ] + ["NEW"]
    return [
        gr.update(choices=new_choices, value=model_id or model_name),
        gr.update(visible=True),
    ]


def delete_llm(selected_model_id):
    rag_config = rag_client.get_config()
    # Find the LLM configuration by model_id
    rag_config.llms = [
        llm
        for llm in rag_config.llms
        if llm.model != selected_model_id and llm.model_id != selected_model_id
    ]

    update_dict = {}
    update_dict["llms"] = rag_config.llms
    rag_client.patch_config(update_dict)

    new_choices = [
        llm.model_id if llm.model_id else llm.model for llm in rag_config.llms
    ] + ["NEW"]
    return [
        gr.update(choices=new_choices, value=new_choices[0]),
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
    index_name = f"INDEX_{len(index_map.knowledgebases)}"
    return KnowledgeBase(
        name=index_name,
        embedding_config=HuggingFaceEmbeddingConfig(),
        vector_store_config=FaissVectorStoreConfig(
            persist_path=os.path.join(
                DEFAULT_KNOWLEDGEBASE_PATH, index_name, ".index", ".faiss"
            )
        ),
    )


def change_vector_index(index_name):
    index_map = get_index_map()
    index_list = [index.name for index in index_map.knowledgebases.values()]
    if index_name.lower() == "new":
        is_new = True
        index_entry = get_default_index_entry(index_map)
    else:
        is_new = False
        index_entry = index_map.knowledgebases[index_name]
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
                value=f"[{datetime.datetime.now()}] OSS Snapshot configuration saved successfully!",
                visible=True,
            ),
        ]
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")


def input_oss_ak_sk(input):
    return (input[:2] + "*" * (len(input) - 4) + input[-2:]) if input else input


def save_pmt_cfg_func(input_elements: List[Any]):
    try:
        update_dict = {}
        for element, value in input_elements.items():
            update_dict[element.elem_id] = value
        rag_client.patch_config(update_dict)

        return gr.update(
            value=f"[{datetime.datetime.now()}] Prompt configuration saved successfully!",
            visible=True,
        )
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")


def save_query_transform_cfg(input_elements: List[Any]):
    try:
        update_dict = {}
        for element, value in input_elements.items():
            update_dict[element.elem_id] = value
        rag_client.patch_config(update_dict)

        return gr.update(
            value=f"[{datetime.datetime.now()}] Query transform prompt configuration saved successfully!",
            visible=True,
        )
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")


def save_trace_cfg(input_elements: List[Any]):
    try:
        update_dict = {}
        for element, value in input_elements.items():
            update_dict[element.elem_id] = value
        rag_client.patch_config(update_dict)

        return gr.update(
            value=f"[{datetime.datetime.now()}] 成功保存OpenTelemetry配置信息!",
            visible=True,
        )
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")
