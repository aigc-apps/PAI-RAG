import os
from typing import List
import gradio as gr

from pai_rag.app.web.ui_constants import (
    DEFAULT_EMBED_SIZE,
    EMBEDDING_DIM_DICT,
    EMBEDDING_MODEL_DEPRECATED,
    EMBEDDING_MODEL_LIST,
    EMBEDDING_TYPE_DICT,
    EMBEDDING_MODEL_LINK_DICT,
)
from pai_rag.core.rag_index_manager import RagIndexEntry
from pai_rag.integrations.index.pai.vector_store_config import (
    DEFAULT_LOCAL_STORAGE_PATH,
    AnalyticDBVectorStoreConfig,
    ElasticSearchVectorStoreConfig,
    FaissVectorStoreConfig,
    HologresVectorStoreConfig,
    MilvusVectorStoreConfig,
    OpenSearchVectorStoreConfig,
    PostgreSQLVectorStoreConfig,
)


index_related_component_keys = [
    "vector_index",
    "new_index_name",
    "add_index_button",
    "update_index_button",
    "delete_index_button",
    "embed_source",
    "embed_model",
    "embed_dim",
    "embed_type",
    "embed_link",
    "embed_batch_size",
    "vectordb_type",
    "faiss_path",
    "adb_ak",
    "adb_sk",
    "adb_account",
    "adb_region_id",
    "adb_instance_id",
    "adb_namespace",
    "adb_collection",
    "adb_account_password",
    "es_url",
    "es_user",
    "es_password",
    "es_index",
    "milvus_host",
    "milvus_port",
    "milvus_collection_name",
    "milvus_user",
    "milvus_database",
    "milvus_password",
    "opensearch_endpoint",
    "opensearch_instance_id",
    "opensearch_username",
    "opensearch_password",
    "opensearch_table_name",
    "hologres_host",
    "hologres_port",
    "hologres_database",
    "hologres_user",
    "hologres_password",
    "hologres_table",
    "hologres_pre_delete",
    "postgresql_host",
    "postgresql_port",
    "postgresql_database",
    "postgresql_username",
    "postgresql_password",
    "postgresql_table_name",
]


def index_to_components_settings(
    index_entry: RagIndexEntry, index_list: List[str], is_new_index: bool = False
):
    if is_new_index:
        index_component_settings = [
            {"value": "NEW", "choices": index_list + ["NEW"]},
            {"placeholder": index_entry.index_name, "visible": True},
            {"visible": True},
            {"visible": False},
            {"visible": False},
        ]
    else:
        index_component_settings = [
            {"value": index_entry.index_name, "choices": index_list + ["NEW"]},
            {"placeholder": "", "visible": False},
            {"visible": False},
            {"visible": True},
            {"visible": False},
        ]

    embed_source = index_entry.embedding_config.source.value
    embed_model = index_entry.embedding_config.model_name
    if (embed_model in EMBEDDING_MODEL_DEPRECATED) or os.getenv(
        "USE_DEPRECATED_EMBEDDING_MODEL", "False"
    ):
        embed_model_setting = {
            "value": embed_model,
            "choices": EMBEDDING_MODEL_LIST + EMBEDDING_MODEL_DEPRECATED,
            "visible": embed_source == "huggingface",
        }
    else:
        embed_model_setting = {
            "value": embed_model,
            "choices": EMBEDDING_MODEL_LIST + EMBEDDING_MODEL_DEPRECATED,
            "visible": embed_source == "huggingface",
        }

    embed_dim_setting = {
        "value": EMBEDDING_DIM_DICT.get(embed_model, DEFAULT_EMBED_SIZE)
        if embed_source == "huggingface"
        else DEFAULT_EMBED_SIZE
    }
    embed_type_setting = {
        "value": EMBEDDING_TYPE_DICT.get(embed_model, "Default")
        if embed_source == "huggingface"
        else "Default"
    }
    embed_link_setting = {
        "value": f"Model Introduction: [{embed_model}]({EMBEDDING_MODEL_LINK_DICT[embed_model]})"
        if embed_source == "huggingface"
        else ""
    }
    embed_batch_size_setting = {"value": index_entry.embedding_config.embed_batch_size}

    embed_component_settings = [
        {"value": embed_source},
        embed_model_setting,
        embed_dim_setting,
        embed_type_setting,
        embed_link_setting,
        embed_batch_size_setting,
    ]

    vector_store_config = index_entry.vector_store_config

    vector_component_settings = [{"value": vector_store_config.type.value}]

    if isinstance(vector_store_config, FaissVectorStoreConfig):
        vector_component_settings.append({"value": vector_store_config.persist_path})
    else:
        vector_component_settings.append({"value": DEFAULT_LOCAL_STORAGE_PATH})

    if isinstance(vector_store_config, AnalyticDBVectorStoreConfig):
        vector_component_settings.extend(
            [
                {"value": vector_store_config.ak},
                {"value": vector_store_config.sk},
                {"value": vector_store_config.account},
                {"value": vector_store_config.region_id},
                {"value": vector_store_config.instance_id},
                {"value": vector_store_config.namespace},
                {"value": vector_store_config.collection},
                {"value": vector_store_config.account_password},
            ]
        )
    else:
        vector_component_settings.extend(
            [
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
            ]
        )

    if isinstance(vector_store_config, ElasticSearchVectorStoreConfig):
        vector_component_settings.extend(
            [
                {"value": vector_store_config.es_url},
                {"value": vector_store_config.es_user},
                {"value": vector_store_config.es_password},
                {"value": vector_store_config.es_index},
            ]
        )
    else:
        vector_component_settings.extend(
            [
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
            ]
        )

    if isinstance(vector_store_config, MilvusVectorStoreConfig):
        vector_component_settings.extend(
            [
                {"value": vector_store_config.host},
                {"value": vector_store_config.port},
                {"value": vector_store_config.collection_name},
                {"value": vector_store_config.user},
                {"value": vector_store_config.database},
                {"value": vector_store_config.password},
            ]
        )
    else:
        vector_component_settings.extend(
            [
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
            ]
        )

    if isinstance(vector_store_config, OpenSearchVectorStoreConfig):
        vector_component_settings.extend(
            [
                {"value": vector_store_config.endpoint},
                {"value": vector_store_config.instance_id},
                {"value": vector_store_config.username},
                {"value": vector_store_config.password},
                {"value": vector_store_config.table_name},
            ]
        )
    else:
        vector_component_settings.extend(
            [
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
            ]
        )

    if isinstance(vector_store_config, HologresVectorStoreConfig):
        vector_component_settings.extend(
            [
                {"value": vector_store_config.host},
                {"value": vector_store_config.port},
                {"value": vector_store_config.database},
                {"value": vector_store_config.user},
                {"value": vector_store_config.password},
                {"value": vector_store_config.table_name},
                {"value": vector_store_config.pre_delete_table},
            ]
        )
    else:
        vector_component_settings.extend(
            [
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
            ]
        )

    if isinstance(vector_store_config, PostgreSQLVectorStoreConfig):
        vector_component_settings.extend(
            [
                {"value": vector_store_config.host},
                {"value": vector_store_config.port},
                {"value": vector_store_config.database},
                {"value": vector_store_config.username},
                {"value": vector_store_config.password},
                {"value": vector_store_config.table_name},
            ]
        )
    else:
        vector_component_settings.extend(
            [
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
                {"value": ""},
            ]
        )
    component_settings = [
        *index_component_settings,
        *embed_component_settings,
        *vector_component_settings,
    ]

    settings = dict(zip(index_related_component_keys, component_settings))
    return settings


def index_to_components(
    index_entry: RagIndexEntry, index_list: List[str], is_new_index: bool = False
):
    component_settings = index_to_components_settings(
        index_entry, index_list, is_new_index
    )
    print("+++", index_entry.index_name)
    return [gr.update(**setting) for setting in component_settings.values()] + [
        gr.update(choices=index_list, value=index_entry.index_name),
        gr.update(choices=index_list, value=index_entry.index_name),
    ]


def components_to_index(
    vector_index,
    new_index_name,
    embed_source,
    embed_model,
    embed_batch_size,
    vectordb_type,
    hologres_host,
    hologres_port,
    hologres_user,
    hologres_password,
    hologres_database,
    hologres_table,
    hologres_pre_delete,
    faiss_path,
    opensearch_endpoint,
    opensearch_username,
    opensearch_password,
    opensearch_instance_id,
    opensearch_table_name,
    postgresql_host,
    postgresql_port,
    postgresql_database,
    postgresql_username,
    postgresql_password,
    postgresql_table_name,
    adb_ak,
    adb_sk,
    adb_region_id,
    adb_instance_id,
    adb_account,
    adb_account_password,
    adb_namespace,
    adb_collection,
    es_index,
    es_url,
    es_user,
    es_password,
    milvus_host,
    milvus_port,
    milvus_user,
    milvus_password,
    milvus_database,
    milvus_collection_name,
    **kwargs,
) -> RagIndexEntry:
    if vector_index.lower() == "new":
        index_name = new_index_name
    else:
        index_name = vector_index

    embedding = {
        "source": embed_source,
        "model_name": embed_model,
        "embed_batch_size": int(embed_batch_size),
    }

    if vectordb_type.lower() == "hologres":
        vector_store = {
            "type": vectordb_type.lower(),
            "host": hologres_host,
            "port": hologres_port,
            "user": hologres_user,
            "password": hologres_password,
            "database": hologres_database,
            "table_name": hologres_table,
            "pre_delete_table": hologres_pre_delete,
        }
    elif vectordb_type.lower() == "faiss":
        vector_store = {
            "type": vectordb_type.lower(),
            "persist_path": faiss_path,
        }
    elif vectordb_type.lower() == "analyticdb":
        vector_store = {
            "type": vectordb_type.lower(),
            "ak": adb_ak,
            "sk": adb_sk,
            "region_id": adb_region_id,
            "instance_id": adb_instance_id,
            "account": adb_account,
            "account_password": adb_account_password,
            "namespace": adb_namespace,
            "collection": adb_collection,
        }

    elif vectordb_type.lower() == "elasticsearch":
        vector_store = {
            "type": vectordb_type.lower(),
            "es_url": es_url,
            "es_user": es_user,
            "es_password": es_password,
            "es_index": es_index,
        }

    elif vectordb_type.lower() == "milvus":
        vector_store = {
            "type": vectordb_type.lower(),
            "host": milvus_host,
            "port": milvus_port,
            "user": milvus_user,
            "password": milvus_password,
            "database": milvus_database,
            "collection_name": milvus_collection_name,
        }

    elif vectordb_type.lower() == "opensearch":
        vector_store = {
            "type": vectordb_type.lower(),
            "endpoint": opensearch_endpoint,
            "instance_id": opensearch_instance_id,
            "username": opensearch_username,
            "password": opensearch_password,
            "table_name": opensearch_table_name,
        }

    elif vectordb_type.lower() == "postgresql":
        vector_store = {
            "type": vectordb_type.lower(),
            "host": postgresql_host,
            "port": postgresql_port,
            "database": postgresql_database,
            "table_name": postgresql_table_name,
            "username": postgresql_username,
            "password": postgresql_password,
        }
    else:
        raise ValueError(f"Unknown vector db type: {vectordb_type}")

    index_entry = RagIndexEntry.model_validate(
        {
            "index_name": index_name,
            "vector_store_config": vector_store,
            "embedding_config": embedding,
        }
    )

    return index_entry
