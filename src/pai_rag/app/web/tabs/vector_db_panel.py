import gradio as gr
from typing import Any, Set, Dict
from pai_rag.app.web.utils import components_to_dict
import os
import pai_rag.app.web.event_listeners as ev_listeners

DEFAULT_IS_INTERACTIVE = os.environ.get("PAIRAG_RAG__SETTING__interactive", "true")


def create_vector_db_panel(
    input_elements: Set[Any],
) -> Dict[str, Any]:
    components = []
    with gr.Column():
        with gr.Column():
            _ = gr.Markdown(value="\N{WHITE MEDIUM STAR} **Vector Store**")
            vectordb_type = gr.Radio(
                [
                    "FAISS",
                    "ElasticSearch",
                    "Milvus",
                    "Hologres",
                    "OpenSearch",
                    "PostgreSQL",
                ],
                label="Which VectorStore do you want to use?",
                elem_id="vectordb_type",
                interactive=DEFAULT_IS_INTERACTIVE.lower() != "false",
            )
            # Adb
            with gr.Column(visible=(vectordb_type == "AnalyticDB")) as adb_col:
                adb_ak = gr.Textbox(
                    label="access-key-id",
                    type="password",
                    elem_id="adb_ak",
                    interactive=True,
                )
                adb_sk = gr.Textbox(
                    label="access-key-secret",
                    type="password",
                    elem_id="adb_sk",
                    interactive=True,
                )
                adb_region_id = gr.Dropdown(
                    [
                        "cn-hangzhou",
                        "cn-beijing",
                        "cn-zhangjiakou",
                        "cn-huhehaote",
                        "cn-shanghai",
                        "cn-shenzhen",
                        "cn-chengdu",
                    ],
                    label="RegionId",
                    elem_id="adb_region_id",
                )
                adb_instance_id = gr.Textbox(
                    label="InstanceId",
                    elem_id="adb_instance_id",
                    interactive=True,
                )
                adb_account = gr.Textbox(
                    label="Account",
                    elem_id="adb_account",
                    interactive=True,
                )
                adb_account_password = gr.Textbox(
                    label="Password",
                    type="password",
                    elem_id="adb_account_password",
                    interactive=True,
                )
                adb_namespace = gr.Textbox(
                    label="Namespace",
                    elem_id="adb_namespace",
                    interactive=True,
                )
                adb_collection = gr.Textbox(
                    label="CollectionName",
                    elem_id="adb_collection",
                    interactive=True,
                )

            # Hologres
            with gr.Column(visible=(vectordb_type == "Hologres")) as holo_col:
                with gr.Row():
                    hologres_host = gr.Textbox(
                        label="Host",
                        elem_id="hologres_host",
                        interactive=True,
                    )
                    hologres_port = gr.Textbox(
                        label="Port",
                        elem_id="hologres_port",
                        interactive=True,
                    )
                with gr.Row():
                    hologres_user = gr.Textbox(
                        label="User",
                        elem_id="hologres_user",
                        interactive=True,
                    )
                    hologres_password = gr.Textbox(
                        label="Password",
                        type="password",
                        elem_id="hologres_password",
                        interactive=True,
                    )
                with gr.Row():
                    hologres_database = gr.Textbox(
                        label="Database",
                        elem_id="hologres_database",
                        interactive=True,
                    )
                    hologres_table = gr.Textbox(
                        label="Table",
                        elem_id="hologres_table",
                        interactive=True,
                    )
                hologres_pre_delete = gr.Checkbox(
                    label="Yes",
                    info="Clear hologres table on connection.",
                    elem_id="hologres_pre_delete",
                )

            with gr.Column(visible=(vectordb_type == "ElasticSearch")) as es_col:
                with gr.Row():
                    es_url = gr.Textbox(
                        label="ElasticSearch Url", elem_id="es_url", interactive=True
                    )
                    es_index = gr.Textbox(
                        label="Index Name", elem_id="es_index", interactive=True
                    )
                with gr.Row():
                    es_user = gr.Textbox(
                        label="ES User", elem_id="es_user", interactive=True
                    )
                    es_password = gr.Textbox(
                        label="ES password",
                        type="password",
                        elem_id="es_password",
                        interactive=True,
                    )

            with gr.Column(visible=(vectordb_type == "Milvus")) as milvus_col:
                with gr.Row():
                    milvus_host = gr.Textbox(
                        label="Host", elem_id="milvus_host", interactive=True
                    )
                    milvus_port = gr.Textbox(
                        label="Port", elem_id="milvus_port", interactive=True
                    )
                with gr.Row():
                    milvus_user = gr.Textbox(
                        label="User", elem_id="milvus_user", interactive=True
                    )
                    milvus_password = gr.Textbox(
                        label="Password",
                        type="password",
                        elem_id="milvus_password",
                        interactive=True,
                    )
                with gr.Row():
                    milvus_database = gr.Textbox(
                        label="Database",
                        elem_id="milvus_database",
                        interactive=True,
                    )
                    milvus_collection_name = gr.Textbox(
                        label="Collection name",
                        elem_id="milvus_collection_name",
                        interactive=True,
                    )

            with gr.Column(visible=(vectordb_type == "FAISS")) as faiss_col:
                faiss_path = gr.Textbox(
                    label="Path", elem_id="faiss_path", interactive=True
                )

            with gr.Column(visible=(vectordb_type == "OpenSearch")) as opensearch_col:
                with gr.Row():
                    opensearch_endpoint = gr.Textbox(
                        label="Endpoint",
                        elem_id="opensearch_endpoint",
                        interactive=True,
                    )
                    opensearch_instance_id = gr.Textbox(
                        label="InstanceId",
                        elem_id="opensearch_instance_id",
                        interactive=True,
                    )
                with gr.Row():
                    opensearch_username = gr.Textbox(
                        label="UserName",
                        elem_id="opensearch_username",
                        interactive=True,
                    )
                    opensearch_password = gr.Textbox(
                        label="Password",
                        type="password",
                        elem_id="opensearch_password",
                        interactive=True,
                    )
                opensearch_table_name = gr.Textbox(
                    label="TableName", elem_id="opensearch_table_name", interactive=True
                )

            with gr.Column(visible=(vectordb_type == "PostgreSQL")) as postgresql_col:
                with gr.Row():
                    postgresql_host = gr.Textbox(
                        label="Host", elem_id="postgresql_host", interactive=True
                    )
                    postgresql_port = gr.Textbox(
                        label="Port", elem_id="postgresql_port", interactive=True
                    )
                with gr.Row():
                    postgresql_username = gr.Textbox(
                        label="UserName",
                        elem_id="postgresql_username",
                        interactive=True,
                    )
                    postgresql_password = gr.Textbox(
                        label="Password",
                        type="password",
                        elem_id="postgresql_password",
                        interactive=True,
                    )
                with gr.Row():
                    postgresql_database = gr.Textbox(
                        label="Database",
                        elem_id="postgresql_database",
                        interactive=True,
                    )
                    postgresql_table_name = gr.Textbox(
                        label="TableName",
                        elem_id="postgresql_table_name",
                        interactive=True,
                    )

            vectordb_type.change(
                fn=ev_listeners.change_vectordb_conn,
                inputs=vectordb_type,
                outputs=[
                    adb_col,
                    holo_col,
                    es_col,
                    faiss_col,
                    milvus_col,
                    opensearch_col,
                    postgresql_col,
                ],
            )
            db_related_elements = [
                vectordb_type,
                # faiss
                faiss_path,
                # hologres
                hologres_host,
                hologres_port,
                hologres_user,
                hologres_database,
                hologres_password,
                hologres_table,
                hologres_pre_delete,
                # elasticsearch
                es_url,
                es_index,
                es_user,
                es_password,
                # milvus
                milvus_host,
                milvus_port,
                milvus_user,
                milvus_password,
                milvus_database,
                milvus_collection_name,
                # opensearch
                opensearch_endpoint,
                opensearch_instance_id,
                opensearch_username,
                opensearch_password,
                opensearch_table_name,
                # postgresql
                postgresql_host,
                postgresql_port,
                postgresql_database,
                postgresql_table_name,
                postgresql_username,
                postgresql_password,
                # analytic db
                adb_ak,
                adb_sk,
                adb_region_id,
                adb_instance_id,
                adb_collection,
                adb_account,
                adb_account_password,
                adb_namespace,
            ]
            union_input_elements = input_elements.union(db_related_elements)
            components.extend(db_related_elements)
    return union_input_elements, components_to_dict(components)
