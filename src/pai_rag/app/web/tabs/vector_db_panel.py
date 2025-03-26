import gradio as gr
from typing import Any, Dict
from pai_rag.app.web.utils import components_to_dict
import pai_rag.app.web.event_listeners as ev_listeners


def create_vector_db_panel() -> Dict[str, Any]:
    components = []
    with gr.Row():
        with gr.Column():
            _ = gr.Markdown(value="### **向量数据库配置**")
            vectordb_type = gr.Radio(
                [
                    "hologres",
                    "milvus",
                    "elasticsearch",
                    "faiss",
                    "opensearch",
                    "postgresql",
                    "tablestore",
                    "dashvector",
                ],
                label="请配置您的向量数据库连接信息",
                elem_id="vectordb_type",
                interactive=True,
            )
            # Adb
            with gr.Column(visible=(vectordb_type == "analyticdb")) as adb_col:
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
                    value="cn-hangzhou",
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
            with gr.Column(visible=(vectordb_type == "hologres")) as holo_col:
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

            with gr.Column(visible=(vectordb_type == "elasticsearch")) as es_col:
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

            with gr.Column(visible=(vectordb_type == "milvus")) as milvus_col:
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

            with gr.Column(visible=(vectordb_type == "faiss")) as faiss_col:
                faiss_path = gr.Textbox(
                    label="Path",
                    elem_id="faiss_path",
                    interactive=True,
                    visible=False,
                )

            with gr.Column(visible=(vectordb_type == "opensearch")) as opensearch_col:
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

            with gr.Column(visible=(vectordb_type == "postgresql")) as postgresql_col:
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

            with gr.Column(visible=(vectordb_type == "tablestore")) as tablestore_col:
                with gr.Row():
                    tablestore_endpoint = gr.Textbox(
                        label="tablestore_endpoint",
                        elem_id="tablestore_endpoint",
                        interactive=True,
                    )
                    tablestore_instance_name = gr.Textbox(
                        label="tablestore_instance_name",
                        elem_id="tablestore_instance_name",
                        interactive=True,
                    )
                with gr.Row():
                    tablestore_access_key_id = gr.Textbox(
                        label="tablestore_access_key_id",
                        elem_id="tablestore_access_key_id",
                        interactive=True,
                    )
                    tablestore_access_key_secret = gr.Textbox(
                        label="tablestore_access_key_secret",
                        type="password",
                        elem_id="tablestore_access_key_secret",
                        interactive=True,
                    )
                with gr.Row():
                    tablestore_table_name = gr.Textbox(
                        label="tablestore_table_name",
                        elem_id="tablestore_table_name",
                        interactive=True,
                    )

            with gr.Column(visible=(vectordb_type == "dashvector")) as dashvector_col:
                with gr.Row():
                    dashvector_endpoint = gr.Textbox(
                        label="Endpoint",
                        elem_id="dashvector_endpoint",
                        interactive=True,
                    )
                    dashvector_api_key = gr.Textbox(
                        label="ApiKey",
                        elem_id="dashvector_api_key",
                        interactive=True,
                        type="password",
                    )
                with gr.Row():
                    dashvector_collection_name = gr.Textbox(
                        label="Collection Name",
                        elem_id="dashvector_collection_name",
                        interactive=True,
                        placeholder="pai_rag",
                        info="leave it empty to use the default collection name: 'pai_rag'",
                    )
                    dashvector_partition_name = gr.Textbox(
                        label="Partition Name",
                        elem_id="dashvector_partition_name",
                        interactive=True,
                        placeholder="default",
                        info="leave it empty to specify the default partition: 'default'",
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
                    tablestore_col,
                    dashvector_col,
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
                # tablestore
                tablestore_endpoint,
                tablestore_instance_name,
                tablestore_access_key_id,
                tablestore_access_key_secret,
                tablestore_table_name,
                # dashvector
                dashvector_endpoint,
                dashvector_api_key,
                dashvector_collection_name,
                dashvector_partition_name,
            ]
            components.extend(db_related_elements)
        with gr.Column(visible=True):
            _ = gr.Markdown(value="### **检索参数设置**")
            retrieval_mode = gr.Radio(
                ["向量检索", "关键字检索", "混合检索"],
                label="检索模式",
                elem_id="retrieval_mode",
            )

            vector_weight = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.7,
                elem_id="vector_weight",
                label="向量检索权重",
                visible=(retrieval_mode == "混合检索"),
            )
            keyword_weight = gr.Slider(
                minimum=0,
                maximum=1,
                value=float(1 - vector_weight.value),
                elem_id="keyword_weight",
                label="关键字检索权重",
                interactive=False,
                visible=(retrieval_mode == "混合检索"),
            )

            similarity_top_k = gr.Slider(
                minimum=0,
                maximum=100,
                step=1,
                elem_id="similarity_top_k",
                label="返回Top-K条文本结果 (0 到 100)",
            )
            image_similarity_top_k = gr.Slider(
                minimum=0,
                maximum=10,
                step=1,
                elem_id="image_similarity_top_k",
                label="返回Top-K条图片结果 (0 到 10)",
            )
            similarity_threshold = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.01,
                elem_id="similarity_threshold",
                label="相似度分数阈值 (内容越相似，分数越大)",
            )

            reranker_type = gr.Radio(
                ["无重排序", "基于模型的重排序"],
                label="重排序类型",
                elem_id="reranker_type",
            )
            with gr.Column(
                visible=(reranker_type == "基于模型的重排序"),
                elem_id="model_reranker_col",
            ) as model_reranker_col:
                reranker_model = gr.Radio(
                    [
                        "bge-reranker-base",
                        "bge-reranker-large",
                    ],
                    label="重排序模型（注意：首次使用该模型时，加载模型将需要较长时间）",
                    elem_id="reranker_model",
                )
                reranker_similarity_threshold = gr.Slider(
                    minimum=-10,
                    maximum=10,
                    step=0.01,
                    elem_id="reranker_similarity_threshold",
                    label="重排序相似度分数阈值（结果越相似，数值越大）",
                )
                reranker_similarity_top_k = gr.Slider(
                    minimum=0,
                    maximum=50,
                    step=1,
                    elem_id="reranker_similarity_top_k",
                    label="重排序文本 Top-K (0 到 50)",
                )

            def change_weight(change_weight):
                return round(float(1 - change_weight), 2)

            vector_weight.input(
                fn=change_weight,
                inputs=vector_weight,
                outputs=[keyword_weight],
            )

            def change_reranker_type(reranker_type):
                if reranker_type == "无重排序":
                    return {
                        model_reranker_col: gr.update(visible=False),
                    }
                elif reranker_type == "基于模型的重排序":
                    return {
                        model_reranker_col: gr.update(visible=True),
                    }
                else:
                    return {
                        model_reranker_col: gr.update(visible=False),
                    }

            def change_retrieval_mode(retrieval_mode):
                if retrieval_mode == "混合检索":
                    return {
                        vector_weight: gr.update(visible=True),
                        keyword_weight: gr.update(visible=True),
                    }
                else:
                    return {
                        vector_weight: gr.update(visible=False),
                        keyword_weight: gr.update(visible=False),
                    }

            reranker_type.input(
                fn=change_reranker_type,
                inputs=reranker_type,
                outputs=[model_reranker_col],
            )

            retrieval_mode.input(
                fn=change_retrieval_mode,
                inputs=retrieval_mode,
                outputs=[vector_weight, keyword_weight],
            )

            db_retrieval_elements = [
                retrieval_mode,
                reranker_type,
                vector_weight,
                keyword_weight,
                similarity_top_k,
                image_similarity_top_k,
                similarity_threshold,
                reranker_similarity_threshold,
                reranker_model,
                reranker_similarity_top_k,
            ]
            components.extend(db_retrieval_elements)
    return db_related_elements, components_to_dict(components)
