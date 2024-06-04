import gradio as gr
from typing import Any, Set, Callable, Dict
from pai_rag.app.web.view_model import view_model
from pai_rag.app.web.utils import components_to_dict


def create_vector_db_panel(
    input_elements: Set[Any],
    connect_vector_func: Callable[[Any], str],
) -> Dict[str, Any]:
    components = []
    with gr.Column():
        with gr.Column():
            _ = gr.Markdown(
                value=f"**Please check your Vector Store for {view_model.vectordb_type}.**"
            )
            vectordb_type = gr.Dropdown(
                ["Hologres", "Milvus", "ElasticSearch", "AnalyticDB", "FAISS"],
                label="Which VectorStore do you want to use?",
                elem_id="vectordb_type",
            )
            # Adb
            with gr.Column(
                visible=(view_model.vectordb_type == "AnalyticDB")
            ) as adb_col:
                adb_ak = gr.Textbox(
                    label="access-key-id",
                    type="password",
                    elem_id="adb_ak",
                )
                adb_sk = gr.Textbox(
                    label="access-key-secret",
                    type="password",
                    elem_id="adb_sk",
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
                )
                adb_account = gr.Textbox(label="Account", elem_id="adb_account")
                adb_account_password = gr.Textbox(
                    label="Password",
                    type="password",
                    elem_id="adb_account_password",
                )
                adb_namespace = gr.Textbox(
                    label="Namespace",
                    elem_id="adb_namespace",
                )
                adb_collection = gr.Textbox(
                    label="CollectionName",
                    elem_id="adb_collection",
                )

                connect_btn_adb = gr.Button("Connect AnalyticDB", variant="primary")
                con_state_adb = gr.Textbox(label="Connection Info: ")
                inputs_adb = input_elements.union(
                    {
                        vectordb_type,
                        adb_ak,
                        adb_sk,
                        adb_region_id,
                        adb_instance_id,
                        adb_account,
                        adb_account_password,
                        adb_namespace,
                        adb_collection,
                    }
                )
                connect_btn_adb.click(
                    fn=connect_vector_func,
                    inputs=inputs_adb,
                    outputs=con_state_adb,
                    api_name="connect_adb",
                )
            # Hologres
            with gr.Column(
                visible=(view_model.vectordb_type == "Hologres")
            ) as holo_col:
                hologres_host = gr.Textbox(
                    label="Host",
                    elem_id="hologres_host",
                )
                hologres_port = gr.Textbox(
                    label="Port",
                    elem_id="hologres_port",
                )
                hologres_database = gr.Textbox(
                    label="Database",
                    elem_id="hologres_database",
                )
                hologres_user = gr.Textbox(
                    label="User",
                    elem_id="hologres_user",
                )
                hologres_password = gr.Textbox(
                    label="Password",
                    type="password",
                    elem_id="hologres_password",
                )
                hologres_table = gr.Textbox(
                    label="Table",
                    elem_id="hologres_table",
                )
                hologres_pre_delete = gr.Dropdown(
                    ["True", "False"],
                    label="Pre Delete",
                    elem_id="hologres_pre_delete",
                )

                connect_btn_hologres = gr.Button("Connect Hologres", variant="primary")
                con_state_hologres = gr.Textbox(label="Connection Info: ")
                inputs_hologres = input_elements.union(
                    {
                        vectordb_type,
                        hologres_host,
                        hologres_user,
                        hologres_database,
                        hologres_password,
                        hologres_table,
                        hologres_pre_delete,
                    }
                )
                connect_btn_hologres.click(
                    fn=connect_vector_func,
                    inputs=inputs_hologres,
                    outputs=con_state_hologres,
                    api_name="connect_hologres",
                )

            with gr.Column(
                visible=(view_model.vectordb_type == "ElasticSearch")
            ) as es_col:
                es_url = gr.Textbox(label="ElasticSearch Url", elem_id="es_url")
                es_index = gr.Textbox(label="Index Name", elem_id="es_index")
                es_user = gr.Textbox(label="ES User", elem_id="es_user")
                es_password = gr.Textbox(
                    label="ES password",
                    type="password",
                    elem_id="es_password",
                )

                inputs_es = input_elements.union(
                    {vectordb_type, es_url, es_index, es_user, es_password}
                )
                connect_btn_es = gr.Button("Connect ElasticSearch", variant="primary")
                con_state_es = gr.Textbox(label="Connection Info: ")
                connect_btn_es.click(
                    fn=connect_vector_func,
                    inputs=inputs_es,
                    outputs=con_state_es,
                    api_name="connect_elasticsearch",
                )

            with gr.Column(
                visible=(view_model.vectordb_type == "Milvus")
            ) as milvus_col:
                milvus_host = gr.Textbox(label="Host", elem_id="milvus_host")
                milvus_port = gr.Textbox(label="Port", elem_id="milvus_port")

                milvus_user = gr.Textbox(label="User", elem_id="milvus_user")
                milvus_password = gr.Textbox(
                    label="Password",
                    type="password",
                    elem_id="milvus_password",
                )
                milvus_database = gr.Textbox(
                    label="Database",
                    elem_id="milvus_database",
                )
                milvus_collection_name = gr.Textbox(
                    label="Collection name",
                    elem_id="milvus_collection_name",
                )

                inputs_milvus = input_elements.union(
                    {
                        vectordb_type,
                        milvus_host,
                        milvus_port,
                        milvus_user,
                        milvus_password,
                        milvus_database,
                        milvus_collection_name,
                    }
                )
                connect_btn_milvus = gr.Button("Connect Milvus", variant="primary")
                con_state_milvus = gr.Textbox(label="Connection Info: ")
                connect_btn_milvus.click(
                    fn=connect_vector_func,
                    inputs=inputs_milvus,
                    outputs=con_state_milvus,
                    api_name="connect_milvus",
                )

            with gr.Column(visible=(view_model.vectordb_type == "FAISS")) as faiss_col:
                faiss_path = gr.Textbox(label="Path", elem_id="faiss_path")
                connect_btn_faiss = gr.Button("Connect Faiss", variant="primary")
                con_state_faiss = gr.Textbox(label="Connection Info: ")
                inputs_faiss = input_elements.union({vectordb_type, faiss_path})
                connect_btn_faiss.click(
                    fn=connect_vector_func,
                    inputs=inputs_faiss,
                    outputs=con_state_faiss,
                    api_name="connect_faiss",
                )

            def change_vectordb_conn(vectordb_type):
                adb_visible = False
                hologres_visible = False
                faiss_visible = False
                es_visible = False
                milvus_visible = False
                if vectordb_type == "AnalyticDB":
                    adb_visible = True
                elif vectordb_type == "Hologres":
                    hologres_visible = True
                elif vectordb_type == "ElasticSearch":
                    es_visible = True
                elif vectordb_type == "Milvus":
                    milvus_visible = True
                elif vectordb_type == "FAISS":
                    faiss_visible = True

                return {
                    adb_col: gr.update(visible=adb_visible),
                    holo_col: gr.update(visible=hologres_visible),
                    es_col: gr.update(visible=es_visible),
                    faiss_col: gr.update(visible=faiss_visible),
                    milvus_col: gr.update(visible=milvus_visible),
                }

            vectordb_type.change(
                fn=change_vectordb_conn,
                inputs=vectordb_type,
                outputs=[adb_col, holo_col, faiss_col, es_col, milvus_col],
            )

            components.extend(
                [
                    vectordb_type,
                    adb_ak,
                    adb_sk,
                    adb_region_id,
                    adb_instance_id,
                    adb_collection,
                    adb_account,
                    adb_account_password,
                    adb_namespace,
                    hologres_host,
                    hologres_port,
                    hologres_database,
                    hologres_user,
                    hologres_password,
                    hologres_table,
                    hologres_pre_delete,
                    milvus_host,
                    milvus_port,
                    milvus_database,
                    milvus_collection_name,
                    milvus_user,
                    milvus_password,
                    faiss_path,
                    es_url,
                    es_index,
                    es_user,
                    es_password,
                ]
            )

    return components_to_dict(components)
