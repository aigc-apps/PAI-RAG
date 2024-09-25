import os
from typing import Dict, Any, List
import gradio as gr
from pai_rag.app.web.rag_client import RagApiError, rag_client
import tempfile
import json
import pandas as pd
from datetime import datetime
from pai_rag.app.web.tabs.upload_tab import upload_knowledge, clear_files


def generate_and_download_qa_file(overwrite, eval_exp_id):
    tmpdir = tempfile.mkdtemp()
    try:
        qa_content = rag_client.evaluate_for_generate_qa(bool(overwrite), eval_exp_id)
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")

    outputPath = os.path.join(tmpdir, "qa_dataset_output.json")
    with open(outputPath, "w", encoding="utf-8") as f:
        json.dump(qa_content, f, ensure_ascii=False, indent=4)
    return outputPath, qa_content["result"]["examples"][0:2]


def eval_retrieval_stage(input_elements: List[Any]):
    update_dict = {}
    eval_exp_id = ""
    for element, value in input_elements.items():
        if element.elem_id == "eval_retrieval_mode":
            update_dict["retrieval_mode"] = value
        if element.elem_id == "eval_reranker_type":
            update_dict["reranker_type"] = value
        if element.elem_id == "eval_exp_id":
            eval_exp_id = value
        update_dict[element.elem_id] = value

    try:
        rag_client.patch_config(update_dict)
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")

    try:
        retrieval_res = rag_client.evaluate_for_retrieval_stage(eval_exp_id)
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")

    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pd_results = {
        "Metrics": ["HitRate", "MRR", "LastModified"],
        "Value": [
            retrieval_res["result"]["hit_rate_mean"],
            retrieval_res["result"]["mrr_mean"],
            formatted_time,
        ],
    }
    return pd.DataFrame(pd_results)


def eval_response_stage(input_elements: List[Any]):
    update_dict = {}
    eval_exp_id = ""
    for element, value in input_elements.items():
        if element.elem_id == "eval_retrieval_mode":
            update_dict["retrieval_mode"] = value
        if element.elem_id == "eval_reranker_type":
            update_dict["reranker_type"] = value
        if element.elem_id == "eval_exp_id":
            eval_exp_id = value
        update_dict[element.elem_id] = value

    try:
        rag_client.patch_config(update_dict)
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")

    try:
        response_res = rag_client.evaluate_for_response_stage(eval_exp_id)
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pd_results = {
        "Metrics": ["Faithfulness", "Correctness", "Similarity", "LastModified"],
        "Value": [
            response_res["result"]["faithfulness_mean"],
            response_res["result"]["correctness_mean"],
            response_res["result"]["similarity_mean"],
            formatted_time,
        ],
    }
    return pd.DataFrame(pd_results)


def change_weight(change_weight):
    return round(float(1 - change_weight), 2)


def change_reranker_type(retrieval_mode, reranker_type):
    if retrieval_mode == "Hybrid" and reranker_type == "simple-weighted-reranker":
        return [gr.update(visible=True), gr.update(visible=False)]
    elif reranker_type == "model-based-reranker":
        return [gr.update(visible=False), gr.update(visible=True)]
    else:
        return [gr.update(visible=False), gr.update(visible=False)]


def create_evaluation_tab() -> Dict[str, Any]:
    with gr.Row():
        with gr.Column(scale=2):
            _ = gr.Markdown(
                value="\N{WHITE MEDIUM STAR} **Upload Files for Evaluation**"
            )
            eval_exp_id = gr.Textbox(
                label="\N{rocket} Evaluation Experiment ID (e.g. table name)",
                elem_id="eval_exp_id",
            )
            with gr.Row():
                chunk_size = gr.Textbox(
                    label="Chunk Size (The size of the chunks into which a document is divided)",
                    elem_id="chunk_size",
                    value=500,
                )
                chunk_overlap = gr.Textbox(
                    label="Chunk Overlap (The portion of adjacent document chunks that overlap with each other)",
                    elem_id="chunk_overlap",
                    value=10,
                )
                enable_qa_extraction = gr.Checkbox(
                    label="Yes",
                    info="Process with QA Extraction Model",
                    elem_id="enable_qa_extraction",
                )
                enable_raptor = gr.Checkbox(
                    label="Yes",
                    info="Process with Raptor Node Enhancement",
                    elem_id="enable_raptor",
                )
                enable_multimodal = gr.Checkbox(
                    label="Yes",
                    info="Process with MultiModal",
                    elem_id="enable_multimodal",
                    visible=False,
                )
                enable_table_summary = gr.Checkbox(
                    label="Yes",
                    info="Process with Table Summary ",
                    elem_id="enable_table_summary",
                )
                enable_eval = gr.Checkbox(
                    value=True,
                    info="Process with Evaluation",
                    elem_id="enable_eval",
                    visible=False,
                )
                with gr.Column(scale=8):
                    with gr.Tab("Files"):
                        upload_file = gr.File(
                            label="Upload a knowledge file for evaluation.",
                            file_count="multiple",
                        )
                        upload_file_state_df = gr.DataFrame(
                            label="Upload Status Info", visible=False
                        )
                        upload_file_state = gr.Textbox(
                            label="Upload Status", visible=False
                        )
                    with gr.Tab("Directory"):
                        upload_file_dir = gr.File(
                            label="Upload a knowledge directory for evaluation.",
                            file_count="directory",
                        )
                        upload_dir_state_df = gr.DataFrame(
                            label="Upload Status Info", visible=False
                        )
                        upload_dir_state = gr.Textbox(
                            label="Upload Status", visible=False
                        )
                    upload_file.upload(
                        fn=upload_knowledge,
                        inputs=[
                            upload_file,
                            chunk_size,
                            chunk_overlap,
                            enable_qa_extraction,
                            enable_raptor,
                            enable_multimodal,
                            enable_table_summary,
                            enable_eval,  # enable_eval
                            eval_exp_id,
                        ],
                        outputs=[upload_file_state_df, upload_file_state],
                        api_name="upload_knowledge_for_eval",
                    )
                    upload_file.clear(
                        fn=clear_files,
                        inputs=[],
                        outputs=[upload_file_state_df, upload_file_state],
                        api_name="clear_file_for_eval",
                    )
                    upload_file_dir.upload(
                        fn=upload_knowledge,
                        inputs=[
                            upload_file_dir,
                            chunk_size,
                            chunk_overlap,
                            enable_qa_extraction,
                            enable_raptor,
                            enable_multimodal,
                            enable_table_summary,
                        ],
                        outputs=[upload_dir_state_df, upload_dir_state],
                        api_name="upload_knowledge_dir_for_eval",
                    )
                    upload_file_dir.clear(
                        fn=clear_files,
                        inputs=[],
                        outputs=[upload_dir_state_df, upload_dir_state],
                        api_name="clear_file_dir_for_eval",
                    )

        with gr.Column(scale=2):
            _ = gr.Markdown(value="\N{WHITE MEDIUM STAR} **Automated QA Generation**")
            generate_btn = gr.Button(
                "Generate QA Dataset", variant="primary", elem_id="generate_btn"
            )
            overwrite_qa = gr.Checkbox(
                label="Overwrite QA (If the index has changed, regenerate and overwrite the existing QA dataset.)",
                elem_id="overwrite_qa",
                value=False,
            )
            qa_dataset_file = gr.components.File(
                label="QA Dadaset Results",
                elem_id="qa_dataset_file",
            )
            qa_dataset_json_text = gr.JSON(
                label="Displaying QA Dataset Results: Only the first 2 records are shown here. For a complete view, please download the file.",
                scale=1,
                elem_id="qa_dataset_json_text",
            )

            generate_btn.click(
                fn=generate_and_download_qa_file,
                inputs=[overwrite_qa, eval_exp_id],
                outputs=[qa_dataset_file, qa_dataset_json_text],
            )

        with gr.Column(scale=2):
            _ = gr.Markdown(value="\N{WHITE MEDIUM STAR} **Parameters of Evaluation**")
            retrieval_mode = gr.Radio(
                ["Embedding Only", "Keyword Only", "Hybrid"],
                label="Retrieval Mode",
                elem_id="eval_retrieval_mode",
            )

            reranker_type = gr.Radio(
                ["simple-weighted-reranker", "model-based-reranker"],
                label="Reranker Type",
                elem_id="eval_reranker_type",
            )
            with gr.Row(visible=False) as simple_reranker_col:
                vector_weight = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.7,
                    elem_id="vector_weight",
                    label="Weight of embedding retrieval results",
                )
                keyword_weight = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=float(1 - vector_weight.value),
                    elem_id="keyword_weight",
                    label="Weight of keyword retrieval results",
                    interactive=False,
                )

            with gr.Column(visible=False) as model_reranker_col:
                reranker_model = gr.Radio(
                    [
                        "bge-reranker-base",
                        "bge-reranker-large",
                    ],
                    label="Re-Ranker Model (Note: It will take a long time to load the model when using it for the first time.)",
                    elem_id="reranker_model",
                )

            with gr.Row():
                similarity_top_k = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    elem_id="similarity_top_k",
                    label="Text Top K (choose between 0 and 100)",
                )
                image_similarity_top_k = gr.Slider(
                    minimum=0,
                    maximum=10,
                    step=1,
                    elem_id="image_similarity_top_k",
                    label="Image Top K (choose between 0 and 10)",
                )
                need_image = gr.Checkbox(
                    label="Inference with multi-modal LLM",
                    info="Inference with multi-modal LLM.",
                    elem_id="need_image",
                    value=True,
                    visible=False,
                )
                similarity_threshold = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    elem_id="similarity_threshold",
                    label="Similarity Score Threshold (The more similar the items, the bigger the value.)",
                )

            vector_weight.input(
                fn=change_weight,
                inputs=vector_weight,
                outputs=[keyword_weight],
            )

            reranker_type.change(
                fn=change_reranker_type,
                inputs=[retrieval_mode, reranker_type],
                outputs=[simple_reranker_col, model_reranker_col],
            )

            retrieval_mode.change(
                fn=change_reranker_type,
                inputs=[retrieval_mode, reranker_type],
                outputs=[simple_reranker_col, model_reranker_col],
            )

            vec_args = {
                retrieval_mode,
                reranker_type,
                vector_weight,
                keyword_weight,
                similarity_top_k,
                image_similarity_top_k,
                need_image,
                similarity_threshold,
                reranker_model,
                eval_exp_id,
            }

        with gr.Column(scale=2):
            _ = gr.Markdown(value="\N{WHITE MEDIUM STAR} **Evaluation Results**")

            eval_retrieval_btn = gr.Button(
                "Evaluate Retrieval Stage",
                variant="primary",
                elem_id="eval_retrieval_btn",
            )
            eval_retrieval_res = gr.DataFrame(
                label="\N{rocket} Evaluation Resultes for Retrieval ",
                elem_id="eval_retrieval_res",
            )

            eval_retrieval_btn.click(
                fn=eval_retrieval_stage, inputs=vec_args, outputs=eval_retrieval_res
            )

            eval_response_btn = gr.Button(
                "Evaluate Response Stage",
                variant="primary",
                elem_id="eval_response_btn",
            )
            eval_response_res = gr.DataFrame(
                label="\N{rocket} Evaluation Resultes for Response ",
                elem_id="eval_response_res",
            )
            eval_response_btn.click(
                fn=eval_response_stage, inputs=vec_args, outputs=eval_response_res
            )

        return {
            qa_dataset_file.elem_id: qa_dataset_file,
            qa_dataset_json_text.elem_id: qa_dataset_json_text,
            eval_retrieval_res.elem_id: eval_retrieval_res,
            eval_response_res.elem_id: eval_response_res,
        }
