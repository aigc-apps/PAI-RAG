import os
from typing import Dict, Any
import gradio as gr
from pai_rag.app.web.rag_client import rag_client
import tempfile
import json


def generate_and_download_qa_file():
    tmpdir = tempfile.mkdtemp()
    qa_content = rag_client.evaluate_for_generate_qa()
    outputPath = os.path.join(tmpdir, "qa_dataset_output.json")
    with open(outputPath, "w", encoding="utf-8") as f:
        json.dump(qa_content, f, ensure_ascii=False, indent=4)

    return outputPath


def create_evaluation_tab() -> Dict[str, Any]:
    with gr.Row():
        with gr.Column(scale=2):
            generate_btn = gr.Button("Generate QA Dataset", variant="primary")
            qa_dataset_file = gr.components.File(
                label="\N{rocket} QA Dadaset Results",
                elem_id="qa_dataset_file",
            )
            generate_btn.click(
                fn=generate_and_download_qa_file, outputs=qa_dataset_file
            )

        return {qa_dataset_file.elem_id: qa_dataset_file}
