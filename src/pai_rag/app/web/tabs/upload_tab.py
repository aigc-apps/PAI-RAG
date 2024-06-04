import os
from typing import Dict, Any
import gradio as gr
from pai_rag.app.web.rag_client import rag_client
from pai_rag.app.web.view_model import view_model


def upload_knowledge(upload_files, chunk_size, chunk_overlap, enable_qa_extraction):
    view_model.chunk_size = chunk_size
    view_model.chunk_overlap = chunk_overlap
    new_config = view_model.to_app_config()
    rag_client.reload_config(new_config)

    if not upload_files:
        return "No file selected. Please choose at least one file."

    for file in upload_files:
        file_dir = os.path.dirname(file.name)
        rag_client.add_knowledge(file_dir, enable_qa_extraction)
    return (
        "Upload "
        + str(len(upload_files))
        + " files Success! \n \n Relevant content has been added to the vector store, you can now start chatting and asking questions."
    )


def create_upload_tab() -> Dict[str, Any]:
    with gr.Row():
        with gr.Column(scale=2):
            chunk_size = gr.Textbox(
                label="\N{rocket} Chunk Size (The size of the chunks into which a document is divided)",
                elem_id="chunk_size",
            )

            chunk_overlap = gr.Textbox(
                label="\N{fire} Chunk Overlap (The portion of adjacent document chunks that overlap with each other)",
                elem_id="chunk_overlap",
            )
            enable_qa_extraction = gr.Checkbox(
                label="Yes",
                info="Process with QA Extraction Model",
                elem_id="enable_qa_extraction",
            )
        with gr.Column(scale=8):
            with gr.Tab("Files"):
                upload_file = gr.File(
                    label="Upload a knowledge file.", file_count="multiple"
                )
                upload_file_btn = gr.Button("Upload", variant="primary")
                upload_file_state = gr.Textbox(label="Upload State")
            with gr.Tab("Directory"):
                upload_file_dir = gr.File(
                    label="Upload a knowledge directory.",
                    file_count="directory",
                )
                upload_dir_btn = gr.Button("Upload", variant="primary")
                upload_dir_state = gr.Textbox(label="Upload State")
            upload_file_btn.click(
                fn=upload_knowledge,
                inputs=[
                    upload_file,
                    chunk_size,
                    chunk_overlap,
                    enable_qa_extraction,
                ],
                outputs=upload_file_state,
                api_name="upload_knowledge",
            )
            upload_dir_btn.click(
                fn=upload_knowledge,
                inputs=[
                    upload_file_dir,
                    chunk_size,
                    chunk_overlap,
                    enable_qa_extraction,
                ],
                outputs=upload_dir_state,
                api_name="upload_knowledge_dir",
            )
            return {
                chunk_size.elem_id: chunk_size,
                chunk_overlap.elem_id: chunk_overlap,
                enable_qa_extraction.elem_id: enable_qa_extraction,
            }
