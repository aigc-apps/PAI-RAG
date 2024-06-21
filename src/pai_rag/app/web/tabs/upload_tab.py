import os
from typing import Dict, Any
import gradio as gr
import time
from pai_rag.app.web.rag_client import rag_client
from pai_rag.utils.file_utils import MyUploadFile
import pandas as pd
import asyncio


def upload_knowledge(upload_files, chunk_size, chunk_overlap, enable_qa_extraction):
    rag_client.patch_config({"chunk_size": chunk_size, "chunk_overlap": chunk_overlap})

    if not upload_files:
        yield [
            gr.update(visible=False),
            gr.update(
                visible=True,
                value="No file selected. Please choose at least one file.",
            ),
        ]

    my_upload_files = []
    for file in upload_files:
        file_dir = os.path.dirname(file.name)
        response = rag_client.add_knowledge(file_dir, enable_qa_extraction)
        my_upload_files.append(
            MyUploadFile(os.path.basename(file.name), response["task_id"])
        )

    result = {"Info": ["StartTime", "EndTime", "Duration(s)", "Status"]}
    while not all(file.finished is True for file in my_upload_files):
        for file in my_upload_files:
            response = asyncio.run(rag_client.get_knowledge_state(str(file.task_id)))
            file.update_state(response["status"])
            file.update_process_duration()
            result[file.file_name] = file.__info__()
            if response["status"] in ["completed", "failed"]:
                file.is_finished()
        yield [
            gr.update(visible=True, value=pd.DataFrame(result)),
            gr.update(visible=False),
        ]
        time.sleep(2)

    yield [
        gr.update(visible=True, value=pd.DataFrame(result)),
        gr.update(
            visible=True,
            value="Uploaded all files successfully!  \n Relevant content has been added to the vector store, you can now start chatting and asking questions.",
        ),
    ]


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
                upload_file_state_df = gr.DataFrame(
                    label="Upload Status Info", visible=False
                )
                upload_file_state = gr.Textbox(label="Upload Status", visible=False)
            with gr.Tab("Directory"):
                upload_file_dir = gr.File(
                    label="Upload a knowledge directory.",
                    file_count="directory",
                )
                upload_dir_btn = gr.Button("Upload", variant="primary")
                upload_dir_state_df = gr.DataFrame(
                    label="Upload Status Info", visible=False
                )
                upload_dir_state = gr.Textbox(label="Upload Status", visible=False)
            upload_file_btn.click(
                fn=upload_knowledge,
                inputs=[
                    upload_file,
                    chunk_size,
                    chunk_overlap,
                    enable_qa_extraction,
                ],
                outputs=[upload_file_state_df, upload_file_state],
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
                outputs=[upload_dir_state_df, upload_dir_state],
                api_name="upload_knowledge_dir",
            )
            return {
                chunk_size.elem_id: chunk_size,
                chunk_overlap.elem_id: chunk_overlap,
                enable_qa_extraction.elem_id: enable_qa_extraction,
            }
