from typing import Dict, Any
import gradio as gr
from pai_rag.app.web.rag_local_client import RagApiError, rag_client
import pandas as pd

IGNORE_FILE_LIST = [".DS_Store"]


async def upload_oss_knowledge(
    oss_path,
    number_workers,
    chunk_size,
    chunk_overlap,
    enable_multimodal,
    enable_mandatory_ocr,
    upload_index,
):
    if not oss_path:
        yield [
            gr.update(visible=False),
            gr.update(
                visible=True,
                value="No file selected. Please choose at least one file.",
            ),
        ]

    async for state_info in upload_knowledge(
        upload_files=[],
        oss_path=oss_path,
        number_workers=number_workers,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_multimodal=enable_multimodal,
        enable_mandatory_ocr=enable_mandatory_ocr,
        index_name=upload_index,
        from_oss=True,
    ):
        yield state_info


async def upload_files(
    upload_files,
    number_workers,
    chunk_size,
    chunk_overlap,
    enable_multimodal,
    enable_mandatory_ocr,
    upload_index,
):
    if not upload_files:
        yield [
            gr.update(visible=False),
            gr.update(
                visible=True,
                value="No file selected. Please choose at least one file.",
            ),
        ]

    async for state_info in upload_knowledge(
        upload_files=upload_files,
        oss_path=None,
        number_workers=number_workers,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_multimodal=enable_multimodal,
        enable_mandatory_ocr=enable_mandatory_ocr,
        index_name=upload_index,
    ):
        yield state_info


async def upload_knowledge(
    upload_files,
    oss_path,
    number_workers,
    chunk_size,
    chunk_overlap,
    enable_multimodal,
    enable_mandatory_ocr,
    index_name,
    from_oss: bool = False,
):
    try:
        rag_client.patch_config(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "enable_mandatory_ocr": enable_mandatory_ocr,
                "number_workers": int(number_workers),
            }
        )
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")

    if from_oss:
        response = rag_client.add_knowledge(
            oss_path=oss_path,
            index_name=index_name,
            enable_multimodal=enable_multimodal,
        )
    else:
        input_files = [file.name for file in upload_files]
        if len(input_files) == 0:
            return
        response = rag_client.add_knowledge(
            input_files=input_files,
            index_name=index_name,
            enable_multimodal=enable_multimodal,
        )

    error_msg = ""
    async for r in response:
        error_msg = r.get("detail")[0]
        yield [
            gr.update(visible=True, value=pd.DataFrame(r)),
            gr.update(
                visible=True,
                value="",
            ),
        ]

    upload_result = "Upload success."

    if error_msg:
        upload_result = f"Upload failed: {error_msg}"

    yield [
        gr.update(visible=True, value=pd.DataFrame(r)),
        gr.update(
            visible=True,
            value=upload_result,
        ),
    ]


def clear_files():
    yield [
        gr.update(visible=False, value=pd.DataFrame()),
        gr.update(visible=False, value=""),
    ]


def create_upload_tab() -> Dict[str, Any]:
    with gr.Row():
        with gr.Column(scale=2):
            upload_index = gr.Dropdown(
                choices=[],
                label="\N{bookmark} Index Name",
                elem_id="upload_index",
                allow_custom_value=True,
            )
            number_workers = gr.Slider(
                minimum=0,
                maximum=10,
                step=1,
                elem_id="number_workers",
                label="\N{fire} Number of workers to parallelize data-loading over.",
            )
            chunk_size = gr.Textbox(
                label="\N{rocket} Chunk Size (The size of the chunks into which a document is divided)",
                elem_id="chunk_size",
            )

            chunk_overlap = gr.Textbox(
                label="\N{fire} Chunk Overlap (The portion of adjacent document chunks that overlap with each other)",
                elem_id="chunk_overlap",
            )
            enable_multimodal = gr.Checkbox(
                label="Yes",
                info="Process with MultiModal",
                elem_id="enable_multimodal",
                visible=True,
            )
            enable_mandatory_ocr = gr.Checkbox(
                label="Yes",
                info="Process PDF with OCR",
                elem_id="enable_mandatory_ocr",
                visible=True,
            )
        with gr.Column(scale=8):
            with gr.Tab("Files"):
                upload_file = gr.File(
                    label="Upload a knowledge file.", file_count="multiple"
                )
                upload_file_state_df = gr.DataFrame(
                    label="Upload Status Info", visible=False
                )
                upload_file_state = gr.Textbox(label="Upload Status", visible=False)
            with gr.Tab("Directory"):
                upload_file_dir = gr.File(
                    label="Upload a knowledge directory.",
                    file_count="directory",
                )
                upload_dir_state_df = gr.DataFrame(
                    label="Upload Status Info", visible=False
                )
                upload_dir_state = gr.Textbox(label="Upload Status", visible=False)
            with gr.Tab("Aliyun OSS"):
                oss_path = gr.Textbox(
                    label="Aliyun OSS Path",
                    placeholder="oss://bucket_name/path/to/knowledge/",
                    elem_id="oss_path",
                )
                upload_oss = gr.Button(value="Upload from OSS", variant="primary")

                upload_oss_state_df = gr.DataFrame(
                    label="Upload Status Info", visible=False
                )
                upload_oss_state = gr.Textbox(label="Upload Status", visible=False)

                upload_oss.click(
                    fn=upload_oss_knowledge,
                    inputs=[
                        oss_path,
                        number_workers,
                        chunk_size,
                        chunk_overlap,
                        enable_multimodal,
                        enable_mandatory_ocr,
                        upload_index,
                    ],
                    outputs=[upload_oss_state_df, upload_oss_state],
                    api_name="upload_oss",
                )

            upload_file.upload(
                fn=upload_files,
                inputs=[
                    upload_file,
                    number_workers,
                    chunk_size,
                    chunk_overlap,
                    enable_multimodal,
                    enable_mandatory_ocr,
                    upload_index,
                ],
                outputs=[upload_file_state_df, upload_file_state],
                api_name="upload_knowledge",
            )
            upload_file.clear(
                fn=clear_files,
                inputs=[],
                outputs=[upload_file_state_df, upload_file_state],
                api_name="clear_file",
            )
            dummy_component = gr.Textbox(visible=False, value="")
            upload_file_dir.upload(
                fn=upload_knowledge,
                inputs=[
                    upload_file_dir,
                    dummy_component,
                    number_workers,
                    chunk_size,
                    chunk_overlap,
                    enable_multimodal,
                    enable_mandatory_ocr,
                    upload_index,
                ],
                outputs=[upload_dir_state_df, upload_dir_state],
                api_name="upload_knowledge_dir",
            )
            upload_file_dir.clear(
                fn=clear_files,
                inputs=[],
                outputs=[upload_dir_state_df, upload_dir_state],
                api_name="clear_file_dir",
            )
            return {
                upload_index.elem_id: upload_index,
                number_workers.elem_id: number_workers,
                chunk_size.elem_id: chunk_size,
                chunk_overlap.elem_id: chunk_overlap,
                enable_multimodal.elem_id: enable_multimodal,
                enable_mandatory_ocr.elem_id: enable_mandatory_ocr,
            }
