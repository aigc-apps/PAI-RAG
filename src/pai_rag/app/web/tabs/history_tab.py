import gradio as gr
import pandas as pd
from pai_rag.app.web.rag_local_client import rag_client


def refresh_upload_history(knowledgebase_name):
    upload_jobs = rag_client.get_upload_history(knowledgebase_name)
    if upload_jobs:
        history_data = pd.DataFrame(upload_jobs)
        history_data["文件名"] = history_data["file_name"]
        history_data["上传状态"] = history_data["status"]
        history_data["更新时间"] = history_data["last_modified_time"]
        history_data["错误原因"] = history_data["message"]
        history_data = history_data[["文件名", "上传状态", "更新时间", "错误原因"]].sort_values(
            by="last_modified_time", ascending=False
        )
    else:
        history_data = pd.DataFrame(columns=["文件名", "上传状态", "更新时间", "错误原因"])
    summary = f"累计上传{len(upload_jobs)}个文件。"
    if len(upload_jobs) > 0:
        finished = [job for job in upload_jobs if job["status"] == "done"]
        failed = [job for job in upload_jobs if job["status"] == "failed"]
        summary += f"成功上传{len(finished)}个文件，失败{len(failed)}个文件。"

    return [
        gr.update(value=summary),
        gr.update(value=history_data),
    ]


def create_upload_history():
    with gr.Row():
        history_index = gr.Dropdown(
            choices=[],
            value="",
            label="\N{bookmark} 知识库名称",
            elem_id="history_index",
            allow_custom_value=True,
        )

        upload_summary = gr.HTML(value="累计上传0个文件", elem_id="upload_summary")
        refresh_button = gr.Button(
            value="刷新",
            elem_id="refresh_button",
            variant="primary",
        )
    upload_history = gr.DataFrame(
        label="上传历史",
        visible=True,
        elem_id="upload_history",
        headers=["文件名", "上传状态", "更新时间", "错误原因"],
    )

    history_index.change(
        fn=refresh_upload_history,
        inputs=[history_index],
        outputs=[upload_summary, upload_history],
    )

    refresh_button.click(
        fn=refresh_upload_history,
        inputs=[history_index],
        outputs=[upload_summary, upload_history],
    )

    return {
        history_index.elem_id: history_index,
        upload_summary.elem_id: upload_summary,
        upload_history.elem_id: upload_history,
    }
