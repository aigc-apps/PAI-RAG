from typing import Dict, Any
import json
from pai_rag.app.web.rag_client import rag_client
import gradio as gr


def upload_config_file_fn(upload_config_file):
    if upload_config_file is None:
        return None
    with open(upload_config_file.name, "rb") as file:
        res = json.loads(file.read())
    return res


def update_agent_config(upload_config_file):
    res = rag_client.load_agent_config(upload_config_file.name)
    return res


def respond(agent_question, agent_chatbot):
    response_gen = rag_client.query(agent_question, with_intent=True)
    content = ""
    agent_chatbot.append((agent_question, content))
    for resp in response_gen:
        agent_chatbot[-1] = (agent_question, resp.result)
        yield agent_chatbot


def clear_history(chatbot):
    rag_client.clear_history()
    chatbot = []
    return chatbot


def reset_textbox():
    return gr.update(value="")


def create_agent_tab() -> Dict[str, Any]:
    with gr.Row():
        with gr.Column(scale=4):
            _ = gr.Markdown(
                value="**Upload the agent config file with function calling**"
            )
            with gr.Row():
                with gr.Column(scale=7):
                    upload_config_file = gr.File(
                        label="Upload agent config file for function calling.",
                        file_count="single",
                        file_types=[".json"],
                        elem_id="upload_config_file",
                        scale=8,
                    )
                with gr.Column(variant="panel", scale=3):
                    config_submit = gr.Button(
                        "Submit Config",
                        elem_id="config_submit",
                        variant="primary",
                        scale=6,
                    )
                    config_submit_state = gr.Textbox(label="Submit Status", scale=4)
            upload_file_content = gr.JSON(
                label="Displaying config content", elem_id="upload_file_content"
            )

        upload_config_file.upload(
            fn=upload_config_file_fn,
            inputs=[upload_config_file],
            outputs=[upload_file_content],
            api_name="upload_config_file_fn",
        )
        config_submit.click(
            update_agent_config,
            [upload_config_file],
            [config_submit_state],
            api_name="agent_config_submit_clk",
        )

        with gr.Column(scale=6):
            _ = gr.Markdown(value="**Agentic RAG Chatbot Test**")
            agent_chatbot = gr.Chatbot(height=500, elem_id="agent_chatbot")
            agent_question = gr.Textbox(
                label="Enter your question.", elem_id="agent_question"
            )
            with gr.Row():
                submitBtn = gr.Button("Submit", variant="primary")
                clearBtn = gr.Button("Clear History", variant="secondary")
            submitBtn.click(
                respond,
                [agent_question, agent_chatbot],
                [agent_chatbot],
                api_name="agent_respond_clk",
            )
            agent_question.submit(
                respond,
                [agent_question, agent_chatbot],
                [agent_chatbot],
                api_name="agent_respond_q",
            )
            submitBtn.click(
                reset_textbox,
                [],
                [agent_question],
                api_name="agent_reset_clk",
            )
            agent_question.submit(
                reset_textbox,
                [],
                [agent_question],
                api_name="agent_reset_q",
            )

            clearBtn.click(clear_history, [agent_chatbot], [agent_chatbot])
        return {
            upload_config_file.elem_id: upload_config_file,
            upload_file_content.elem_id: upload_file_content,
        }
