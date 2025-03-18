from typing import Dict, Any
from pai_rag.app.web.rag_local_client import rag_client
import gradio as gr


async def respond(
    agent_api_definition,
    agent_function_definition,
    agent_python_scripts,
    agent_system_prompt,
    agent_question,
    agent_chatbot,
):
    update_dict = {
        "agent_api_definition": agent_api_definition,
        "agent_function_definition": agent_function_definition,
        "agent_python_scripts": agent_python_scripts,
        "agent_system_prompt": agent_system_prompt,
    }

    rag_client.patch_config(update_dict)

    q_msg = {"content": agent_question, "role": "user"}
    agent_chatbot.append(q_msg)
    content = ""
    a_msg = {"content": "", "role": "assistant"}
    agent_chatbot.append(a_msg)

    response_gen = rag_client.query(
        chat_messages=agent_chatbot[:-1], with_intent=True, stream=True
    )

    yield agent_chatbot

    async for resp in response_gen:
        content += resp.delta
        agent_chatbot[-1]["content"] = content
        yield agent_chatbot


def clear_history(chatbot):
    chatbot = []
    return chatbot


def reset_textbox():
    return gr.update(value="")


def create_agent_tab() -> Dict[str, Any]:
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Tab(label="API 工具"):
                agent_system_prompt = gr.Textbox(
                    label="工具调用系统提示词模板",
                    elem_id="agent_system_prompt",
                    lines=5,
                    interactive=True,
                )
                agent_api_definition = gr.Code(
                    label="API 工具定义",
                    elem_id="agent_api_definition",
                    interactive=True,
                    language="json",
                )
            with gr.Tab(label="Python 工具"):
                agent_function_definition = gr.Code(
                    label="Python 工具定义",
                    elem_id="agent_function_definition",
                    interactive=True,
                    language="json",
                )
                agent_python_scripts = gr.Code(
                    label="Python 工具脚本",
                    elem_id="agent_python_scripts",
                    language="python",
                    interactive=True,
                )

        with gr.Column(scale=6):
            _ = gr.Markdown(value="**智能体对话测试**")
            agent_chatbot = gr.Chatbot(
                height=500, elem_id="agent_chatbot", type="messages"
            )
            agent_question = gr.Textbox(label="在这里输入您的问题.", elem_id="agent_question")
            with gr.Row():
                submitBtn = gr.Button("提交", variant="primary")
                clearBtn = gr.Button("清空历史", variant="secondary")
            submitBtn.click(
                respond,
                [
                    agent_api_definition,
                    agent_function_definition,
                    agent_python_scripts,
                    agent_system_prompt,
                    agent_question,
                    agent_chatbot,
                ],
                [agent_chatbot],
                api_name="agent_respond_clk",
            )
            agent_question.submit(
                respond,
                [
                    agent_api_definition,
                    agent_function_definition,
                    agent_python_scripts,
                    agent_system_prompt,
                    agent_question,
                    agent_chatbot,
                ],
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
            agent_system_prompt.elem_id: agent_system_prompt,
            agent_api_definition.elem_id: agent_api_definition,
            agent_function_definition.elem_id: agent_function_definition,
            agent_python_scripts.elem_id: agent_python_scripts,
        }
