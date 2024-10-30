from typing import Dict, Any
from pai_rag.app.web.rag_client import rag_client
import gradio as gr


def respond(
    intent_description,
    agent_api_definition,
    agent_function_definition,
    agent_python_scripts,
    agent_system_prompt,
    agent_question,
    agent_chatbot,
):
    update_dict = {
        "intent_description": intent_description,
        "agent_api_definition": agent_api_definition,
        "agent_function_definition": agent_function_definition,
        "agent_python_scripts": agent_python_scripts,
        "agent_system_prompt": agent_system_prompt,
    }

    rag_client.patch_config(update_dict)

    response_gen = rag_client.query(agent_question, with_intent=True, stream=True)
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
            with gr.Tab(label="Intents & Prompts"):
                intent_description = gr.Code(
                    label="Intent Descriptions",
                    elem_id="intent_description",
                    interactive=True,
                    language="json",
                )
            with gr.Tab(label="API tools"):
                agent_system_prompt = gr.Textbox(
                    label="Function-call system Prompt",
                    elem_id="agent_system_prompt",
                    lines=5,
                    interactive=True,
                )
                agent_api_definition = gr.Code(
                    label="API Tool Definitions",
                    elem_id="agent_api_definition",
                    interactive=True,
                    language="json",
                )
            with gr.Tab(label="Python tools"):
                agent_function_definition = gr.Code(
                    label="Python Tool Definitions",
                    elem_id="agent_function_definition",
                    interactive=True,
                    language="json",
                )
                agent_python_scripts = gr.Code(
                    label="Python Tool Scripts",
                    elem_id="agent_python_scripts",
                    language="python",
                    interactive=True,
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
                [
                    intent_description,
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
                    intent_description,
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
            intent_description.elem_id: intent_description,
            agent_system_prompt.elem_id: agent_system_prompt,
            agent_api_definition.elem_id: agent_api_definition,
            agent_function_definition.elem_id: agent_function_definition,
            agent_python_scripts.elem_id: agent_python_scripts,
        }
