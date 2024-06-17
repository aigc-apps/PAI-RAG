from typing import Dict, Any, List
import gradio as gr
from pai_rag.app.web.rag_client import rag_client
from pai_rag.app.web.view_model import view_model
from pai_rag.app.web.ui_constants import (
    SIMPLE_PROMPTS,
    GENERAL_PROMPTS,
    EXTRACT_URL_PROMPTS,
    ACCURATE_CONTENT_PROMPTS,
)
import json
import asyncio

current_session_id = None


def clear_history(chatbot):
    chatbot = []
    global current_session_id
    current_session_id = None
    return chatbot, 0


async def respond(input_elements: List[Any]):
    global current_session_id

    update_dict = {}
    for element, value in input_elements.items():
        update_dict[element.elem_id] = value

    # empty input.
    if not update_dict["question"]:
        yield "", update_dict["chatbot"], 0

    view_model.update(update_dict)
    new_config = view_model.to_app_config()
    rag_client.reload_config(new_config)

    query_type = update_dict["query_type"]
    msg = update_dict["question"]
    chatbot = update_dict["chatbot"]
    is_streaming = update_dict["is_streaming"]

    if not update_dict["include_history"]:
        current_session_id = None

    if query_type == "LLM":
        response = rag_client.query_llm(
            text=msg, session_id=current_session_id, stream=is_streaming
        )
        #
    elif query_type == "Retrieval":
        response = rag_client.query_vector(msg)
    else:
        response = rag_client.query(msg, session_id=current_session_id)
        current_session_id = response.session_id

    if is_streaming and query_type != "Retrieval":
        current_session_id = response.headers["x-session-id"]
        chatbot.append([msg, None])
        chatbot[-1][1] = ""
        from datetime import datetime

        # for chunk in response.iter_lines(chunk_size=8192,
        #                             decode_unicode=False,
        #                             delimiter=b'\0'):
        async for chunk in response:
            print(datetime.now())
            print(chunk.delta, end="")
            print("Gradio UI ===== ", chunk.delta)
            if chunk:
                # chatbot[-1][1] += chunk.decode("utf-8")
                chatbot[-1][1] += chunk.delta
                yield "", chatbot, 0
                await asyncio.sleep(0.1)
    else:
        response = json.loads(response.text)
        current_session_id = response["session_id"]
        chatbot.append((msg, response["answer"]))
        yield "", chatbot, 0


def create_chat_tab() -> Dict[str, Any]:
    with gr.Row():
        with gr.Column(scale=2):
            query_type = gr.Radio(
                ["Retrieval", "LLM", "RAG (Retrieval + LLM)"],
                label="\N{fire} Which query do you want to use?",
                elem_id="query_type",
                value="RAG (Retrieval + LLM)",
            )
            is_streaming = gr.Checkbox(
                label="Streaming Output",
                info="Streaming Output",
                elem_id="is_streaming",
                value=True,
            )
            with gr.Column(visible=True) as vs_col:
                vec_model_argument = gr.Accordion(
                    "Parameters of Vector Retrieval", open=False
                )

                with vec_model_argument:
                    similarity_top_k = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        elem_id="similarity_top_k",
                        label="Top K (choose between 0 and 100)",
                    )
                    # similarity_cutoff = gr.Slider(minimum=0, maximum=1, step=0.01,elem_id="similarity_cutoff",value=view_model.similarity_cutoff, label="Similarity Distance Threshold (The more similar the vectors, the smaller the value.)")
                    rerank_model = gr.Radio(
                        [
                            "no-reranker",
                            "bge-reranker-base",
                            "bge-reranker-large",
                            "llm-reranker",
                        ],
                        label="Re-Rank Model (Note: It will take a long time to load the model when using it for the first time.)",
                        elem_id="rerank_model",
                    )
                    retrieval_mode = gr.Radio(
                        ["Embedding Only", "Keyword Only", "Hybrid"],
                        label="Retrieval Mode",
                        elem_id="retrieval_mode",
                    )
                vec_args = {
                    similarity_top_k,
                    # similarity_cutoff,
                    rerank_model,
                    retrieval_mode,
                }
            with gr.Column(visible=True) as llm_col:
                model_argument = gr.Accordion("Inference Parameters of LLM", open=False)
                with model_argument:
                    include_history = gr.Checkbox(
                        label="Chat history",
                        info="Query with chat history.",
                        elem_id="include_history",
                    )
                    llm_temp = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.001,
                        value=0.1,
                        elem_id="llm_temperature",
                        label="Temperature (choose between 0 and 1)",
                    )
                llm_args = {llm_temp, include_history}

            with gr.Column(visible=True) as lc_col:
                prm_type = gr.Radio(
                    [
                        "Simple",
                        "General",
                        "Extract URL",
                        "Accurate Content",
                        "Custom",
                    ],
                    label="\N{rocket} Please choose the prompt template type",
                    elem_id="prm_type",
                )
                text_qa_template = gr.Textbox(
                    label="prompt template",
                    placeholder="",
                    elem_id="text_qa_template",
                    lines=4,
                )

                def change_prompt_template(prm_type):
                    if prm_type == "Simple":
                        return {
                            text_qa_template: gr.update(
                                value=SIMPLE_PROMPTS, interactive=False
                            )
                        }
                    elif prm_type == "General":
                        return {
                            text_qa_template: gr.update(
                                value=GENERAL_PROMPTS, interactive=False
                            )
                        }
                    elif prm_type == "Extract URL":
                        return {
                            text_qa_template: gr.update(
                                value=EXTRACT_URL_PROMPTS, interactive=False
                            )
                        }
                    elif prm_type == "Accurate Content":
                        return {
                            text_qa_template: gr.update(
                                value=ACCURATE_CONTENT_PROMPTS,
                                interactive=False,
                            )
                        }
                    else:
                        return {text_qa_template: gr.update(value="", interactive=True)}

                prm_type.change(
                    fn=change_prompt_template,
                    inputs=prm_type,
                    outputs=[text_qa_template],
                )

            cur_tokens = gr.Textbox(
                label="\N{fire} Current total count of tokens", visible=False
            )

            def change_query_radio(query_type):
                global current_session_id
                current_session_id = None
                if query_type == "Retrieval":
                    return {
                        vs_col: gr.update(visible=True),
                        vec_model_argument: gr.update(open=True),
                        llm_col: gr.update(visible=False),
                        model_argument: gr.update(open=False),
                        lc_col: gr.update(visible=False),
                    }
                elif query_type == "LLM":
                    return {
                        vs_col: gr.update(visible=False),
                        vec_model_argument: gr.update(open=False),
                        llm_col: gr.update(visible=True),
                        model_argument: gr.update(open=True),
                        lc_col: gr.update(visible=False),
                    }
                elif query_type == "RAG (Retrieval + LLM)":
                    return {
                        vs_col: gr.update(visible=True),
                        vec_model_argument: gr.update(open=False),
                        llm_col: gr.update(visible=True),
                        model_argument: gr.update(open=False),
                        lc_col: gr.update(visible=True),
                    }

            query_type.change(
                fn=change_query_radio,
                inputs=query_type,
                outputs=[vs_col, vec_model_argument, llm_col, model_argument, lc_col],
            )

        with gr.Column(scale=8):
            chatbot = gr.Chatbot(height=500, elem_id="chatbot")
            question = gr.Textbox(label="Enter your question.", elem_id="question")
            with gr.Row():
                submitBtn = gr.Button("Submit", variant="primary")
                clearBtn = gr.Button("Clear History", variant="secondary")

        chat_args = (
            {text_qa_template, question, query_type, chatbot, is_streaming}
            .union(vec_args)
            .union(llm_args)
        )

        submitBtn.click(
            respond,
            chat_args,
            [question, chatbot, cur_tokens],
            api_name="respond",
        )
        question.submit(
            respond,
            chat_args,
            [question, chatbot, cur_tokens],
            api_name="respond",
        )
        clearBtn.click(clear_history, [chatbot], [chatbot, cur_tokens])
        return {
            similarity_top_k.elem_id: similarity_top_k,
            rerank_model.elem_id: rerank_model,
            retrieval_mode.elem_id: retrieval_mode,
            prm_type.elem_id: prm_type,
            text_qa_template.elem_id: text_qa_template,
        }
