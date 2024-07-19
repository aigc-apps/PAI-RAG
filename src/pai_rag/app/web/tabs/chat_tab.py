from typing import Dict, Any, List
import gradio as gr
from pai_rag.app.web.rag_client import RagApiError, rag_client
from pai_rag.app.web.ui_constants import (
    SIMPLE_PROMPTS,
    GENERAL_PROMPTS,
    EXTRACT_URL_PROMPTS,
    ACCURATE_CONTENT_PROMPTS,
)
import time
import urllib.parse

current_session_id = None


def clear_history(chatbot):
    chatbot = []
    global current_session_id
    current_session_id = None
    return chatbot, 0


def reset_textbox():
    return gr.update(value="")


def respond(input_elements: List[Any]):
    global current_session_id
    update_dict = {}
    for element, value in input_elements.items():
        update_dict[element.elem_id] = value

    # empty input.
    if not update_dict["question"]:
        yield "", update_dict["chatbot"], 0

    try:
        rag_client.patch_config(update_dict)
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")

    query_type = update_dict["query_type"]
    msg = update_dict["question"]
    chatbot = update_dict["chatbot"]
    is_streaming = update_dict["is_streaming"]

    if not update_dict["include_history"]:
        current_session_id = None

    try:
        if query_type == "LLM":
            response = rag_client.query_llm(
                msg, session_id=current_session_id, stream=is_streaming
            )
        elif query_type == "Retrieval":
            response = rag_client.query_vector(msg)
        else:
            response = rag_client.query(
                msg, session_id=current_session_id, stream=is_streaming
            )

    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")

    if query_type == "Retrieval":
        chatbot.append((msg, response.answer))
        yield chatbot
    elif is_streaming:
        current_session_id = response.headers["x-session-id"]
        chatbot.append([msg, None])
        chatbot[-1][1] = ""
        for chunk in response.iter_lines(
            chunk_size=8192, decode_unicode=False, delimiter=b"\0"
        ):
            if chunk:
                chatbot[-1][1] += chunk.decode("utf-8")
                yield chatbot
                time.sleep(0.1)
        if query_type != "LLM":
            images_decoded_string = urllib.parse.unquote(response.headers["images"])
            chatbot[-1][1] += f"\n\n{images_decoded_string}"
            docs_decoded_string = urllib.parse.unquote(response.headers["docs"])
            chatbot[-1][1] += "\n\n **Reference:** \n" + docs_decoded_string.replace(
                "+++", "\n"
            )
            yield chatbot
    else:
        current_session_id = response["session_id"]
        chatbot.append((msg, response["answer"]))
        yield chatbot


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
                    retrieval_mode = gr.Radio(
                        ["Embedding Only", "Keyword Only", "Hybrid"],
                        label="Retrieval Mode",
                        elem_id="retrieval_mode",
                    )

                    reranker_type = gr.Radio(
                        ["simple-weighted-reranker", "model-based-reranker"],
                        label="Reranker Type",
                        elem_id="reranker_type",
                    )

                    with gr.Column(
                        visible=(reranker_type == "simple-weighted-reranker")
                    ) as simple_reranker_col:
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

                    with gr.Column(
                        visible=(reranker_type == "model-based-reranker")
                    ) as model_reranker_col:
                        reranker_model = gr.Radio(
                            [
                                "bge-reranker-base",
                                "bge-reranker-large",
                            ],
                            label="Re-Ranker Model (Note: It will take a long time to load the model when using it for the first time.)",
                            elem_id="reranker_model",
                        )

                    with gr.Column():
                        similarity_top_k = gr.Slider(
                            minimum=0,
                            maximum=100,
                            step=1,
                            elem_id="similarity_top_k",
                            label="Top K (choose between 0 and 100)",
                        )
                        similarity_threshold = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            elem_id="similarity_threshold",
                            label="Similarity Score Threshold (The more similar the items, the bigger the value.)",
                        )

                    def change_weight(change_weight):
                        return round(float(1 - change_weight), 2)

                    vector_weight.change(
                        fn=change_weight,
                        inputs=vector_weight,
                        outputs=[keyword_weight],
                    )

                    def change_reranker_type(reranker_type):
                        if reranker_type == "simple-weighted-reranker":
                            return {
                                simple_reranker_col: gr.update(visible=True),
                                model_reranker_col: gr.update(visible=False),
                            }
                        elif reranker_type == "model-based-reranker":
                            return {
                                simple_reranker_col: gr.update(visible=False),
                                model_reranker_col: gr.update(visible=True),
                            }
                        else:
                            return {
                                simple_reranker_col: gr.update(visible=False),
                                model_reranker_col: gr.update(visible=False),
                            }

                    def change_retrieval_mode(retrieval_mode):
                        if retrieval_mode == "Hybrid":
                            return {simple_reranker_col: gr.update(visible=True)}
                        else:
                            return {simple_reranker_col: gr.update(visible=False)}

                    reranker_type.change(
                        fn=change_reranker_type,
                        inputs=reranker_type,
                        outputs=[simple_reranker_col, model_reranker_col],
                    )

                    retrieval_mode.change(
                        fn=change_retrieval_mode,
                        inputs=retrieval_mode,
                        outputs=[simple_reranker_col],
                    )

                vec_args = {
                    retrieval_mode,
                    reranker_type,
                    vector_weight,
                    keyword_weight,
                    similarity_top_k,
                    similarity_threshold,
                    reranker_model,
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
            [chatbot],
            api_name="respond_clk",
        )
        question.submit(
            respond,
            chat_args,
            [chatbot],
            api_name="respond_q",
        )
        submitBtn.click(
            reset_textbox,
            [],
            [question],
            api_name="reset_clk",
        )
        question.submit(
            reset_textbox,
            [],
            [question],
            api_name="reset_q",
        )

        clearBtn.click(clear_history, [chatbot], [chatbot, cur_tokens])
        return {
            similarity_top_k.elem_id: similarity_top_k,
            retrieval_mode.elem_id: retrieval_mode,
            reranker_type.elem_id: reranker_type,
            reranker_model.elem_id: reranker_model,
            vector_weight.elem_id: vector_weight,
            keyword_weight.elem_id: keyword_weight,
            similarity_threshold.elem_id: similarity_threshold,
            prm_type.elem_id: prm_type,
            text_qa_template.elem_id: text_qa_template,
        }
