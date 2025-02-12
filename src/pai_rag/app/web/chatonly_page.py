import gradio as gr
from pai_rag.app.web.rag_client import RagApiError, rag_client


def clear_history(chatbot):
    rag_client.clear_history()
    chatbot = []
    return chatbot


def reset_textbox():
    return gr.update(value="")


def respond(retrieve_only, question, chatbot):
    # empty input.
    if not question:
        yield chatbot
        return

    if chatbot is not None:
        chatbot.append((question, ""))
        yield chatbot

    try:
        if retrieve_only:
            response_gen = rag_client.query_vector(question, index_name="default_index")

        else:
            response_gen = rag_client.query(
                question,
                with_history=False,
                stream=True,
                citation=False,
                index_name="default_index",
            )
        for resp in response_gen:
            chatbot[-1] = (question, resp.result)
            yield chatbot

    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")
    except Exception as e:
        raise gr.Error(f"Error: {e}")
    finally:
        yield chatbot


def create_chat_ui():
    with gr.Blocks() as chatpage:
        chatbot = gr.Chatbot(height=600, elem_id="chatbot")
        with gr.Row():
            retrieve_only = gr.Checkbox(
                label="Retrieve only",
                info="Query knowledge base directly without LLM.",
                elem_id="retrieve_only",
                value=False,
                scale=1,
            )
            question = gr.Textbox(
                label="Enter your question.", elem_id="question", scale=9
            )
            with gr.Row():
                submitBtn = gr.Button("Submit", variant="primary")
                clearBtn = gr.Button("Clear History", variant="secondary")

        submitBtn.click(
            respond,
            [
                retrieve_only,
                question,
                chatbot,
            ],
            [chatbot],
            api_name="respond_clk",
        )
        question.submit(
            respond,
            [
                retrieve_only,
                question,
                chatbot,
            ],
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

        clearBtn.click(clear_history, [chatbot], [chatbot])
        return chatpage
