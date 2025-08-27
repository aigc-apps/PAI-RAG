from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ChatResponseAsyncGen,
    ChatResponse,
)


def response_from_text(text: str, additional_kwargs:dict={}) -> ChatResponse:
    print(text)
    return ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT.value,
            content=text,
        ),
        additional_kwargs=additional_kwargs,
        delta=text,
    )


async def response_gen_from_text(text: str, additional_kwargs:dict={}) -> ChatResponseAsyncGen:
    yield response_from_text(text, additional_kwargs=additional_kwargs)
