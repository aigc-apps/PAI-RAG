from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponse,
    CompletionResponseGen,
    CompletionResponseAsyncGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.generic_utils import (
    completion_to_chat_decorator,
    stream_completion_to_chat_decorator,
    acompletion_to_chat_decorator,
)
from typing import Any, Dict, Optional, Sequence
from llama_index.core.bridge.pydantic import Field
import json
import requests
import httpx


DEFAULT_EAS_MODEL_NAME = "pai-eas-custom-llm"
DEFAULT_EAS_MAX_NEW_TOKENS = 512


class PaiEAS(CustomLLM):
    """PaiEas LLM."""

    model_name: str = Field(
        default=DEFAULT_EAS_MODEL_NAME,
        description="The DashScope model to use.",
    )
    endpoint: str = Field(default=None, description="The PAI EAS endpoint.")
    token: str = Field(default=None, description="The PAI EAS token.")
    max_new_tokens: int = Field(
        default=DEFAULT_EAS_MAX_NEW_TOKENS, description="Max new tokens."
    )
    temperature: Optional[float] = Field(
        description="The temperature to use during generation.",
        default=0.1,
        gte=0.0,
        lte=2.0,
    )
    top_p: float = Field(
        default=0.8, description="Sample probability threshold when generate."
    )
    top_k: int = Field(default=30, description="Sample counter when generate.")
    top_k: int = Field(default=30, description="Sample counter when generate.")
    version: str = Field(default="2.0", description="PAI EAS endpoint version.")

    """
    Llm model deployed in Aliyun PAI EAS.
    """

    def __init__(
        self,
        endpoint: str = "",
        token: str = "",
        model_name: str = "",
        max_new_tokens: int = DEFAULT_EAS_MAX_NEW_TOKENS,
        temperature: float = 0.1,
        top_p: float = 0.8,
        top_k: int = 30,
        version: str = "2.0",
        **kwargs: Any,
    ):
        super().__init__(
            endpoint=endpoint,
            token=token,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            version=version,
            kwargs=kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "PaiEAS"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model_name)

    def _default_params(self):
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
        }
        return params

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        params = self._default_params()
        params.update(kwargs)
        response = self._call_eas(prompt, params=params)
        text = self._process_eas_response(response)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        params = self._default_params()
        params.update(kwargs)
        response = await self._call_eas_async(prompt, params=params)
        text = self._process_eas_response(response)
        return CompletionResponse(text=text)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        chat_fn = completion_to_chat_decorator(self.complete)
        return chat_fn(messages, **kwargs)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        chat_fn = acompletion_to_chat_decorator(self.acomplete)
        return await chat_fn(messages, **kwargs)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        return self._stream(prompt=prompt, kwargs=kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        stream_chat_fn = stream_completion_to_chat_decorator(self.stream_complete)
        return stream_chat_fn(messages, **kwargs)

    def _process_eas_response(self, response: Any) -> str:
        if self.version == "1.0":
            text = response
        else:
            text = response["response"]
        return text

    def _call_eas(self, prompt: str = "", params: Dict = {}) -> Any:
        """Generate text from the eas service."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"{self.token}",
        }
        if self.version == "1.0":
            body = {
                "input_ids": f"{prompt}",
            }
        else:
            body = {
                "prompt": f"{prompt}",
            }

        # add params to body
        for key, value in params.items():
            body[key] = value

        # make request
        response = requests.post(self.endpoint, headers=headers, json=body)

        if response.status_code != 200:
            raise Exception(
                f"Request failed with status code {response.status_code}"
                f" and message {response.text}"
            )

        try:
            return json.loads(response.text)
        except Exception as e:
            raise e

    async def _call_eas_async(self, prompt: str = "", params: Dict = {}) -> Any:
        """Generate text from the eas service."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"{self.token}",
        }
        if self.version == "1.0":
            body = {
                "input_ids": f"{prompt}",
            }
        else:
            body = {
                "prompt": f"{prompt}",
            }

        # add params to body
        for key, value in params.items():
            body[key] = value

        # make request
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                url=self.endpoint, headers=headers, json=body, timeout=60
            )

        if response.status_code != 200:
            raise Exception(
                f"Request failed with status code {response.status_code}"
                f" and message {response.text}"
            )

        try:
            return json.loads(response.text)
        except Exception as e:
            raise e

    def _stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        params = self._default_params()
        headers = {"User-Agent": "PAI Rag Client", "Authorization": f"{self.token}"}
        print("_stream self.version", self.version)
        if self.version == "1.0":
            pload = {"input_ids": prompt, **params}
            response = requests.post(
                self.endpoint, headers=headers, json=pload, stream=True
            )

            res = CompletionResponse(text=response.text)

            # yield text, if any
            yield res
        else:
            pload = {"prompt": prompt, "use_stream_chat": "True", **params}

            response = requests.post(
                self.endpoint, headers=headers, json=pload, stream=True
            )

            previous_text = ""
            for chunk in response.iter_lines(
                chunk_size=8192, decode_unicode=False, delimiter=b"\0"
            ):
                if chunk:
                    data = json.loads(chunk.decode("utf-8"))
                    text = data["response"]

                    # yield text, if any
                    if text:
                        res = CompletionResponse(
                            text=text, delta=text[len(previous_text) :]
                        )
                        previous_text = text
                        yield res

    # TODO: true async request
    async def _astream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen:
        async def gen() -> CompletionResponseAsyncGen:
            for message in self._stream(prompt=prompt, kwargs=kwargs):
                yield message

        # NOTE: convert generator to async generator
        return gen()

        # params = self._default_params()
        # headers = {"User-Agent": "PAI Rag Client", "Authorization": f"{self.token}"}

        # if self.version == "1.0":
        #     pload = {"input_ids": prompt, **params}
        #     # make request
        #     async with httpx.AsyncClient() as http_client:
        #         response = await http_client.post(
        #             url=self.endpoint, headers=headers, json=pload, timeout=60
        #         )

        #     if response.status_code != 200:
        #         raise Exception(
        #             f"Request failed with status code {response.status_code}"
        #             f" and message {response.text}"
        #         )
        #     res = CompletionResponse(text=response.text)

        #     # yield text, if any
        #     yield res
        # else:
        #     pload = {"prompt": prompt, "use_stream_chat": "True", **params}

        #     async with httpx.AsyncClient() as http_client:
        #         # response = await http_client.post(
        #         #     url=self.endpoint, headers=headers, json=pload
        #         # )
        #         async with http_client.stream(
        #             "POST", url=self.endpoint, headers=headers, json=pload
        #         ) as response:
        #             async for chunk in response.aiter_lines():
        #                 print("chunk", type(chunk), chunk)
        #                 yield CompletionResponse(text="None")

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        async def gen() -> CompletionResponseAsyncGen:
            for message in self.stream_chat(prompt, **kwargs):
                yield message

        # NOTE: convert generator to async generator
        return gen()

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        async def gen() -> ChatResponseAsyncGen:
            for message in self.stream_chat(messages, **kwargs):
                yield message

        # NOTE: convert generator to async generator
        return gen()
