"""
Multimodal LLM client using OpenAI-compatible API.
Supports vision models for image and video understanding.
"""

from typing import List, Dict, Any, Optional
from openai import OpenAI, AsyncOpenAI
from loguru import logger


class OpenAIMultimodalLLM:
    """
    A multimodal LLM client that uses OpenAI-compatible API.
    Supports streaming internally but returns complete response.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
    ):
        """
        Initialize the MultimodalLLM client.
        
        Args:
            base_url: The base URL of the OpenAI-compatible API endpoint.
            api_key: The API key for authentication.
            model: The model name to use (e.g., 'gpt-4-vision-preview').
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        
        # Async client
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        
        # Sync client
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
    
    async def achat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat request with multimodal content (text, images, videos).
        Uses streaming internally but returns the complete response.
        
        Args:
            messages: List of message dicts with role and content.
                     Content can include text, image_url, etc.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            
        Returns:
            The complete response content as a string.
            
        Example messages format:
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                    ]
                }
            ]
        """
        try:
            # Build request kwargs
            request_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }
            
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens
            
            # Make streaming request
            stream = await self.async_client.chat.completions.create(**request_kwargs)
            
            # Collect all chunks into complete response
            full_response = ""
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        full_response += delta.content
            
            return full_response
            
        except Exception as e:
            logger.error(f"MultimodalLLM achat error: {e}")
            raise
    
    async def achat_with_images(
        self,
        image_urls: List[str],
        user_prompt: str = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Convenience method for chat with images.
        
        Args:
            user_prompt: The user's text prompt.
            image_urls: List of image URLs or base64 data URLs.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            
        Returns:
            The complete response content as a string.
        """
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        # Build user message with text and images
        user_content = []

        if user_prompt:
            user_content.append({
                "type": "text",
                "text": user_prompt,
            })
        
        for image_url in image_urls:
            # Support both URL and base64 formats
            if image_url.startswith("data:") or image_url.startswith("http"):
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            else:
                # Assume it's base64 without prefix, add it
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_url}"}
                })
        
        if not user_content:
            logger.warning(f"No user content provided for chat with images.")
            return None
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return await self.achat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    async def achat_with_video(
        self,
        video_urls: List[str],
        user_prompt: str = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Async convenience method for chat with videos.
        
        Args:
            user_prompt: The user's text prompt.
            video_urls: List of video URLs or base64 data URLs.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            
        Returns:
            The complete response content as a string.
        """
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        # Build user message with text and videos
        user_content = []

        if user_prompt:
            user_content.append({
                "type": "text",
                "text": user_prompt,
            })
        
        for video_url in video_urls:
            # Support both URL and base64 formats
            if video_url.startswith("data:") or video_url.startswith("http"):
                user_content.append({
                    "type": "video_url",
                    "video_url": {"url": video_url}
                })
            else:
                # Assume it's base64 without prefix, add it
                user_content.append({
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_url}"}
                })
        
        if not user_content:
            logger.warning(f"No user content provided for chat with videos.")
            return None
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return await self.achat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Sync version: Send a chat request with multimodal content.
        Uses streaming internally but returns the complete response.
        
        Args:
            messages: List of message dicts with role and content.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            
        Returns:
            The complete response content as a string.
        """
        try:
            # Build request kwargs
            request_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }
            
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens
            
            # Make streaming request
            stream = self.client.chat.completions.create(**request_kwargs)
            
            # Collect all chunks into complete response
            full_response = ""
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        full_response += delta.content
            
            return full_response
            
        except Exception as e:
            logger.error(f"OpenAIMultimodalLLM chat error: {e}")
            raise
    
    def chat_with_images(
        self,
        image_urls: List[str],
        user_prompt: str = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Sync version: Convenience method for chat with images.
        
        Args:
            user_prompt: The user's text prompt.
            image_urls: List of image URLs or base64 data URLs.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            
        Returns:
            The complete response content as a string.
        """
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        user_content = []

        if user_prompt:
            user_content.append({
                "type": "text",
                "text": user_prompt,
            })
        
        for image_url in image_urls:
            # Support both URL and base64 formats
            if image_url.startswith("data:") or image_url.startswith("http"):
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            else:
                # Assume it's base64 without prefix, add it
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_url}"}
                })
        
        if not user_content:
            logger.warning(f"No user content provided for chat with images.")
            return None
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def chat_with_video(
        self,
        video_urls: List[str],
        user_prompt: str = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Sync convenience method for chat with videos.
        
        Args:
            user_prompt: The user's text prompt.
            video_urls: List of video URLs or base64 data URLs.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            
        Returns:
            The complete response content as a string.
        """
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        user_content = []

        if user_prompt:
            user_content.append({
                "type": "text",
                "text": user_prompt,
            })
        
        for video_url in video_urls:
            # Support both URL and base64 formats
            if video_url.startswith("data:") or video_url.startswith("http"):
                user_content.append({
                    "type": "video_url",
                    "video_url": {"url": video_url}
                })
            else:
                # Assume it's base64 without prefix, add it
                user_content.append({
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_url}"}
                })
        
        if not user_content:
            logger.warning(f"No user content provided for chat with videos.")
            return None
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

