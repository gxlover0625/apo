import os
import random

from abc import ABC, abstractmethod
from openai import OpenAI
from typing import Dict, Any, Optional

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, base_url=None, api_key=None, model=None, extra_params:Optional[Dict[str, Any]]=None):
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", None)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", None)
        self.model = model
        self.extra_params = extra_params or {}
        assert self.api_key is not None, "Please set OPENAI_API_KEY environment variable"
        assert self.base_url is not None, "Please set OPENAI_BASE_URL environment variable"
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate(self, user_prompt, sys_prompt=None, **kwargs):
        if sys_prompt is None:
            messages = [{"role": "user", "content": user_prompt}]
        else:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            **self.extra_params,
            **kwargs
        )
        return response.choices[0].message.content
    
class OpenAIStreamProvider(LLMProvider):
    def __init__(self, base_url=None, api_key=None, model=None, extra_params:Optional[Dict[str, Any]]=None):
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", None)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", None)
        self.model = model
        self.extra_params = extra_params or {}
        assert self.api_key is not None, "Please set OPENAI_API_KEY environment variable"
        assert self.base_url is not None, "Please set OPENAI_BASE_URL environment variable"
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate(self, user_prompt, sys_prompt=None, **kwargs):
        if sys_prompt is None:
            messages = [{"role": "user", "content": user_prompt}]
        else:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        stream_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **self.extra_params,
            **kwargs
        )
        response = ""
        for chunk in stream_response:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        return response

class WhaleProvider(LLMProvider):
    def __init__(self, base_url=None, api_key=None, model=None, extra_params:Optional[Dict[str, Any]]=None):
        api_key = api_key or os.getenv("WHALE_API_KEY", None)
        assert api_key is not None, "Please set WHALE_API_KEY environment variable"
        self.keys = api_key.split(",")

        self.base_url = base_url or os.getenv("WHALE_BASE_URL", None)
        assert self.base_url is not None, "Please set WHALE_BASE_URL environment variable"

        self.model = model
        self.extra_params = extra_params or {}

    def generate(self, user_prompt, sys_prompt=None, **kwargs):
        from whale import TextGeneration
        select_key = random.choice(self.keys)
        TextGeneration.set_api_key(select_key, base_url=self.base_url)
        if sys_prompt is None:
            messages = [{"role": "user", "content": user_prompt}]
        else:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        response = TextGeneration.chat(
            model=self.model,
            messages=messages,
            stream=False,
            **self.extra_params,
            **kwargs
        )
        return response.choices[0].message.content

class LLMFactory:
    @staticmethod
    def get_provider(model_name:str):
        pass

if __name__ == "__main__":
    provider = OpenAIStreamProvider(model="Qwen3-Next-80B-A3B-Instruct")
    response = provider.generate("Please tell me a story.")
    print(response)