import os
import json
import random

from abc import ABC, abstractmethod
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import Dict, Any, Optional

from utils import get_logger

class LLMProvider(ABC):
    @retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=20))
    def generate(self, *args, **kwargs):
        return self._generate(*args, **kwargs)

    @abstractmethod
    def _generate(self, *args, **kwargs):
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

    def _generate(self, user_prompt, sys_prompt=None, **kwargs):
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
        if not int(os.environ['disable_logging']):
            logger = get_logger()
            all_request_params = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                **self.extra_params,
                **kwargs
            }
            logger.info("LLM Request Params:\n%s", json.dumps(all_request_params, indent=2, ensure_ascii=False, sort_keys=True))
            logger.info("LLM Response:\n%s", response.choices[0].message.content)
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

    def _generate(self, user_prompt, sys_prompt=None, **kwargs):
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
        if not int(os.environ['disable_logging']):
            logger = get_logger()
            all_request_params = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                **self.extra_params,
                **kwargs
            }
            logger.info("LLM Request Params:\n%s", json.dumps(all_request_params, indent=2, ensure_ascii=False, sort_keys=True))
            logger.info("LLM Response:\n%s", response)
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

    def _generate(self, user_prompt, sys_prompt=None, **kwargs):
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
        if not int(os.environ['disable_logging']):
            logger = get_logger()
            all_request_params = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                **self.extra_params,
                **kwargs
            }
            logger.info("LLM Request Params:\n%s", json.dumps(all_request_params, indent=2, ensure_ascii=False, sort_keys=True))
            logger.info("LLM Response:\n%s", response.choices[0].message.content)
        return response.choices[0].message.content

class LLMFactory:
    @staticmethod
    def get_llm(model_name:str, temperature:float=0.):
        if model_name.lower() in ["qwen3-8b"]:
            extra_params = {
                "frequency_penalty": 0.8,
                "presence_penalty": 0.3,
                "temperature": temperature,
                "max_tokens": 5000,
                "extra_body": {
                    "extendParams": {
                        "enable_thinking": False
                    }
                }
            }
            return OpenAIStreamProvider(model=model_name, extra_params=extra_params)
        elif model_name.lower() in ["qwen3-32b", "qwen3-14b"]:
            extra_params = {
                "frequency_penalty": 0.8,
                "presence_penalty": 0.3,
                "temperature": temperature,
                "max_tokens": 5000,
                "extend_fields": {
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            }
            return WhaleProvider(model=model_name, extra_params=extra_params)
        else:
            extra_params = {
                "temperature": temperature,
            }
            return OpenAIProvider(model=model_name, extra_params=extra_params)

if __name__ == "__main__":
    llm = LLMFactory.get_llm("qwen3-8b", temperature=0.7)
    print(llm.extra_params)
    print(llm.generate("请介绍一下自己。"))