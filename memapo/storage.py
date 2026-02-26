import chromadb
import os
import random
import numpy as np

from abc import ABC, abstractmethod
from chromadb import Documents, EmbeddingFunction, Embeddings
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI

def normalize_vec(vecs):
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm_vecs = vecs / np.where(norm == 0, 1.0, norm)
    return norm_vecs.tolist()

class EmbeddingProvider(ABC):
    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=10))
    def encode(self, input: Documents) -> Embeddings:
        return self._encode(input)

    @abstractmethod
    def _encode(self, input: Documents) -> Embeddings:
        pass

class OpenAIEmbedProvider(EmbeddingProvider):
    def __init__(self, base_url=None, api_key=None, model=None):
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        assert self.base_url is not None, "Please set OPENAI_BASE_URL environment variable"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        assert self.api_key is not None, "Please set OPENAI_API_KEY environment variable" 
        self.model = model

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
    
    def _encode(self, input: Documents) -> Embeddings:
        response = self.client.embeddings.create(
            model=self.model,
            input=input,
        )
        raw_embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
        norm_embeddings = normalize_vec(raw_embeddings)
        return norm_embeddings

class WhaleEmbedProvider(EmbeddingProvider):
    def __init__(self, base_url=None, api_key=None, model=None):
        self.base_url = base_url or os.getenv("WHALE_BASE_URL")
        assert self.base_url is not None, "Please set WHALE_BASE_URL environment variable"
        api_key = api_key or os.getenv("WHALE_API_KEY")
        assert api_key is not None, "Please set WHALE_API_KEY environment variable" 
        self.keys = api_key.split(",")

        self.model = model

    def _encode(self, input: Documents) -> Embeddings:
        from whale import TextGeneration
        select_key = random.choice(self.keys)
        TextGeneration.set_api_key(select_key, base_url=self.base_url)
        response = TextGeneration.embed(
            model=self.model,
            input=input,
            time=60
        ).to_dict()
        raw_embeddings = np.array([item["embedding"] for item in response["data"]], dtype=np.float32)
        norm_embeddings = normalize_vec(raw_embeddings)
        return norm_embeddings

class EmbedFunctionFactory:
    @staticmethod
    def get_embed_function(model_name:str):
        if "qwen3-embedding" in model_name.lower():
            return WhaleEmbedProvider(model=model_name)
        else:
            return OpenAIEmbedProvider(model=model_name)

if __name__ == "__main__":
    embed_func = EmbedFunctionFactory.get_embed_function("Doubao-Embedding-Large-Text")
    res = np.array(embed_func.encode(["hello world", "你好，世界", "重力加速度小g"]))
    print(res @ res.T)
