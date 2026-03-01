import chromadb
import os
import random
import numpy as np

from abc import ABC, abstractmethod
from chromadb import Documents, EmbeddingFunction, Embeddings
from uuid import uuid4
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI

from utils import get_logger, get_timestamp

def normalize_vec(vecs):
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm_vecs = vecs / np.where(norm == 0, 1.0, norm)
    return norm_vecs.tolist()

class EmbeddingProvider(ABC):
    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=10))
    def encode(self, input: Documents) -> Embeddings:
        if not int(os.environ["disable_logging"]):
            logger = get_logger()
            logger.info(f"Embedding input:\n{input}")
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

class MyCustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name):
        self.embed_function = EmbedFunctionFactory.get_embed_function(model_name)
    
    def __call__(self, input: Documents) -> Embeddings:
        return self.embed_function.encode(input)

class VectorStore:
    def __init__(self, restore_path="./db", collection_name="default", emb_model=None, threshold=0.7, topk=3):
        self.client = chromadb.PersistentClient(path=restore_path)
        self.ef = MyCustomEmbeddingFunction(model_name=emb_model)
        self.collection_name = collection_name
        self.threshold = threshold
        self.topk = topk
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )
        if not int(os.environ["disable_logging"]):
            self.logger = get_logger()
            self.logger.info(f"Loading Vectore Store\nDB Path: {restore_path}\nCollection: {collection_name}\nEmbedding Model: {emb_model}\nThreshold: {threshold}\nTopK: {topk}")
    
    def add(self, doc_id=None, doc_content=None, doc_metadata=None):
        if doc_id is None:
            doc_id = str(uuid4())
        assert doc_content is not None, "Document cannot be None"
        if doc_metadata is None:
            doc_metadata = {"id": doc_id, "timestamp": get_timestamp(), "content": doc_content}
        self.collection.add(
            ids=[doc_id],
            documents=[doc_content],
            metadatas=[doc_metadata]
        )
        return doc_id

    def delete(self, doc_id):
        self.collection.delete(ids=[doc_id])

    def update(self, doc_id, doc_content, doc_metadata=None):
        if doc_metadata is None:
            doc_metadata = {"id": doc_id, "content": doc_content, "updated_timestamp": get_timestamp()}
        self.collection.update(
            ids=[doc_id],
            documents=[doc_content],
            metadatas=[doc_metadata],
        )
        return doc_id
    
    # 纯top-k检索, 返回相似度最高的k条结果
    def query_topk(self, query:str, **kwargs):
        res = self.collection.query(
            query_texts=[query],
            n_results=min(self.topk*2, 20),
            include=["documents", "distances", "metadatas"]
        )
        docs = res["documents"][0]
        distances = res["distances"][0]
        metas = res["metadatas"][0]

        results = []
        for doc, dist, meta in zip(docs, distances, metas):
            sim = 1 - dist
            results.append({
                "document": doc,
                "similarity": sim,
                "metadata": meta
            })
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)
        return results[:self.topk]
    
    # 先过滤掉低相似度，再去k条最相似的
    def query_topk_threshold(self, query:str, **kwargs):
        res = self.collection.query(
            query_texts=[query],
            n_results=min(self.topk*2, 20),
            include=["documents", "distances", "metadatas"]
        )
        docs = res["documents"][0]
        distances = res["distances"][0]
        metas = res["metadatas"][0]

        results = []
        for doc, dist, meta in zip(docs, distances, metas):
            sim = 1 - dist
            if sim >= self.threshold:
                results.append({
                    "document": doc,
                    "similarity": sim,
                    "metadata": meta
                })
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)
        return results[:self.topk]

if __name__ == "__main__":
    embed_func = EmbedFunctionFactory.get_embed_function("Doubao-Embedding-Large-Text")
    res = np.array(embed_func.encode(["hello world", "你好，世界", "重力加速度小g"]))
    print(res @ res.T)
