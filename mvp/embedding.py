import os
import numpy as np
import chromadb

from openai import OpenAI
from chromadb import Documents, EmbeddingFunction, Embeddings

def call_emb(model, texts):
    if isinstance(texts, str):
        texts = [texts]
    client = OpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    response = client.embeddings.create(
        model=model,
        input=texts,
    )
    return response.data # list, item.embedding

class MyCustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name):
        self.model = model_name
        self.client = OpenAI(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
        )
    
    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.embeddings.create(
            model=self.model,
            input=input,
        )
        raw_embeddings = np.array([item.embedding for item in response.data])
        norm = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        norm_embeddings = raw_embeddings / np.where(norm == 0, 1, norm)
        return norm_embeddings.tolist()

def get_db(restore_path="./db", collection_name="default", emb_model=None, inference=False):
    client = chromadb.PersistentClient(path=restore_path)
    custom_ef = MyCustomEmbeddingFunction(model_name=emb_model)
    if inference:
        collection = client.get_collection(
            name=collection_name, 
            embedding_function=custom_ef,
        )
    else:
        collection = client.create_collection(
            name=collection_name, 
            embedding_function=custom_ef,
            metadata={"hnsw:space": "cosine"}
        )
    return collection

def query_topk_threshold(collection, query:str, topk=3, threshold=0.5):
    res = collection.query(
        query_texts=[query],
        n_results=20,
        include=["documents", "distances", "metadatas"]
    )
    docs = res["documents"][0]
    distances = res["distances"][0]
    metas = res["metadatas"][0]

    results = []
    for doc, dist, meta in zip(docs, distances, metas):
        sim = 1 - dist
        if sim >= threshold:
            results.append({
                "document": doc,
                "similarity": sim,
                "metadata": meta
            })
    return results[:topk]

def add_single_doc(collection, doc_content:str, doc_id:str, doc_metadata:dict=None):
    collection.add(
        documents=[doc_content],
        ids=[doc_id],
        metadatas=[doc_metadata]
    )

if __name__ == "__main__":
    emb_model = "GLM-Embedding-3"
    # texts = "hello"
    # print(len(call_emb(emb_model, texts)[0].embedding))

    func = MyCustomEmbeddingFunction(model_name=emb_model)
    print(func(['hi']))
