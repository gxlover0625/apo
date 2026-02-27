from dataclasses import dataclass
from typing import List
from uuid import uuid4

from storage import VectorStore

@dataclass
class GoodCase:
    question: str
    ground_truth: str
    correct_pred: str

class Template:
    def __init__(self, description:str, strategy:str, good_cases:List[GoodCase]):
        self.idx = str(uuid4())
        self.description = description
        self.strategy = strategy
        self.good_cases = good_cases
    
    def update(self, *args, **kwargs):
        pass

class CorrectTemplateMemory:
    def __init__(self, restore_path:str="./db", collection_name:str=None, emb_model:str=None, threshold:float=0.7, topk:int=3):
        self.all_templates = []
        self.db = VectorStore(restore_path, collection_name, emb_model, threshold, topk)
    
    def add_template(self, template:Template):
        self.all_templates.append(template)
        # TODO
        pass