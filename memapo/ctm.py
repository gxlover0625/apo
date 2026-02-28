from dataclasses import dataclass
from typing import List
from uuid import uuid4

from storage import VectorStore
from utils import get_id, get_timestamp

@dataclass
class GoodCase:
    question: str
    ground_truth: str
    correct_pred: str

class Template:
    def __init__(self, description:str, strategy:str, good_cases:List[GoodCase]):
        self.idx = get_id(prefix="template")
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
        doc_id = template.idx
        doc_content = template.description
        doc_metadata = {
            "id": doc_id,
            "timestamp": get_timestamp(),
            "type": "template",
            "description": template.description,
            "strategy": template.strategy,
        }
        self.db.add(doc_id, doc_content, doc_metadata)

    def retrieve(self, question:str, *args, **kwargs)->List[Template]:
        pass