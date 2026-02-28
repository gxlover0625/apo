from dataclasses import dataclass, field
from typing import List
from uuid import uuid4

from storage import VectorStore
from utils import get_id, get_timestamp

@dataclass
class GoodCase:
    question: str
    ground_truth: str
    correct_pred: str
    idx: str = field(default_factory=lambda: get_id(prefix="good_case"))

class Template:
    def __init__(self, when_to_use:str, strategy:str, good_cases:List[GoodCase]):
        self.idx = get_id(prefix="template")
        self.when_to_use = when_to_use
        self.strategy = strategy
        self.good_cases = good_cases
    
    def update(self, *args, **kwargs):
        pass

class CorrectTemplateMemory:
    def __init__(self, restore_path:str="./db", collection_name:str=None, emb_model:str=None, threshold:float=0.7, topk:int=3):
        self.all_templates = {}
        self.db = VectorStore(restore_path, collection_name, emb_model, threshold, topk)
    
    def add_template(self, template:Template):
        doc_id = template.idx
        doc_content = template.when_to_use
        doc_metadata = {
            "id": doc_id,
            "timestamp": get_timestamp(),
            "type": "template",
            "when_to_use": template.when_to_use,
            "strategy": template.strategy,
        }
        self.db.add(doc_id, doc_content, doc_metadata)
        self.all_templates[doc_id] = template

    def retrieve(self, question:str, *args, **kwargs)->List[Template]:
        # threshold, topk的参数是在初始化向量数据库的时候传入的
        retrieved_results = self.db.query_topk_threshold(query=question)
        if len(retrieved_results) == 0:
            # TODO
            return []
        template_ids = [res["metadata"]["id"] for res in retrieved_results]
        templates = [self.all_templates[tid] for tid in template_ids if tid in self.all_templates]
        return templates
    
    def update(self, *args, **kwargs):
        # TODO 还未实现模板的更新
        pass