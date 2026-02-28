from dataclasses import dataclass
from typing import List, Set, Dict, Any
from uuid import uuid4

from utils import get_id, get_timestamp
from storage import VectorStore

@dataclass
class BadCase:
    question: str
    ground_truth: str
    wrong_pred: str
    reflection: str = None

class ErrorPattern:
    def __init__(self, pattern:str, bad_cases:List[BadCase], metadata:Dict[str, Any]=None):
        self.idx = get_id(prefix="error_pattern")
        self.pattern = pattern
        self.bad_cases = bad_cases
    
    def update(self, *args, **kwargs):
        pass
        
class ErrorPatternMemory:
    def __init__(self, restore_path:str="./db", collection_name:str=None, emb_model:str=None, threshold:float=None, topk:int=None):
        self.all_bad_cases:Set[BadCase] = set()
        self.error_pattern_clusters = {}
        self.db = VectorStore(restore_path, collection_name, emb_model, threshold, topk)

    def add_bad_case(self, bad_case:BadCase):
        self.all_bad_cases.add(bad_case)
        # TODO

    def add_error_pattern(self, error_pattern:ErrorPattern):
        doc_id = error_pattern.idx
        doc_content = error_pattern.pattern
        doc_metadata = {
            "id": doc_id,
            "timestamp": get_timestamp(),
            "type": "error_pattern",
            "pattern": error_pattern.pattern,
        }
        self.db.add(doc_id, doc_content, doc_metadata)
        self.error_pattern_clusters[doc_id] = error_pattern 

    def retrieve(self, question:str, *args, **kwargs)->List[ErrorPattern]:
        # 当前只考虑最简单的实现，召回所有的error pattern
        return list(self.error_pattern_clusters.values())