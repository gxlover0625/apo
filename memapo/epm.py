from dataclasses import dataclass, field
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
    idx: str = field(default_factory=lambda: get_id(prefix="bad_case"))

class ErrorPattern:
    def __init__(self, pattern:str, bad_cases:Set[BadCase]):
        self.idx = get_id(prefix="error_pattern")
        self.pattern = pattern
        self.bad_cases = bad_cases
    
    def update(self, *args, **kwargs):
        pass
        
class ErrorPatternMemory:
    def __init__(self, restore_path:str="./db", collection_name:str=None, emb_model:str=None, threshold:float=None, topk:int=None):
        self.all_bad_cases:Dict[str, BadCase] = {}
        self.error_pattern_clusters = {}
        self.db = VectorStore(restore_path, collection_name, emb_model, threshold, topk)

    def add_bad_case(self, bad_case:BadCase):
        bad_case_id = bad_case.idx
        self.all_bad_cases[bad_case_id] = bad_case
        retrieved_results = self.db.query_topk_threshold(query=bad_case.reflection)
        if len(retrieved_results) == 0:
            # TODO 直接拿反思作为簇的描述，先这样写吧，看后续会不会改
            new_pattern_description = bad_case.reflection
            new_bad_cases = set(bad_case)
            new_error_pattern = ErrorPattern(new_pattern_description, new_bad_cases)
            self.add_error_pattern(new_error_pattern)
        else:
            matched_pattern_id = retrieved_results[0]["metadata"]["id"]
            matched_pattern = self.error_pattern_clusters[matched_pattern_id]
            matched_pattern.bad_cases.add(bad_case)
            # TODO 更新簇的描述，一定要实现
            self.update_error_pattern()

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

    def update_error_pattern(self, *args, **kwargs):
        # TODO 还未实现簇的更新
        pass

    def retrieve(self, question:str, *args, **kwargs)->List[ErrorPattern]:
        # 当前只考虑最简单的实现，召回所有的error pattern
        return list(self.error_pattern_clusters.values())