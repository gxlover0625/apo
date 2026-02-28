from dataclasses import dataclass
from typing import List, Set, Dict, Any
from uuid import uuid4

from utils import get_id, get_timestamp

@dataclass
class BadCase:
    question: str
    ground_truth: str
    wrong_pred: str
    reflection: str = None

class ErrorMode:
    def __init__(self, description:str, bad_cases:List[BadCase], metadata:Dict[str, Any]=None):
        self.idx = get_id(prefix="error_mode")
        self.description = description
        self.bad_cases = bad_cases
        self.metadata = metadata or {}
    
    def update(self, *args, **kwargs):
        pass
        
class ErrorModeMemory:
    def __init__(self):
        self.all_bad_cases:Set[BadCase] = set()
        self.error_mode_clusters = []
        pass

    def add_bad_case(self, bad_case:BadCase):
        self.all_bad_cases.add(bad_case)
        # TODO

    def add_error_mode(self):
        pass

    def retrieve(self, *args, **kwargs)->List[ErrorMode]:
        pass