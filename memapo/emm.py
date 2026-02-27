from dataclasses import dataclass
from typing import List
from uuid import uuid4

@dataclass
class BadCase:
    question: str
    ground_truth: str
    wrong_pred: str

class ErrorMode:
    def __init__(self, description:str, bad_cases:List[BadCase]):
        self.idx = str(uuid4())
        self.description = description
        self.bad_cases = bad_cases
    
    def update(self, *args, **kwargs):
        pass
        
class ErrorModeMemory:
    def __init__(self):
        self.all_bad_cases = []
        self.error_mode_clusters = []
        pass

    def add_bad_case(self, bad_case:BadCase):
        pass