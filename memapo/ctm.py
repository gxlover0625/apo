from dataclasses import dataclass
from typing import List
from uuid import uuid4

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
    pass