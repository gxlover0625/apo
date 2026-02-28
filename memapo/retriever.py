from ctm import CorrectTemplateMemory
from memapo.epm import ErrorPatternMemory

class Retriever:
    def __init__(self, correct_template_memory:CorrectTemplateMemory, error_pattern_memory:ErrorPatternMemory):
        self.ctm = correct_template_memory
        self.epm = error_pattern_memory
    
    def retrieve(self, question:str, *args, **kwargs):
        # TODO
        pass