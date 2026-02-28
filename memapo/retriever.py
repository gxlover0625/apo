from ctm import CorrectTemplateMemory
from epm import ErrorPatternMemory

class Retriever:
    def __init__(self, correct_template_memory:CorrectTemplateMemory, error_pattern_memory:ErrorPatternMemory):
        self.ctm = correct_template_memory
        self.epm = error_pattern_memory
    
    def retrieve(self, question:str, *args, **kwargs):
        retrieved_templates = self.ctm.retrieve(question, *args, **kwargs)
        retrieved_error_patterns = self.epm.retrieve(question, *args, **kwargs)
        return retrieved_templates, retrieved_error_patterns