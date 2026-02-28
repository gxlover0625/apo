from ctm import CorrectTemplateMemory
from emm import ErrorModeMemory

class Retriever:
    def __init__(self, correct_template_memory:CorrectTemplateMemory, error_mode_memory:ErrorModeMemory):
        self.ctm = correct_template_memory
        self.emm = error_mode_memory
    
    def retrieve(self, question:str, *args, **kwargs):
        # TODO
        pass