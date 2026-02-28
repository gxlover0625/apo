from ctm import CorrectTemplateMemory
from epm import ErrorPatternMemory

class Updater:
    def __init__(self, correct_template_memory:CorrectTemplateMemory, error_pattern_memory:ErrorPatternMemory):
        self.ctm = correct_template_memory
        self.epm = error_pattern_memory
    
    def update(self, *args, **kwargs):
        # TODO
        pass