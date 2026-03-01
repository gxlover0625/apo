from ctm import CorrectTemplateMemory
from epm import ErrorPatternMemory, BadCase

class Updater:
    def __init__(self, correct_template_memory:CorrectTemplateMemory, error_pattern_memory:ErrorPatternMemory):
        self.ctm = correct_template_memory
        self.epm = error_pattern_memory
    
    def update_correct_memory(self, used_templates, *args, **kwargs):
        # TODO
        pass

    def update_error_memory(self, question:str, ground_truth:str, wrong_pred:str, reflection:str=None, *args, **kwargs):
        bad_case = BadCase(
            question=question,
            ground_truth=ground_truth,
            wrong_pred=wrong_pred,
            reflection=reflection,
        )
        self.epm.add_bad_case(bad_case)