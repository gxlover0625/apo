from ctm import CorrectTemplateMemory
from epm import ErrorPatternMemory

class Updater:
    def __init__(self, correct_template_memory:CorrectTemplateMemory, error_pattern_memory:ErrorPatternMemory):
        self.ctm = correct_template_memory
        self.epm = error_pattern_memory
    
    def update_correct_memory(self, used_templates, question:str=None, ground_truth:str=None, correct_pred:str=None, reflections:list=None, client=None, eval_fn=None, init_instruction:str=None, output_format:str=None, *args, **kwargs):
        self.ctm.update(used_templates, question, ground_truth, correct_pred, reflections, client, eval_fn=eval_fn, init_instruction=init_instruction, output_format=output_format, *args, **kwargs)

    def update_error_memory(self, question:str, ground_truth:str, wrong_pred:str, reflection:str=None, client=None, *args, **kwargs):
        self.epm.update(question, ground_truth, wrong_pred, reflection, client, *args, **kwargs)