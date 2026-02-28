from retriever import Retriever
from evaluator import Evaluator
from updater import Updater
from llm import LLMFactory

class MemAPO:
    def __init__(
        self, 
        retriever:Retriever, 
        evaluator:Evaluator, 
        updater:Updater,
        model_name:str=None,
        temperature:float=0.,
        init_instruction:str=None,
        output_format:str=None,
        eval_fn=None,
        **kwargs
    ):
        self.retriever = retriever
        self.evaluator = evaluator
        self.updater = updater
        self.client = LLMFactory.get_llm(model_name, temperature)
        self.init_instruction = init_instruction
        self.output_format = output_format
        self.eval_fn = eval_fn

    def process_sample(self, sample, *args, **kwargs):
        # Stage-1: Memory Retrieve
        question, ground_truth = sample
        templates, error_patterns = self.retriever.retrieve(question, *args, **kwargs)