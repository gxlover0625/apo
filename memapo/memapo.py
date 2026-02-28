from retriever import Retriever
from evaluator import Evaluator
from updater import Updater

class MemAPO:
    def __init__(self, retriever:Retriever, evaluator:Evaluator, updater:Updater):
        self.retriever = retriever
        self.evaluator = evaluator
        self.updater = updater
        pass

    def process_sample(self, sample, *args, **kwargs):
        # Stage-1: Memory Retrieve
        question, ground_truth = sample
        templates, error_patterns = self.retriever.retrieve(question, *args, **kwargs)
        # TODO