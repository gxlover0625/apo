from retriever import Retriever
from evaluator import Evaluator
from updater import Updater
from llm import LLMFactory
from utils import extract_json
from prompts import (
    build_generation_sys_prompt, 
    build_generation_user_prompt,
    build_reflection_sys_prompt,
    build_reflection_user_prompt,
    build_summarize_reflection_sys_prompt,
    build_summarize_reflection_user_prompt,
)

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
        max_attempts:int=3,
        **kwargs
    ):
        self.retriever = retriever
        self.evaluator = evaluator
        self.updater = updater
        self.client = LLMFactory.get_llm(model_name, temperature)
        self.init_instruction = init_instruction
        self.output_format = output_format
        self.eval_fn = eval_fn
        self.max_attempts = max_attempts

    def process_sample(self, sample, *args, **kwargs):
        # Stage-1: Memory Retrieve
        question, ground_truth = sample
        templates, error_patterns = self.retriever.retrieve(question, *args, **kwargs)

        # Stage-2: Evaluate Prompt
        gen_sys_prompt = build_generation_sys_prompt(self.init_instruction, error_patterns)
        reflect_sys_prompt = build_reflection_sys_prompt()
        reflections = []
        correct_update = False
        for attempt in range(1, self.max_attempts + 1):
            gen_user_prompt = build_generation_user_prompt(
                question, self.output_format, templates, reflections
            )
            pred = self.client.generate(gen_user_prompt, gen_sys_prompt)
            is_correct = self.eval_fn(pred, ground_truth)
            if is_correct:
                correct_update = True
                break

            reflection_user_prompt = build_reflection_user_prompt(question, pred, reflections)
            reflection_raw = self.client.generate(reflection_user_prompt, reflect_sys_prompt)
            reflection_json = extract_json(reflection_raw)
            reflection_text = reflection_json["reflection"] if reflection_json else reflection_raw
            reflections.append({
                "attempt": attempt,
                "wrong_pred": pred,
                "reflection": reflection_text,
            })

        # Stage-3: Update Memory
        if correct_update:
            self.updater.update_correct_memory(
                used_templates=templates,
                question=question,
                ground_truth=ground_truth,
                correct_pred=pred,
                reflections=reflections,
                client=self.client,
                eval_fn=self.eval_fn,
                init_instruction=self.init_instruction,
                output_format=self.output_format,
            )
        else:
            sum_reflect_sys_prompt = build_summarize_reflection_sys_prompt()
            sum_reflect_user_prompt = build_summarize_reflection_user_prompt(
                question=question,
                ground_truth=ground_truth,
                reflections=reflections,
            )
            summary_raw = self.client.generate(sum_reflect_user_prompt, sum_reflect_sys_prompt)
            summary_json = extract_json(summary_raw)
            final_reflection = summary_json["reflection"] if summary_json else summary_raw

            self.updater.update_error_memory(
                question=question,
                ground_truth=ground_truth,
                wrong_pred=pred,
                reflection=final_reflection,
                client=self.client
            )