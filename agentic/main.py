import os
import json
import random
import numpy as np
import re
from openai import OpenAI

from prompts.summarize import summarize_prompt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def call_llm(prompt, model, temperature=0):
    client = OpenAI(
        base_url=os.environ['OPENAI_BASE_URL'],
        api_key=os.environ['OPENAI_API_KEY']
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        seed=42
    )
    return response.choices[0].message.content

# copy from
# https://github.com/open-compass/opencompass/blob/b54e28c1db039e962987c31116e6c6d0c3906a14/opencompass/datasets/bbh.py#L32C1-L44C15
def bbh_mcq_postprocess(text: str) -> str:
    ans = text
    ans_line = ans.split('answer is ')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    match = re.search(r'\(([A-Z])\)*', ans)
    if match:
        return match.group(1)
    match = re.search(r'([A-Z])', ans)
    if match:
        return match.group(1)
    return ans

def bbh_mcq_eval_fn(prediction: str, ground_truth_answer: str):
    pred = bbh_mcq_postprocess(prediction)
    ref = bbh_mcq_postprocess(ground_truth_answer)
    return int(pred == ref)

def load_data(dataset_name, data_path):
    if dataset_name == "Logical_Deduction":
        set_seed()
        with open(data_path, 'r') as f:
            data = json.load(f)['examples']
        random.shuffle(data)
        train_set = data[:50]
        val_set = data[50:150]
        test_set = data[150:]
        initial_prompt = """A logical deduction task which requires deducing the order of a sequence of objects."""
        eval_fn = bbh_mcq_eval_fn
        return train_set, val_set, test_set, initial_prompt, eval_fn
    else:
        pass

class Prototype:
    def __init__(self, context, solution_steps, demos):
        self.context = context
        self.solution_steps = solution_steps
        self.demos = demos

train_set, val_set, test_set, initial_prompt, eval_fn = load_data("Logical_Deduction", "/root/shared-nvme/apo/opro/data/BIG-Bench-Hard-data/logical_deduction_seven_objects.json")
first_question = train_set[0]['input']
first_gt = train_set[0]['target']
output_format = """You must give your final answer by starting with 'So the answer is'"""
full_prompt = f"{initial_prompt}\n{first_question}\n{output_format}"
first_rep = call_llm(full_prompt, "Qwen3-Next-80B-A3B-Instruct", 0)
# print(first_rep)
is_correct = eval_fn(first_rep, first_gt)
print(is_correct)

if is_correct:
    summarize_prompt_full = summarize_prompt.format(
        question=first_question,
        reasoning_trajectory=first_rep
    )
    print(summarize_prompt_full)
    summary = call_llm(summarize_prompt_full, "Qwen3-Next-80B-A3B-Instruct", 0)
    print(summary)
    result_dict = json.loads(summary)
    print(result_dict['context'])
    print(result_dict['solution_steps'])
    