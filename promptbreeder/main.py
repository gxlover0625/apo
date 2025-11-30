from pb import create_population, init_run, run_for_n
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles

import os
import logging
import argparse
import asyncio
import random

from dotenv import load_dotenv
from rich import print
# import cohere
from openai import OpenAI
from dataclasses import dataclass
from typing import List

import json
import re
import numpy as np
import time

# copy from
# https://github.com/open-compass/opencompass/blob/b54e28c1db039e962987c31116e6c6d0c3906a14/opencompass/datasets/bbh.py#L48C1-L62C15
def bbh_freeform_postprocess(text: str) -> str:
    ans = text
    ans_line = ans.split('answer is ')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    ans = ans.split('\n')[0].strip()

    if ans.endswith('.'):
        ans = ans[:-1].strip()

    match = re.search(r'\*\*(.*?)\*\*', ans)
    if match:
        return match.group(1)

    return ans

def bbh_freeform_eval_fn(prediction: str, ground_truth_answer: str):
    pred = bbh_freeform_postprocess(prediction)
    ref = ground_truth_answer
    return int(pred == ref)

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

def load_task(dataset_name:str, data_path: str):
    random.seed(42)
    np.random.seed(42)
    if dataset_name == "causal_judgement":
        with open(data_path, "r") as f:
            data = json.load(f)['examples']
        random.shuffle(data)
        train_data = data[:37]
        eval_data = data[37: 37+74]
        test_data = data[37+74:]
        eval_fn = bbh_freeform_eval_fn
        return train_data, eval_data, test_data, eval_fn
    elif dataset_name in ["logical_deduction_seven_objects", "geometric_shapes"]:
        with open(data_path, "r") as f:
            data = json.load(f)['examples']
        random.shuffle(data)
        train_data = data[:50]
        eval_data = data[50: 150]
        test_data = data[150:]
        eval_fn = bbh_mcq_eval_fn
        return train_data, eval_data, test_data, eval_fn
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
        
@dataclass
class Generation:
    text: str

class OpenAIWrapper:
    def __init__(self, base_url, api_key, model, *args, **kwargs):
        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    
    def generate(self, prompt:str, temperature:float=None, **kwargs) -> List[Generation]:
        if temperature is None:
            temperature = 0.7
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temperature,
                    stream=True,
                    seed=42
                )
                full_content = ""
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_content += content
                return [Generation(text=full_content)]
            except Exception as e:
                print(f"{attempt + 1} attempt failed: {e}")
                time.sleep(2 ** (attempt + 1))
        return [Generation(text="")]
    
    def batch_generate(self, prompts:List[str], temperature:float=None, **kwargs) -> List[List[Generation]]:
        if temperature is None:
            temperature = 0.7
        res_list = []
        for p in prompts:
            res_list.append(
                self.generate(p, temperature=temperature, **kwargs)
            )
        return res_list      

load_dotenv() # load environment variables

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Run the PromptBreeder Algorithm. Number of units is mp * ts.')
parser.add_argument('-mp', '--num_mutation_prompts', default=5, type=int)     
parser.add_argument('-ts', '--num_thinking_styles', default=5, type=int)     
parser.add_argument('-e', '--num_evals', default=10, type=int)     
parser.add_argument('-n', '--simulations', default=10, type=int)     
parser.add_argument('-p', '--problem', default="Solve the math word problem, giving your answer as an arabic numeral.")
parser.add_argument('-d', '--data', default="causal_judgement")
parser.add_argument('--path', type=str, required=True)       

args = vars(parser.parse_args())
train_set, eval_set, test_set, eval_fn = load_task(args['data'], args['path'])
os.environ['TASK'] = args['data']
args['num_evals'] = len(train_set)
# args['problem'] = """You must give your final answer by starting with 'So the answer is'"""
# args['problem'] = """Let's think step by step."""
if args['data'] == "geometric_shapes":
    args['problem'] = """Name geometric shapes from their SVG paths."""
elif args['data'] == "causal_judgement":
    args['problem'] = """Answer questions about causal attribution."""
elif args['data'] == "logical_deduction_seven_objects":
    args['problem'] = """A logical deduction task which requires deducing the order of a sequence of objects."""
else:
    raise ValueError(f"Unsupported data type: {args['data']}")

total_evaluations = args['num_mutation_prompts']*args['num_thinking_styles']*args['num_evals']

# set num_workers to total_evaluations so we always have a thread 
# co = cohere.Client(api_key=os.environ['COHERE_API_KEY'],  num_workers=total_evaluations, max_retries=5, timeout=30) #override the 2 min timeout with 30s. 
opt_model = OpenAIWrapper(
    base_url=os.environ['OPENAI_BASE_URL'],
    api_key=os.environ['OPENAI_API_KEY'],
    model=os.environ['OPENAI_OPT_MODEL']
)

task_model = OpenAIWrapper(
    base_url=os.environ['OPENAI_BASE_URL'],
    api_key=os.environ['OPENAI_API_KEY'],
    model=os.environ['OPENAI_TASK_MODEL']
)

# tp_set = mutation_prompts[:int(args['num_mutation_prompts'])]
tp_set = random.sample(thinking_styles, int(args['num_thinking_styles']))
# mutator_set= thinking_styles[:int(args['num_thinking_styles'])]
mutator_set = random.sample(mutation_prompts, int(args['num_mutation_prompts']))

logger.info(f'You are prompt-optimizing for the problem: {args["problem"]}')

logger.info(f'Creating the population...')
p = create_population(tp_set=tp_set, mutator_set=mutator_set, problem_description=args['problem'])

logger.info(f'Generating the initial prompts...')
init_run(p, opt_model, task_model, int(args['num_evals']), train_set, eval_fn)

logger.info(f'Starting the genetic algorithm...')
run_for_n(n=int(args['simulations']), population=p, opt_model=opt_model, task_model=task_model, num_evals=int(args['num_evals']), train_set=train_set, eval_fn=eval_fn)

print("%"*80)
print("done processing! final gen:")
# print(p.units)
print(p.elites)
