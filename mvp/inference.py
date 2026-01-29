import argparse
import os
import random
import numpy as np
import logging
import pickle

from uuid import uuid4
from pathlib import Path
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from textgrad.tasks import load_task

from task import get_eval_fn
from embedding import get_db, add_single_doc, query_topk_threshold
from agent import AnswerAgent, SummaryAgent
from prototype import Prototype, Demonstration, Strategy

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

@retry(stop=stop_after_attempt(10), wait=wait_random_exponential(multiplier=1, max=10))
def call_llm(user_prompt, sys_prompt="You are a helpful assistant.", model=None, temperature=0):
    client = OpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    
    is_qwen = 'qwen' in model.lower() if model else False
    if is_qwen:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            frequency_penalty=0.8,
            presence_penalty=0.3,
            stop=None,
            temperature=temperature,
            seed=42,
            stream=True,
            extra_body={"extendParams": {"enable_thinking": False}},
            max_tokens=5000,
        )
        
        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        return response
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content

# Environment setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
seed_everything()
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--model",type=str, required=True)
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--embed_model", type=str, required=True)
parser.add_argument("--db_path", type=str, required=True)
parser.add_argument("--dict_path", type=str, required=True)
parser.add_argument("--topk", type=int, default=1)
args = parser.parse_args()

# Data & Prompt setup
data_dir = Path(__file__).resolve().parent.parent / "data"
train_set, val_set, test_set, _ = load_task(args.dataset, evaluation_api=None, data_dir=data_dir)
eval_fn = get_eval_fn(args.dataset)
print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
if args.dataset in ["Geo_Group", "BBH_geometric_shapes", "bbeh_geometric_shapes"]:
    init_instruction = """Identify geometric shapes from their SVG paths."""
    output_format = "When you provide the final answer, please use the prefix \"The answer is:\" without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\""
elif args.dataset in ["Logical_Group", "BBH_logical_deduction_seven_objects", "bbeh_boardgame_qa"]:
    init_instruction = """Let's solve the problem."""
    output_format = "When you provide the final answer, please use the prefix \"The answer is:\" without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\""
elif args.dataset in ["gpqa"]:
    init_instruction = """Let's solve the problem."""
    output_format = f"Format your response as follows: \"The correct answer is (insert answer here)\""
elif args.dataset in ["agieval_aqua", "agieval_gaokao_math", "agieval_sat", "math_group", "gaokao_group"]:
    init_instruction = """Let's solve the problem."""
    output_format = f"Format your response as follows: \"The correct answer is (insert answer here)\""
elif args.dataset in ["agieval_gaokao_history", "agieval_gaokao_chinese", "agieval_gaokao_geography", "human_group"]:
    init_instruction = """Let's solve the problem."""
    output_format = f"Format your response as follows: \"The correct answer is (insert answer here)\""

# Training setup
with open(f"{args.dict_path}", "rb") as f:
    prototype_dict = pickle.load(f)
db = get_db(collection_name=f"{args.db_path}", emb_model=args.embed_model, inference=True)
ans_agent = AnswerAgent(model=args.model, temperature=0., call_fn=call_llm)
sum_agent = SummaryAgent(model=args.model, temperature=0., call_fn=call_llm)

def process_single_sample(train_sample):
    question, gt = train_sample
    retrieve_prototypes_data = query_topk_threshold(
        db, query=question, topk=args.topk, threshold=0.
    )
    if len(retrieve_prototypes_data) == 0:
        #### reflexion
        is_success, prediction = ans_agent.answer_without_reflection(
            instruction=init_instruction,
            output_format=output_format,
            question=question,
            gt=gt,
            eval_fn=eval_fn
        )
    else:
        chosen_prototypes = []
        for item in retrieve_prototypes_data:
            pid = item["metadata"]["prototype_id"]
            chosen_prototypes.append(prototype_dict[pid])

        if args.topk == 1:
            # chosen_id = retrieve_prototype[0]["metadata"]["prototype_id"]
            # chosen_prototype = prototype_dict[chosen_id]
            chosen_prototype = chosen_prototypes[0]
            is_success, final_response = ans_agent.answer_with_prototype_v2(
                instruction=init_instruction,
                output_format=output_format,
                question=question,
                gt=gt,
                eval_fn=eval_fn,
                prototype=chosen_prototype
            )
        else:
            is_success, final_response = ans_agent.answer_with_topk_prototypes(
                instruction=init_instruction,
                output_format=output_format,
                question=question,
                gt=gt,
                eval_fn=eval_fn,
                prototypes=chosen_prototypes
            )

    return 1 if is_success else 0

## Start Training
correct_count = 0
results = []
lock = Lock()
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_single_sample, sample) for sample in test_set]
    pbar = tqdm(as_completed(futures), total=len(futures))
    for future in pbar:
        result = future.result()
        with lock:
            results.append(result)
            correct_count += result
            current_acc = correct_count / len(results)
        pbar.set_description(f"acc: {current_acc:.4f}")
