import argparse
import os
import random
import numpy as np

from uuid import uuid4
from pathlib import Path
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from textgrad.tasks import load_task

from task import get_eval_fn
from embedding import get_db, add_single_doc, query_topk_threshold
from agent import AnswerAgent, SummaryAgent
from prototype import Prototype, Demonstration

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=4))
def call_llm(prompt=None, model=None, temperature=0):
    client = OpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content

seed_everything()
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
args = parser.parse_args()

data_dir = Path(__file__).resolve().parent.parent / "data"
train_set, val_set, test_set, _ = load_task(args.dataset, evaluation_api=None, data_dir=data_dir)
eval_fn = get_eval_fn(args.dataset)
print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))

db = get_db(collection_name="mvp_test_main_02", emb_model="Doubao-Embedding-Large-Text")
print(db)

ans_agent = AnswerAgent(model="Qwen3-235B-A22B-Instruct-2507", temperature=0., call_fn=call_llm)
sum_agent = SummaryAgent(model="Qwen3-235B-A22B-Instruct-2507", temperature=0., call_fn=call_llm)
# print(ans_agent("请问你是谁?"))

## Before Training
prototype_dict = {}
if args.dataset == "Geo_Group":
    init_instruction = """Identify geometric shapes from their SVG paths."""
    output_format = "When you provide the final answer, please use the prefix \"The answer is:\" without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\""

### Start Training
for idx, train_sample in enumerate(train_set):
    question, gt = train_sample
    retrieve_prototype = query_topk_threshold(
        db, query=question, topk=1, threshold=0.5
    )
    if len(retrieve_prototype) == 0:
        #### reflexion
        is_success, trajectory, memory = ans_agent.reflexion_answer(
            instruction=init_instruction,
            output_format=output_format,
            question=question,
            gt=gt,
            eval_fn=eval_fn
        )
        if is_success:
            # TODO 进入总结prototype环节，目标是context、strategy


            prototype_id = str(uuid4())
            demo = Demonstration(
                question=question,
                trajectory=trajectory
            )
            demos = [demo]
        else:
            #### skip this case
            pass