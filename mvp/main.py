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
from datetime import datetime

from textgrad.tasks import load_task

from task import get_eval_fn
from embedding import get_db, add_single_doc, query_topk_threshold
from agent import AnswerAgent, SummaryAgent
from prototype import Prototype, Demonstration, Strategy

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=4))
def call_llm(user_prompt, sys_prompt="You are a helpful assistant.", model=None, temperature=0):
    client = OpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )
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
args = parser.parse_args()

# Data & Prompt setup
data_dir = Path(__file__).resolve().parent.parent / "data"
train_set, val_set, test_set, _ = load_task(args.dataset, evaluation_api=None, data_dir=data_dir)
eval_fn = get_eval_fn(args.dataset)
print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
if args.dataset in ["Geo_Group", "BBH_geometric_shapes", "bbeh_geometric_shapes"]:
    init_instruction = """Identify geometric shapes from their SVG paths."""
    output_format = "When you provide the final answer, please use the prefix \"The answer is:\" without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\""

# Training setup
prototype_dict = {}
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
db = get_db(collection_name=f"{args.exp_name}_{current_date}", emb_model=args.embed_model)
ans_agent = AnswerAgent(model=args.model, temperature=0., call_fn=call_llm)
sum_agent = SummaryAgent(model=args.model, temperature=0., call_fn=call_llm)

## Start Training
for idx, train_sample in tqdm(enumerate(train_set)):
    question, gt = train_sample
    retrieve_prototype = query_topk_threshold(
        db, query=question, topk=1, threshold=0.7
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
            prototype_id = str(uuid4())

            past_reflections = "\n".join([f"- {m}" for m in memory]) if memory else "None"
            context, solution_steps, pitfalls = sum_agent.summary(question, trajectory, past_reflections)
            strategy = Strategy(
                solution_steps=solution_steps,
                pitfalls=pitfalls
            )
            demo = Demonstration(
                question=question,
                trajectory=trajectory
            )
            demos = [demo]
            new_prototype = Prototype(
                prototype_id=prototype_id,
                context=context,
                demos=demos,
                strategy=strategy,
            )
            add_single_doc(
                db, context, doc_id=prototype_id, doc_metadata={"prototype_id": prototype_id}
            )
            prototype_dict[prototype_id] = new_prototype
            logging.info(f"[Sample {idx}] is creating prototype")
            print(f"[Sample {idx}] is creating prototype")
            with open(f"prototype_dict_{args.exp_name}_{current_date}.pkl", "wb") as f:
                pickle.dump(prototype_dict, f)
        else:
            #### skip this case
            logging.info(f"[Sample {idx}] not pass after reflexion")
            print(f"[Sample {idx}] not pass after reflexion")
    else:
        chosen_id = retrieve_prototype[0]["metadata"]["prototype_id"]
        chosen_prototype = prototype_dict[chosen_id]
        is_success, final_response = ans_agent.answer_with_prototype(
            instruction=init_instruction,
            output_format=output_format,
            question=question,
            gt=gt,
            eval_fn=eval_fn,
            prototype=chosen_prototype
        )
        if is_success:
            new_demo = Demonstration(question=question, trajectory=final_response)
            chosen_prototype.update_demo(new_demo)
            logging.info(f"[Sample {idx}] pass with chosen prototype")
            print(f"[Sample {idx}] pass with chosen prototype")
        else:
            logging.info(f"[Sample {idx}] not pass with chosen prototype")
            print(f"[Sample {idx}] not pass with chosen prototype")