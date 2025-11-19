import os
import json
from openai import OpenAI
from functools import partial
from tqdm import tqdm

from tasks import load_logical7, bbh_mcq_postprocess, bbh_mcq_eval_fn

api_key = os.environ['OPENAI_API_KEY']
base_url = os.environ['OPENAI_BASE_URL']

# check
def call_openai(prompt, model):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

call_task_model = partial(call_openai, model="Qwen3-30B-A3B-Instruct-2507")
call_opt_model = partial(call_openai, model='Qwen3-235B-A22B-Instruct-2507')

train_data, eval_data, test_data = load_logical7("/root/shared-nvme/apo/opro/data/BIG-Bench-Hard-data/logical_deduction_three_objects.json")
output_path = './strong_logical7.json'
strong_samples = []
for idx, sample in tqdm(enumerate(train_data)):
    initial_prompt = """You must give your final answer by starting with 'So the answer is'"""
    prompt = f"{initial_prompt}\n{sample['input']}"
    raw_pred = call_opt_model(prompt)
    processed_pred = bbh_mcq_postprocess(raw_pred)
    is_correct = bbh_mcq_eval_fn(raw_pred, sample['target'])
    if is_correct:
        strong_samples.append({
            **sample,
            'raw_pred': raw_pred,
            'processed_pred': processed_pred,
            'is_correct': is_correct,
            'idx': idx
        })

print(len(strong_samples)/len(train_data))
with open(output_path, 'w') as f:
    json.dump(strong_samples, f, indent=4, ensure_ascii=False)
