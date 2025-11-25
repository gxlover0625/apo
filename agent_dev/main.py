import os
from tasks import load_logical7
from openai import OpenAI
import re

def call_openai(prompt, model='Qwen3-Next-80B-A3B-Instruct'):
    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=os.environ['OPENAI_BASE_URL']
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

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

train_data, eval_data, test_data = load_logical7()
initial_prompt = """You must give your final answer by starting with 'So the answer is'"""

error_idx = []
for i, sample in enumerate(train_data):
    prompt = f"{initial_prompt}\n{sample['input']}"
    raw_response = call_openai(prompt)
    is_correct = bbh_mcq_eval_fn(raw_response, sample['target'])
    if not is_correct:
        error_idx.append(i)

error_samples = [train_data[i] for i in error_idx]