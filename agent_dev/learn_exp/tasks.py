import json
import random
import numpy as np
import re
# check
def load_logical7(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)['examples']
    random.seed(42)
    np.random.seed(42)
    random.shuffle(data)
    train_data = data[:50]
    eval_data = data[50:150]
    test_data = data[150:]
    return train_data, eval_data, test_data

# check
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

# check
def bbh_mcq_eval_fn(prediction: str, ground_truth_answer: str):
    pred = bbh_mcq_postprocess(prediction)
    ref = bbh_mcq_postprocess(ground_truth_answer)
    return int(pred == ref)

# train_data, eval_data, test_data = load_logical7()