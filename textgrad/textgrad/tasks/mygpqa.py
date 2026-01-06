import json
import re
import textgrad as tg

from .base import Dataset
from pathlib import Path

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def gpqa_process_pred(answer):
    patterns = [r'answer is \((.)\)', r'Answer: \((.)\)', r'answer: \((.)\)', r'answer \((.)\)', r'\((.)\)']
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match and match.group(1) in ['A', 'B', 'C', 'D', 'E']:
            return match.group(1)
    return None

def gpqa_eval_fn(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    pred = gpqa_process_pred(str(prediction.value))
    ref = str(ground_truth_answer.value)
    return int(pred == ref)

class GPQA(Dataset):
    def __init__(self, root:str=None, split:str="train"):
        assert split in ["train", "val", "test"]
        train_path = Path(root) / "GPQA" / "gpqa_train.jsonl"
        val_path = Path(root) / "GPQA" / "gpqa_validation.jsonl"
        test_path = Path(root) / "GPQA" / "gpqa_test.jsonl"
        if split == "train":
            self.data = load_jsonl(train_path)
        elif split == "val":
            self.data = load_jsonl(val_path)
        elif split == "test":
            self.data = load_jsonl(test_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        question = row['question']
        answer = row['answer']
        return question, answer
    
    def get_task_description(self):
        return """Let's solve the problem."""