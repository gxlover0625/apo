import json
from .base import Dataset
from pathlib import Path

def load_jsonl(data_path:str):
    data = []
    with open(data_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

class WSC(Dataset):
    def __init__(self, root:str, split:str):
        train_path = Path(root) / "WSC" / "train.jsonl"
        dev_path = Path(root) / "WSC" / "eval.jsonl"
        test_path = Path(root) / "WSC" / "test.jsonl"

        if split == "train":
            self.data = load_jsonl(train_path)
        elif split == "val":
            self.data = load_jsonl(dev_path)
        elif split == "test":
            self.data = load_jsonl(test_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        question = row['input']
        answer = row['output']
        return question, answer
    
    def get_task_description(self):
        return """Let's solve the problem"""

