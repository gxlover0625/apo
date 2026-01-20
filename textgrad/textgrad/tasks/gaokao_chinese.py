import json

from .base import Dataset
from pathlib import Path

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

class GaoKaoChinese(Dataset):
    def __init__(self, root:str=None, split:str="train"):
        assert split in ["train", "val", "test"]
        train_path = Path(root) / "AGIEval_Gaokao" / "gaokao_chinese_train.jsonl"
        val_path = Path(root) / "AGIEval_Gaokao" / "gaokao_chinese_validation.jsonl"
        test_path = Path(root) / "AGIEval_Gaokao" / "gaokao_chinese_test.jsonl"
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