import os
import json
from .base import Dataset


class MTBench(Dataset):
    def __init__(self, category: str, root: str = None, split: str = "train", *args, **kwargs):
        if root is None:
            raise ValueError("root (data_dir) must be provided for MTBench")
        self.root = root
        self.split = split
        self.category = category
        data_path = os.path.join(self.root, "mt_bench", f"{category}_{split}.jsonl")
        assert os.path.exists(data_path), f"Data file not found: {data_path}"
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __getitem__(self, index):
        item = self.data[index]
        question = item["question"]
        return (question, None)

    def __len__(self):
        return len(self.data)
