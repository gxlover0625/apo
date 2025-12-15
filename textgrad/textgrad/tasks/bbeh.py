import textgrad as tg
import os
import platformdirs
import json
import random
import pandas as pd
from .base import Dataset

class BigBenchExtraHard(Dataset):
    def __init__(self, task_name: str, root: str=None, split: str="train", *args, **kwargs):
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        self.root = root
        self.split = split
        self.task_name = task_name
        self._check_or_download_dataset()
        assert split in ["train", "val", "test"]
        data_path = os.path.join(self.root, self.task_name, f"{split}.csv")
        self.data = pd.read_csv(data_path, index_col=0)
        self._task_description = "Think step by step, and when you provide the final answer, please use the prefix \"The answer is:\" without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\""
    
    def get_task_description(self):
        return self._task_description

    def _check_or_download_dataset(self):
        data_path = os.path.join(self.root, self.task_name, f"{self.split}.csv")
        if os.path.exists(data_path):
            return
        
        os.makedirs(os.path.join(self.root, self.task_name), exist_ok=True)
        data = json.load(open(os.path.join(self.root, f"{self.task_name}.json")))
        examples = data["examples"]
        random.shuffle(examples)
        train_examples = [{"x": ex["input"], "y": ex["target"]} for ex in examples[:50]]
        val_examples = [{"x": ex["input"], "y": ex["target"]} for ex in examples[50:100]]
        test_examples = [{"x": ex["input"], "y": ex["target"]} for ex in examples[100:]]
        train_path = os.path.join(self.root, self.task_name, "train.csv")
        with open(train_path, "w") as f:
            pd.DataFrame(train_examples).to_csv(f)
        val_path = os.path.join(self.root, self.task_name, "val.csv")
        with open(val_path, "w") as f:
            pd.DataFrame(val_examples).to_csv(f)
        test_path = os.path.join(self.root, self.task_name, "test.csv")
        with open(test_path, "w") as f:
            pd.DataFrame(test_examples).to_csv(f)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        import numpy as np
        if isinstance(row["y"], np.bool):
            row["y"] = str(row["y"])
        to_remove_inst = [
            """  Reply Yes or No based on the answer the majority of people would give. If you think people would be split roughly 50-50 between Yes and No then reply Ambiguous.""",
            """ Reply based on the answer a logician would give."""
        ]
        for inst_remove in to_remove_inst:
            row["x"] = row["x"].replace(inst_remove, "")
        return row["x"], row["y"]
    
    def __len__(self):
        return len(self.data)