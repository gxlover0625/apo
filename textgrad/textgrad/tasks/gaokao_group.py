from .base import Dataset
from .gaokao_math import GaoKaoMath
from .gaokao_history import GaoKaoHistory
import random

class GaoKaoGroup(Dataset):
    def __init__(self, task_name:str, root:str=None, split:str="train", *args, **kwargs):
        gaokao_math_data = GaoKaoMath(
            root=root,
            split=split,
        )
        gaokao_history_data = GaoKaoHistory(
            root=root,
            split=split,
        )
        self.data = []
        min_length = min(len(gaokao_math_data), len(gaokao_history_data))
        gaokao_math_data = [sample for sample in gaokao_math_data][:min_length]
        gaokao_history_data = [sample for sample in gaokao_history_data][:min_length]
        for idx, sample in enumerate(gaokao_math_data):
            self.data.append({
                "question": sample[0],
                "answer": sample[1],
                "source": "gaokao_math"
            })
        
        for idx, sample in enumerate(gaokao_history_data):
            self.data.append({
                "question": sample[0],
                "answer": sample[1],
                "source": "gaokao_history"
            })
        random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return row["question"], row["answer"]
