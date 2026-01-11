from .base import Dataset
from .gaokao_math import GaoKaoMath
from .aqua import AQUA
import random

class MathGroup(Dataset):
    def __init__(self, task_name:str, root:str=None, split:str="train", *args, **kwargs):
        gaokao_data = GaoKaoMath(
            root=root,
            split=split,
        )
        aqua_data = AQUA(
            root=root,
            split=split,
        )
        self.data = []
        min_length = min(len(gaokao_data), len(aqua_data))
        gaokao_data = [sample for sample in gaokao_data][:min_length]
        aqua_data = [sample for sample in aqua_data][:min_length]
        for idx, sample in enumerate(gaokao_data):
            self.data.append({
                "question": sample[0],
                "answer": sample[1],
                "source": "gaokao_math"
            })
        
        for idx, sample in enumerate(aqua_data):
            self.data.append({
                "question": sample[0],
                "answer": sample[1],
                "source": "aqua_math"
            })
        random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return row["question"], row["answer"]
        
