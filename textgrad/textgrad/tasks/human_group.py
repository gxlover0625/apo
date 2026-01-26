from .base import Dataset
from .gaokao_history import GaoKaoHistory
from .gaokao_geography import GaoKaoGeography
import random

class HumanGroup(Dataset):
    def __init__(self, task_name:str, root:str=None, split:str="train", *args, **kwargs):
        history_data = GaoKaoHistory(
            root=root,
            split=split,
        )
        geography_data = GaoKaoGeography(
            root=root,
            split=split,
        )
        self.data = []
        min_length = min(len(history_data), len(geography_data))
        history_data = [sample for sample in history_data][:min_length]
        geography_data = [sample for sample in geography_data][:min_length]
        for idx, sample in enumerate(history_data):
            self.data.append({
                "question": sample[0],
                "answer": sample[1],
                "source": "gaokao_history"
            })
        
        for idx, sample in enumerate(geography_data):
            self.data.append({
                "question": sample[0],
                "answer": sample[1],
                "source": "gaokao_geography"
            })
        random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return row["question"], row["answer"]
