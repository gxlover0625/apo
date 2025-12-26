from .base import Dataset
from .bbeh import BigBenchExtraHard
from .big_bench_hard import BigBenchHard
import random

class GeoGroup(Dataset):
    def __init__(self, task_name:str, root:str=None, split:str="train", *args, **kwargs):
        bbh_data = BigBenchHard(
            task_name="geometric_shapes",
            split=split,
            *args,
            **kwargs
        )
        bbeh_data = BigBenchExtraHard(
            task_name="bbeh_geometric_shapes",
            split=split,
            *args,
            **kwargs
        )
        self.data = []
        min_length = min(len(bbh_data), len(bbeh_data))
        bbh_data = [sample for sample in bbh_data][:min_length]
        bbeh_data = [sample for sample in bbeh_data][:min_length]
        for idx, sample in enumerate(bbh_data):
            self.data.append({
                "question": sample[0],
                "answer": sample[1],
                "source": "BBH_Geometric_Shapes"
            })
        
        for idx, sample in enumerate(bbeh_data):
            self.data.append({
                "question": sample[0],
                "answer": sample[1],
                "source": "BBEH_Geometric_Shapes"
            })
        random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        return row["question"], row["answer"]