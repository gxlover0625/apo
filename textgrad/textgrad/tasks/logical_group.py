from .base import Dataset
from .big_bench_hard import BigBenchHard
from .bbeh import BigBenchExtraHard
import random

class LogicalGroup(Dataset):
    def __init__(self, task_name:str, root:str=None, split:str="train", *args, **kwargs):
        bbh_data = BigBenchHard(
            task_name="logical_deduction_seven_objects",
            split=split,
            *args,
            **kwargs
        )
        bbeh_data = BigBenchExtraHard(
            task_name="bbeh_boardgame_qa",
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
                "source": "BBH_Logical_Deduction_Seven_Objects"
            })
        
        for idx, sample in enumerate(bbeh_data):
            self.data.append({
                "question": sample[0],
                "answer": sample[1],
                "source": "BBEH_BoardGame_QA"
            })
        random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        return row["question"], row["answer"]