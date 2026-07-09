import json
import re
from typing import List, Dict, Tuple, Union

TASK_INSTRUCTION = "Let's solve the problem. Format your response as follows: \"The correct answer is (insert answer here)\""

CHOICE_PATTERNS = [
    r'answer is \((.)\)',
    r'Answer: \((.)\)',
    r'answer: \((.)\)',
    r'answer \((.)\)',
    r'\((.)\)',
]


def load_data(data_path: Union[str, List[str]]) -> List[Dict]:
    if isinstance(data_path, str):
        data_path = [data_path]

    data = []
    for path in data_path:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def extract_choice(answer: str) -> str:
    for pattern in CHOICE_PATTERNS:
        match = re.search(pattern, answer)
        if match and match.group(1) in ['A', 'B', 'C', 'D', 'E']:
            return match.group(1)
    return None


class DataProcessor:
    def __init__(self, task_name: str):
        self.task_name = task_name

    def _prepare_input(self, item: Dict) -> Tuple[str, str, str]:
        context = ""
        question = f"{item['question']} {TASK_INSTRUCTION}"
        target = item["answer"]
        return context, question, target

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        processed_data = []
        for item in raw_data:
            context, question, target = self._prepare_input(item)
            processed_data.append({
                "context": context,
                "question": question,
                "target": target,
            })
        return processed_data

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        return extract_choice(predicted) == ground_truth

    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        correct_count = 0
        for predicted, ground_truth in zip(out, target):
            if self.answer_is_correct(predicted, ground_truth):
                correct_count += 1
        return correct_count / len(out) if out else 0.0
