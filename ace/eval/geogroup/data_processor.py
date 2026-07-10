import json
import re
from typing import List, Dict, Tuple, Union

TASK_INSTRUCTION = (
    "Identify geometric shapes from their SVG paths. "
    "When you provide the final answer, please use the prefix \"The answer is:\" without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\""
)

# NOTE: the matching logic below is a byte-for-byte port of memapo/task.py
# (strip_latex / extract_answer / preprocess_sample / preprocess_reference /
# fuzzy_match -> bbeh_mcq_eval_fn, which get_eval_fn maps "Geo_Group" to) so that
# ACE numbers stay directly comparable with the memapo baseline.
# Do not "improve" it — the case-sensitive prefix behaviour is intentional.
ANSWER_PREFIXES = [
    "The answer is:",
    "The final answer is ",
    "The final answer is: ",
    "The answer is ",
]


def strip_latex(response: str) -> str:
    """Port of memapo.task.strip_latex."""
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response


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


def extract_answer(sample: str) -> str:
    """Port of memapo.task.extract_answer."""
    answer = sample
    for answer_prefix in ANSWER_PREFIXES:
        if answer_prefix in answer:
            answer = answer.split(answer_prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return strip_latex(answer)


def preprocess_sample(sample: str) -> str:
    """Port of memapo.task.preprocess_sample."""
    prediction = extract_answer(sample.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    return prediction


def preprocess_reference(reference: str) -> str:
    """Port of memapo.task.preprocess_reference."""
    reference = reference.strip().lower()
    reference = reference.replace(", ", ",")
    return reference


def fuzzy_match(prediction: str, reference: str) -> bool:
    """Port of memapo.task.fuzzy_match (BigBench Extra Hard matcher)."""
    if prediction == reference:
        return True

    # (a) vs a
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction

    # Numbers
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass

    # quote issues
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True

    # Bracket issues
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True

    # Question mark issues
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True

    return False


class DataProcessor:
    def __init__(self, task_name: str):
        self.task_name = task_name

    def _prepare_input(self, item: Dict) -> Tuple[str, str, str]:
        context = ""
        question = f"{item['question']}\n{TASK_INSTRUCTION}"
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
        # Identical to memapo's bbeh_mcq_eval_fn (get_eval_fn -> "Geo_Group").
        pred = preprocess_sample(predicted)
        ref = preprocess_reference(ground_truth)
        return bool(fuzzy_match(pred, ref))

    def evaluate_accuracy(self, out: List[str], target: List[str]) -> float:
        correct_count = 0
        for predicted, ground_truth in zip(out, target):
            if self.answer_is_correct(predicted, ground_truth):
                correct_count += 1
        return correct_count / len(out) if out else 0.0
