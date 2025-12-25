from .mmlu import MMLU, MMLUInstanceDataset
from .base import Dataset, DataLoader
from .leetcode import LeetCodeHardEval

from typing import Tuple, Callable
from textgrad import Variable
from textgrad.engine import EngineLM

AVAILABLE_DATASETS = [
    "BBH_object_counting",
    "BBH_word_sorting",
    "GSM8K_DSPy",
]

AVAILABLE_INSTANCE_DATASETS = [
    "MMLU_machine_learning",
    "MMLU_college_physics",
    "GPQA_diamond"
    "LeetCodeHardEval"
]

# copy from
# https://github.com/open-compass/opencompass/blob/b54e28c1db039e962987c31116e6c6d0c3906a14/opencompass/datasets/bbh.py#L32C1-L44C15
import re
def bbh_mcq_postprocess(text: str) -> str:
    ans = text
    ans_line = ans.split('answer is ')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    match = re.search(r'\(([A-Z])\)*', ans)
    if match:
        return match.group(1)
    match = re.search(r'([A-Z])', ans)
    if match:
        return match.group(1)
    return ans

# copy from
# https://github.com/open-compass/opencompass/blob/b54e28c1db039e962987c31116e6c6d0c3906a14/opencompass/datasets/bbh.py#L48C1-L62C15
def bbh_freeform_postprocess(text: str) -> str:
    ans = text
    ans_line = ans.split('answer is ')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    ans = ans.split('\n')[0].strip()

    if ans.endswith('.'):
        ans = ans[:-1].strip()

    match = re.search(r'\*\*(.*?)\*\*', ans)
    if match:
        return match.group(1)

    return ans

import textgrad as tg
def bbh_mcq_eval_fn(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    pred = bbh_mcq_postprocess(str(prediction.value))
    ref = bbh_mcq_postprocess(str(ground_truth_answer.value))
    return int(pred == ref)

wsc_mcq_eval_fn = bbh_mcq_eval_fn

def bbh_freeform_eval_fn(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    pred = bbh_freeform_postprocess(str(prediction.value))
    ref = str(ground_truth_answer.value)
    return int(pred == ref)

# copy from
# https://github.com/open-compass/opencompass/blob/d836b49fee431cd8109b0b687133ce08a8e286a5/opencompass/datasets/bbeh.py#L32C1-L57C34
def bbeh_freeform_postprocess(text: str) -> str:
    # Extract answer using specified prefixes
    prefixes = [
        'The answer is: ', 'The answer is ', 'The final answer is: ',
        'The final answer is '
    ]
    answer = text
    for prefix in prefixes:
        if prefix in text:
            answer = text.split(prefix)[-1]
            break

    # Remove formatting markup
    if '\\boxed' in answer:
        answer = re.sub(r'\\boxed{(.*?)}', r'\1', answer)  # latex box
    if '\\text' in answer:
        answer = re.sub(r'\\text(?:tt)?{(.*?)}', r'\1', answer)  # text/texttt
    if '**' in answer:
        answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)  # bold

    # Take first line and clean
    if '\n' in answer:
        answer = answer.split('\n')[0].strip()

    return answer.strip().lower()

def bbeh_freeform_eval_fn(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    pred = bbeh_freeform_postprocess(str(prediction.value))
    ref = str(ground_truth_answer.value).lower()
    correct = False
    if pred == ref:
        correct = True
    elif pred == ref.strip("'\"()[]"):
        correct = True
    elif ',' in ref:
        norm_pred = re.sub(r'\s*,\s*', ',', pred)
        norm_ref = re.sub(r'\s*,\s*', ',', ref)
        if norm_pred == norm_ref:
            correct = True
    return int(correct)

# copy from
# https://github.com/open-compass/opencompass/blob/b54e28c1db039e962987c31116e6c6d0c3906a14/opencompass/datasets/gsm8k.py#L38C1-L49C23
def gsm8k_dataset_postprocess(text: str) -> str:
    return text.split('#### ')[1].replace(',', '')

def gsm8k_postprocess(text: str) -> str:
    text = text.split('Question:')[0]
    numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
    if not numbers:
        return 'NULL'
    return numbers[-1]

def gsm8k_eval_fn(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    pred = gsm8k_postprocess(str(prediction.value))
    ref = gsm8k_dataset_postprocess(str(ground_truth_answer.value))
    return int(pred == ref)

def load_task(task_name: str, evaluation_api: EngineLM, *args, **kwargs) -> Tuple[Dataset, Dataset, Callable]:
    """
    Args:
        task_name: the name of the task to evaluate
        evaluation_api: the engine to use for evaluation, if needed
    """
    if "object_counting" in task_name:
        from textgrad.loss import MultiFieldTokenParsedEvaluation
        from .big_bench_hard import BigBenchHard, string_based_equality_fn
        from textgrad.autograd.string_based_ops import StringBasedFunction
        task_name = task_name[4:]
        train_set = BigBenchHard(task_name, split="train", *args, **kwargs)
        val_set = BigBenchHard(task_name, split="val", *args, **kwargs)
        test_set = BigBenchHard(task_name, split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Reasoning and prediction from the language model"
        ]
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(string_based_equality_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn

    elif "tracking_shuffled_objects_five_objects" in task_name:
        from .big_bench_hard import BigBenchHard
        from textgrad.autograd.string_based_ops import StringBasedFunction
        task_name = task_name[4:]
        train_set = BigBenchHard(task_name, split="train", *args, **kwargs)
        val_set = BigBenchHard(task_name, split="val", *args, **kwargs)
        test_set = BigBenchHard(task_name, split="test", *args, **kwargs)
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(bbh_mcq_eval_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    
    elif "tracking_shuffled_objects_seven_objects" in task_name:
        from .big_bench_hard import BigBenchHard
        from textgrad.autograd.string_based_ops import StringBasedFunction
        task_name = task_name[4:]
        train_set = BigBenchHard(task_name, split="train", *args, **kwargs)
        val_set = BigBenchHard(task_name, split="val", *args, **kwargs)
        test_set = BigBenchHard(task_name, split="test", *args, **kwargs)
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(bbh_mcq_eval_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    
    elif "logical_deduction_seven_objects" in task_name:
        from .big_bench_hard import BigBenchHard
        from textgrad.autograd.string_based_ops import StringBasedFunction
        task_name = task_name[4:]
        train_set = BigBenchHard(task_name, split="train", *args, **kwargs)
        val_set = BigBenchHard(task_name, split="val", *args, **kwargs)
        test_set = BigBenchHard(task_name, split="test", *args, **kwargs)
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(bbh_mcq_eval_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    
    elif "boolean_expressions" in task_name:
        from .big_bench_hard import BigBenchHard
        from textgrad.autograd.string_based_ops import StringBasedFunction
        task_name = task_name[4:]
        train_set = BigBenchHard(task_name, split="train", *args, **kwargs)
        val_set = BigBenchHard(task_name, split="val", *args, **kwargs)
        test_set = BigBenchHard(task_name, split="test", *args, **kwargs)
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(bbh_freeform_eval_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    
    elif "causal_judgement" in task_name:
        from .big_bench_hard import BigBenchHard
        from textgrad.autograd.string_based_ops import StringBasedFunction
        task_name = task_name[4:]
        train_set = BigBenchHard(task_name, split="train", *args, **kwargs)
        val_set = BigBenchHard(task_name, split="val", *args, **kwargs)
        test_set = BigBenchHard(task_name, split="test", *args, **kwargs)
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(bbh_freeform_eval_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    
    elif "geometric_shapes" in task_name:
        from .big_bench_hard import BigBenchHard
        from textgrad.autograd.string_based_ops import StringBasedFunction
        task_name = task_name[4:]
        train_set = BigBenchHard(task_name, split="train", *args, **kwargs)
        val_set = BigBenchHard(task_name, split="val", *args, **kwargs)
        test_set = BigBenchHard(task_name, split="test", *args, **kwargs)
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(bbh_mcq_eval_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    
    elif task_name == "GSM8K_GPO":
        from textgrad.tasks.gsm8k import GSM8K_GPO
        from textgrad.autograd.string_based_ops import StringBasedFunction
        train_set = GSM8K_GPO(root=kwargs.get("data_dir"), split="train")
        val_set = GSM8K_GPO(root=kwargs.get("data_dir"), split="val")
        test_set = GSM8K_GPO(root=kwargs.get("data_dir"), split="test")
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(gsm8k_eval_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    
    elif task_name == "WSC":
        from textgrad.tasks.wsc import WSC
        from textgrad.autograd.string_based_ops import StringBasedFunction
        train_set = WSC(root=kwargs.get("data_dir"), split="train")
        val_set = WSC(root=kwargs.get("data_dir"), split="val")
        test_set = WSC(root=kwargs.get("data_dir"), split="test")
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(wsc_mcq_eval_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    
    elif task_name == "AGIEvalMath":
        from textgrad.tasks.agieval_math import AGIEvalMath, agieval_math_eval_fn
        from textgrad.autograd.string_based_ops import StringBasedFunction
        train_set = AGIEvalMath(root=kwargs.get("data_dir"), split="train")
        val_set = AGIEvalMath(root=kwargs.get("data_dir"), split="val")
        test_set = AGIEvalMath(root=kwargs.get("data_dir"), split="test")
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(agieval_math_eval_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    
    elif task_name == "BBEH_causal_understanding":
        from .bbeh import BigBenchExtraHard
        from textgrad.autograd.string_based_ops import StringBasedFunction
        task_name = task_name[5:]
        train_set = BigBenchExtraHard(task_name, split="train", *args, **kwargs)
        val_set = BigBenchExtraHard(task_name, split="val", *args, **kwargs)
        test_set = BigBenchExtraHard(task_name, split="test", *args, **kwargs)
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(bbeh_freeform_eval_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn

    elif "BBH" in task_name:
        from textgrad.loss import MultiFieldTokenParsedEvaluation
        from .big_bench_hard import BigBenchHard
        task_name = task_name[4:]
        train_set = BigBenchHard(task_name, split="train", *args, **kwargs)
        val_set = BigBenchHard(task_name, split="val", *args, **kwargs)
        test_set = BigBenchHard(task_name, split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Reasoning and prediction from the language model"
        ]
        
        evaluation_instruction = "Below is a question from a question-answering task, the ground truth answer, and reasoning with the final prediction. Is the final prediction correct, i.e. the same as the ground truth answer? Say only 1 (yes) or 0 (no). Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        eval_instruction = Variable(evaluation_instruction, requires_grad=False, role_description="evaluation instruction for the task")
        eval_fn = MultiFieldTokenParsedEvaluation(
            eval_instruction,
            engine=evaluation_api,
            role_descriptions=role_descriptions,
            parse_tags=["<ACCURACY>", "</ACCURACY>"]
        )
        
        return train_set, val_set, test_set, eval_fn
    
    elif task_name == "GSM8K_DSPy":
        from textgrad.tasks.gsm8k import GSM8K_DSPy
        from .big_bench_hard import string_based_equality_fn
        from textgrad.autograd.string_based_ops import StringBasedFunction
        evaluation_instruction = "Below is a prediction we got for a question answering task, and the correct final answer. Is the final answer correct? Say only 1 (yes) or 0 (no). Return 1 if and only if the final answer is correct. Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        system_prompt = Variable("You are a language model that evaluates the accuracy of a prediction for a mathematical question answering task. Only call a prediction accurate if it is the same as the ground truth answer.", requires_grad=False, role_description="system prompt for the evaluation")
        # Should we do train/test like this?
        train_set = GSM8K_DSPy(split="train", *args, **kwargs)
        val_set = GSM8K_DSPy(split="val", *args, **kwargs)
        test_set = GSM8K_DSPy(split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Prediction from the language model"
        ]
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(string_based_equality_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn
    
    else:
        raise ValueError(f"Task {task_name} not found.")


def load_instance_task(task_name: str, evaluation_api: EngineLM, *args, **kwargs):
    if "MMLU_" in task_name:
        subset = task_name[5:]
        test_set = MMLUInstanceDataset(evaluation_api=evaluation_api, subset=subset, split="test", *args, **kwargs)
        return test_set
    elif "GPQA" in task_name:
        from .gpqa import GPQAInstanceDataset
        test_set = GPQAInstanceDataset(evaluation_api=evaluation_api, subset=task_name.lower(), *args, **kwargs)
        return test_set
    elif task_name in ["LeetCodeHardEval"]:
        dataset = LeetCodeHardEval()
        return dataset
    else:
        raise ValueError(f"Instance task {task_name} not found.")