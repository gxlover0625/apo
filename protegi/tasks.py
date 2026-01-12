import requests
import json
import concurrent.futures
from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import random
import numpy as np
import re
import utils
from tqdm import tqdm
from pathlib import Path

class DataProcessor(ABC):
    def __init__(self, data_dir, max_threads=1):
        self.data_dir = data_dir
        self.max_threads = max_threads

    @abstractmethod
    def get_train_examples(self):
        pass

    @abstractmethod
    def get_test_examples(self):
        pass

    @abstractmethod
    def evaluate(self, predictor, test_exs):
        pass

    @abstractmethod
    def stringify_prediction(self, pred):
        pass




def process_example(ex, predictor, prompt):
    pred = predictor.inference(ex, prompt)
    return ex, pred


class ClassificationTask(DataProcessor):

    def run_evaluate(self, predictor, prompt, test_exs, n=100):
        labels = []
        preds = []
        texts = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(process_example, ex, predictor, prompt) for ex in test_exs[:n]]
            for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='running evaluate'):
                ex, pred = future.result()
                texts.append(ex['text'])
                labels.append(ex['label'])
                preds.append(pred)

        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='micro')
        return f1, texts, labels, preds

    def evaluate(self, predictor, prompt, test_exs, n=100):
        while True:
            try:
                f1, texts, labels, preds = self.run_evaluate(predictor, prompt, test_exs, n=n)
                break
            except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError):
                pass
        return f1, texts, labels, preds


class BinaryClassificationTask(ClassificationTask):
    categories = ['No', 'Yes']

    def stringify_prediction(self, pred):
        return BinaryClassificationTask.categories[pred]


class EthosBinaryTask(BinaryClassificationTask):
    categories = ['No', 'Yes']

    def get_train_examples(self):
        df = pd.read_csv(self.data_dir + '/ethos_ishate_binary_shuf.csv', sep=';', header=None)
        df = df[(df[1] <= 0) | (df[1] >= 0.7)]
        exs = df.reset_index().to_dict('records')
        exs = [{'id': x['index'], 'text': x[0], 'label': 1 if x[1] > 0.4 else 0} for x in exs[200:]]
        return exs
    
    def get_test_examples(self):
        df = pd.read_csv(self.data_dir + '/ethos_ishate_binary_shuf.csv', sep=';', header=None)
        df = df[(df[1] <= 0) | (df[1] >= 0.7)]
        exs = df.reset_index().to_dict('records')
        exs = [{'id': x['index'], 'text': x[0], 'label': 1 if x[1] > 0.4 else 0} for x in exs[:200]]
        return exs


class JailbreakBinaryTask(BinaryClassificationTask):
    categories = ['No', 'Yes']

    def get_train_examples(self):
        exs = []
        for i, l in enumerate(open(self.data_dir + '/train.tsv')):
            convo, label = l.strip().split('\t')
            label = int(label)
            text = ' '.join([x['text'].strip() for x in json.loads(convo) if x['role'] == 'user'])
            exs.append({'id': i, 'text': text, 'label': label})
        return exs
    
    def get_test_examples(self):
        exs = []
        for i, l in enumerate(open(self.data_dir + '/test.tsv')):
            convo, label = l.strip().split('\t')
            label = int(label)
            text = ' '.join([x['text'].strip() for x in json.loads(convo) if x['role'] == 'user'])
            exs.append({'id': i, 'text': text, 'label': label})
        return exs


class DefaultHFBinaryTask(BinaryClassificationTask):
    categories = ['No', 'Yes']

    def get_train_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/train.jsonl')):
            row = json.loads(row.strip())
            exs.append({'id': f'train-{i}', 'label': row['label'], 'text': row['text']})
        return exs
    
    def get_test_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/test.jsonl')):
            row = json.loads(row.strip())
            exs.append({'id': f'test-{i}', 'label': row['label'], 'text': row['text']})
        return exs

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

def bbh_freeform_eval_fn(prediction: str, ground_truth_answer: str):
    pred = bbh_freeform_postprocess(prediction)
    ref = ground_truth_answer
    return int(pred == ref)

# copy from
# https://github.com/open-compass/opencompass/blob/b54e28c1db039e962987c31116e6c6d0c3906a14/opencompass/datasets/bbh.py#L32C1-L44C15
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

def bbh_mcq_eval_fn(prediction: str, ground_truth_answer: str):
    pred = bbh_mcq_postprocess(prediction)
    ref = bbh_mcq_postprocess(ground_truth_answer)
    return int(pred == ref)

class CausalJudgementTask(DataProcessor):
    def __init__(self, data_dir, max_threads, *args, **kwargs):
        super().__init__(data_dir, max_threads)
        with open(data_dir, "r") as f:
            all_data = json.load(f)['examples']

        random.seed(42)
        np.random.seed(42)
        random.shuffle(all_data)
        self.train_data = all_data[:37]
        self.eval_data = all_data[37: 37+74]
        self.test_data = all_data[37+74:]

    def get_train_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.train_data):
            exs.append({'id': f'train-{idx}', 'label': sample['target'], 'text': sample['input']})
        return exs

    def get_test_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.test_data):
            exs.append({'id': f'test-{idx}', 'label': sample['target'], 'text': sample['input']})
        return exs

    def get_eval_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.eval_data):
            exs.append({'id': f'eval-{idx}', 'label': sample['target'], 'text': sample['input']})
        return exs

    def evaluate(self, model, prompt, test_exs, n=None, *args, **kwargs):
        if n is None:
            n = len(test_exs)
        texts, preds, labels = [], [], []
        acc_cnt = 0
        pbar = tqdm(enumerate(test_exs[:n]), total=min(n, len(test_exs)), desc='Evaluating')
        for i, ex in pbar:
            output_format = """You must give your final answer by starting with 'So the answer is'"""
            user_message = f"{prompt}\n{ex['text']}\n{output_format}"
            pred = utils.chatgpt(
                user_message,
                temperature=0.0,
                n=1,
                max_tokens=4096,
                task=True
            )[0]
            preds.append(bbh_freeform_postprocess(pred))
            labels.append(ex['label'])
            texts.append(ex['text'])
            accuracy = bbh_freeform_eval_fn(pred, ex['label'])
            acc_cnt += accuracy
            pbar.set_description(f'acc_score: {acc_cnt / (i + 1)}')
        acc_score = acc_cnt / min(n, len(test_exs))
        return acc_score, texts, labels, preds

    def stringify_prediction(self, pred, *args, **kwargs):
        return pred

class GeometricShapesTask(DataProcessor):
    def __init__(self, data_dir, max_threads, *args, **kwargs):
        super().__init__(data_dir, max_threads)
        with open(data_dir, "r") as f:
            all_data = json.load(f)['examples']

        random.seed(42)
        np.random.seed(42)
        random.shuffle(all_data)
        self.train_data = all_data[:50]
        self.eval_data = all_data[50: 150]
        self.test_data = all_data[150:]
    
    def get_train_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.train_data):
            exs.append({'id': f'train-{idx}', 'label': sample['target'], 'text': sample['input']})
        return exs

    def get_test_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.test_data):
            exs.append({'id': f'test-{idx}', 'label': sample['target'], 'text': sample['input']})
        return exs

    def get_eval_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.eval_data):
            exs.append({'id': f'eval-{idx}', 'label': sample['target'], 'text': sample['input']})
        return exs

    def evaluate(self, model, prompt, test_exs, n=None, *args, **kwargs):
        if n is None:
            n = len(test_exs)
        texts, preds, labels = [], [], []
        acc_cnt = 0
        pbar = tqdm(enumerate(test_exs[:n]), total=min(n, len(test_exs)), desc='Evaluating')
        for i, ex in pbar:
            output_format = """You must give your final answer by starting with 'So the answer is'"""
            user_message = f"{prompt}\n{ex['text']}\n{output_format}"
            pred = utils.chatgpt(
                user_message,
                temperature=0.0,
                n=1,
                max_tokens=4096,
                task=True
            )[0]
            # preds.append(bbh_mcq_postprocess(pred))
            processed_pred = bbh_mcq_postprocess(pred)
            if len(processed_pred) == 1 and processed_pred.isupper():
                processed_pred = f"({processed_pred})"
            preds.append(processed_pred)    
            labels.append(ex['label'])
            texts.append(ex['text'])
            accuracy = bbh_mcq_eval_fn(pred, ex['label'])
            acc_cnt += accuracy
            pbar.set_description(f'acc_score: {acc_cnt / (i + 1)}')
        acc_score = acc_cnt / min(n, len(test_exs))
        return acc_score, texts, labels, preds
    
    def stringify_prediction(self, pred, *args, **kwargs):
        return pred

class LogicalDeductionSevenObjectsTask(GeometricShapesTask):
    pass

def load_jsonl(data_path:str):
    data = []
    with open(data_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

class WSCTask(DataProcessor):
    def __init__(self, data_dir, max_threads, *args, **kwargs):
        super().__init__(data_dir, max_threads)
        train_path = Path(data_dir) / "train.jsonl"
        eval_path = Path(data_dir) / "eval.jsonl"
        test_path = Path(data_dir) / "test.jsonl"

        self.train_data = load_jsonl(train_path)
        self.eval_data = load_jsonl(eval_path)
        self.test_data = load_jsonl(test_path)
    
    def get_train_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.train_data):
            exs.append({'id': f'train-{idx}', 'label': sample['output'], 'text': sample['input']})
        return exs
    
    def get_test_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.test_data):
            exs.append({'id': f'test-{idx}', 'label': sample['output'], 'text': sample['input']})
        return exs
    
    def get_eval_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.eval_data):
            exs.append({'id': f'eval-{idx}', 'label': sample['output'], 'text': sample['input']})
        return exs
    
    def evaluate(self, model, prompt, test_exs, n=None, *args, **kwargs):
        if n is None:
            n = len(test_exs)
        texts, preds, labels = [], [], []
        acc_cnt = 0
        pbar = tqdm(enumerate(test_exs[:n]), total=min(n, len(test_exs)), desc='Evaluating')
        for i, ex in pbar:
            output_format = """You must give your final answer by starting with 'So the answer is'"""
            user_message = f"{prompt}\n{ex['text']}\n{output_format}"
            pred = utils.chatgpt(
                user_message,
                temperature=0.0,
                n=1,
                max_tokens=4096,
                task=True
            )[0]
            # preds.append(bbh_mcq_postprocess(pred))
            processed_pred = bbh_mcq_postprocess(pred)
            if len(processed_pred) == 1 and processed_pred.isupper():
                processed_pred = f"({processed_pred})"
            preds.append(processed_pred)    
            labels.append(ex['label'])
            texts.append(ex['text'])
            accuracy = bbh_mcq_eval_fn(pred, ex['label'])
            acc_cnt += accuracy
            pbar.set_description(f'acc_score: {acc_cnt / (i + 1)}')
        acc_score = acc_cnt / min(n, len(test_exs))
        return acc_score, texts, labels, preds
    
    def stringify_prediction(self, pred, *args, **kwargs):
        return pred

def strip_latex(response: str) -> str:
  if response.startswith("$") and response.endswith("$"):
    response = response[1:-1]
  if "boxed{" in response and response.endswith("}"):
    response = response[0:-1].split("boxed{")[1]
  if "text{" in response and response.endswith("}"):
    response = response[0:-1].split("text{")[1]
  if "texttt{" in response and response.endswith("}"):
    response = response[0:-1].split("texttt{")[1]
  return response

def extract_answer(sample: str) -> str:
  """Extracts the final answer from the sample."""
  answer_prefixes = [
      "The answer is:",
      "The final answer is ",
      "The final answer is: ",
      "The answer is "
  ]
  answer = sample
  for answer_prefix in answer_prefixes:
    if answer_prefix in answer:
      answer = answer.split(answer_prefix)[-1].strip()
  if answer.endswith("."):
    answer = answer[:-1]
  return strip_latex(answer)


def fuzzy_match(prediction: str, reference: str) -> bool:
  """Fuzzy match function for BigBench Extra Hard."""
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

def preprocess_sample(sample: str) -> str:
  prediction = extract_answer(sample.strip()).lower()
  prediction = prediction.replace(", ", ",").replace("**", "")
  prediction = prediction.split("\n")[0]
  prediction = prediction[0:-1] if prediction.endswith(".") else prediction
  return prediction

def preprocess_reference(reference: str) -> str:
  reference = reference.strip().lower()
  reference = reference.replace(", ", ",")
  return reference

def bbeh_mcq_eval_fn(prediction: str, ground_truth_answer: str):
    pred = preprocess_sample(prediction)
    ref = preprocess_reference(ground_truth_answer)
    return fuzzy_match(pred, ref)

class BBEHGeo(DataProcessor):
    def __init__(self, data_dir, max_threads, *args, **kwargs):
        super().__init__(data_dir, max_threads)
        with open(data_dir, "r") as f:
            all_data = json.load(f)['examples']

        random.seed(42)
        np.random.seed(42)
        random.shuffle(all_data)
        self.train_data = all_data[:50]
        self.eval_data = all_data[50: 100]
        self.test_data = all_data[100:]
    
    def get_train_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.train_data):
            exs.append({'id': f'train-{idx}', 'label': sample['target'], 'text': sample['input'], 'source': "BBEH_Geometric_Shapes"})
        return exs
    
    def get_test_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.test_data):
            exs.append({'id': f'test-{idx}', 'label': sample['target'], 'text': sample['input'], 'source': "BBEH_Geometric_Shapes"})
        return exs
    
    def get_eval_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.eval_data):
            exs.append({'id': f'eval-{idx}', 'label': sample['target'], 'text': sample['input'], 'source': "BBEH_Geometric_Shapes"})
        return exs
    
    def evaluate(self, model, prompt, test_exs, n=None, *args, **kwargs):
        if n is None:
            n = len(test_exs)
        texts, preds, labels = [], [], []
        acc_cnt = 0
        pbar = tqdm(enumerate(test_exs[:n]), total=min(n, len(test_exs)), desc='Evaluating')
        for i, ex in pbar:
            output_format = """When you provide the final answer, please use the prefix \"The answer is:\" without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\""""
            user_message = f"{prompt}\n{ex['text']}\n{output_format}"
            pred = utils.chatgpt(
                user_message,
                temperature=0.0,
                n=1,
                # max_tokens=4096,
                task=True
            )[0]
            # preds.append(bbh_mcq_postprocess(pred))
            processed_pred = preprocess_sample(pred)
            processed_pred = processed_pred.upper()
            if len(processed_pred) == 1 and processed_pred.isupper():
                processed_pred = f"({processed_pred})"
            preds.append(processed_pred)    
            labels.append(ex['label'])
            texts.append(ex['text'])
            # accuracy = bbh_mcq_eval_fn(pred, ex['label'])
            accuracy = bbeh_mcq_eval_fn(pred, ex['label'])
            acc_cnt += accuracy
            pbar.set_description(f'acc_score: {acc_cnt / (i + 1)}')
        acc_score = acc_cnt / min(n, len(test_exs))
        return acc_score, texts, labels, preds
    
    def stringify_prediction(self, pred, *args, **kwargs):
        return pred

# TODO, 如果未来需要单独测BBEH_BoardGame任务，需要完整实现其逻辑，目前复用主要是看加载数据上的复用，评估方式会存在差异
class BBEHBoardGame(BBEHGeo):
    pass

class GeoGroup(DataProcessor):
    def __init__(self, data_dir, max_threads, *args, **kwargs):
        super().__init__(data_dir, max_threads)
        bbh_task_dir = kwargs['bbh_task_dir']
        bbeh_task_dir = kwargs['bbeh_task_dir']
        self.bbh_task = GeometricShapesTask(
            bbh_task_dir, max_threads
        )
        self.bbeh_task = BBEHGeo(
            bbeh_task_dir, max_threads
        )

        # 合并 train 数据
        self.train_data = []
        min_length = min(len(self.bbh_task.train_data), len(self.bbeh_task.train_data))
        for sample in self.bbh_task.train_data[:min_length]:
            self.train_data.append({**sample, 'source': 'BBH_Geometric_Shapes'})
        for sample in self.bbeh_task.train_data[:min_length]:
            self.train_data.append({**sample, 'source': 'BBEH_Geometric_Shapes'})
        random.shuffle(self.train_data)
        
        # 合并 eval 数据
        self.eval_data = []
        min_length = min(len(self.bbh_task.eval_data), len(self.bbeh_task.eval_data))
        for sample in self.bbh_task.eval_data[:min_length]:
            self.eval_data.append({**sample, 'source': 'BBH_Geometric_Shapes'})
        for sample in self.bbeh_task.eval_data[:min_length]:
            self.eval_data.append({**sample, 'source': 'BBEH_Geometric_Shapes'})
        random.shuffle(self.eval_data)
        
        # 合并 test 数据
        self.test_data = []
        min_length = min(len(self.bbh_task.test_data), len(self.bbeh_task.test_data))
        for sample in self.bbh_task.test_data[:min_length]:
            self.test_data.append({**sample, 'source': 'BBH_Geometric_Shapes'})
        for sample in self.bbeh_task.test_data[:min_length]:
            self.test_data.append({**sample, 'source': 'BBEH_Geometric_Shapes'})
        random.shuffle(self.test_data)
    
    def get_train_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.train_data):
            exs.append({
                'id': f'train-{idx}',
                'label': sample['target'],
                'text': sample['input'],
                'source': sample['source']
            })
        return exs
    
    def get_eval_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.eval_data):
            exs.append({
                'id': f'eval-{idx}',
                'label': sample['target'],
                'text': sample['input'],
                'source': sample['source']
            })
        return exs

    def get_test_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.test_data):
            exs.append({
                'id': f'test-{idx}',
                'label': sample['target'],
                'text': sample['input'],
                'source': sample['source']
            })
        return exs
    
    def evaluate(self, model, prompt, test_exs, n=None, *args, **kwargs):
        return self.bbeh_task.evaluate(model, prompt, test_exs, n, *args, **kwargs)

    def stringify_prediction(self, pred, *args, **kwargs):
        return pred

class LogicalGroup(GeoGroup):
    def __init__(self, data_dir, max_threads, *args, **kwargs):
        # super().__init__(data_dir, max_threads)
        bbh_task_dir = kwargs['bbh_task_dir']
        bbeh_task_dir = kwargs['bbeh_task_dir']
        self.bbh_task = LogicalDeductionSevenObjectsTask(
            bbh_task_dir, max_threads
        )
        self.bbeh_task = BBEHBoardGame(
            bbeh_task_dir, max_threads
        )

        # 合并 train 数据
        self.train_data = []
        min_length = min(len(self.bbh_task.train_data), len(self.bbeh_task.train_data))
        for sample in self.bbh_task.train_data[:min_length]:
            self.train_data.append({**sample, 'source': 'BBH_Logical7'})
        for sample in self.bbeh_task.train_data[:min_length]:
            self.train_data.append({**sample, 'source': 'BBEH_BoardGame'})
        random.shuffle(self.train_data)
        
        # 合并 eval 数据
        self.eval_data = []
        min_length = min(len(self.bbh_task.eval_data), len(self.bbeh_task.eval_data))
        for sample in self.bbh_task.eval_data[:min_length]:
            self.eval_data.append({**sample, 'source': 'BBH_Logical7'})
        for sample in self.bbeh_task.eval_data[:min_length]:
            self.eval_data.append({**sample, 'source': 'BBEH_BoardGame'})
        random.shuffle(self.eval_data)
        
        # 合并 test 数据
        self.test_data = []
        min_length = min(len(self.bbh_task.test_data), len(self.bbeh_task.test_data))
        for sample in self.bbh_task.test_data[:min_length]:
            self.test_data.append({**sample, 'source': 'BBH_Logical7'})
        for sample in self.bbeh_task.test_data[:min_length]:
            self.test_data.append({**sample, 'source': 'BBEH_BoardGame'})
        random.shuffle(self.test_data)
    
    def evaluate(self, model, prompt, test_exs, n=None, *args, **kwargs):
        if n is None:
            n = len(test_exs)
        texts, preds, labels = [], [], []
        acc_cnt = 0
        pbar = tqdm(enumerate(test_exs[:n]), total=min(n, len(test_exs)), desc='Evaluating')
        for i, ex in pbar:
            output_format = """When you provide the final answer, please use the prefix \"The answer is:\" without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\""""
            user_message = f"{prompt}\n{ex['text']}\n{output_format}"
            pred = utils.chatgpt(
                user_message,
                temperature=0.0,
                n=1,
                # max_tokens=4096,
                task=True
            )[0]
            # preds.append(bbh_mcq_postprocess(pred))
            processed_pred = preprocess_sample(pred)
            if ex['source'] == 'BBH_Logical7':
                processed_pred = processed_pred.upper()
                if len(processed_pred) == 1 and processed_pred.isupper():
                    processed_pred = f"({processed_pred})"
            preds.append(processed_pred)    
            labels.append(ex['label'])
            texts.append(ex['text'])
            # accuracy = bbh_mcq_eval_fn(pred, ex['label'])
            accuracy = bbeh_mcq_eval_fn(pred, ex['label'])
            acc_cnt += accuracy
            pbar.set_description(f'acc_score: {acc_cnt / (i + 1)}')
        acc_score = acc_cnt / min(n, len(test_exs))
        return acc_score, texts, labels, preds

def gpqa_process_pred(answer):
    patterns = [r'answer is \((.)\)', r'Answer: \((.)\)', r'answer: \((.)\)', r'answer \((.)\)', r'\((.)\)']
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match and match.group(1) in ['A', 'B', 'C', 'D', 'E']:
            return match.group(1)
    return None

def gpqa_eval_fn(prediction: str, ground_truth_answer: str):
    pred = gpqa_process_pred(prediction)
    ref = ground_truth_answer
    return pred == ref

class GPQA(DataProcessor):
    def __init__(self, data_dir, max_threads, *args, **kwargs):
        super().__init__(data_dir, max_threads)
        train_path = Path(data_dir) / "gpqa_train.jsonl"
        eval_path = Path(data_dir) / "gpqa_validation.jsonl"
        test_path = Path(data_dir) / "gpqa_test.jsonl"

        self.train_data = load_jsonl(train_path)
        self.eval_data = load_jsonl(eval_path)
        self.test_data = load_jsonl(test_path)
    
    def get_train_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.train_data):
            exs.append({'id': f'train-{idx}', 'label': sample['answer'], 'text': sample['question']})
        return exs
    
    def get_test_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.test_data):
            exs.append({'id': f'test-{idx}', 'label': sample['answer'], 'text': sample['question']})
        return exs
    
    def get_eval_examples(self, *args, **kwargs):
        exs = []
        for idx, sample in enumerate(self.eval_data):
            exs.append({'id': f'eval-{idx}', 'label': sample['answer'], 'text': sample['question']})
        return exs
    
    def evaluate(self, model, prompt, test_exs, n=None, *args, **kwargs):
        if n is None:
            n = len(test_exs)
        texts, preds, labels = [], [], []
        acc_cnt = 0
        pbar = tqdm(enumerate(test_exs[:n]), total=min(n, len(test_exs)), desc='Evaluating')
        for i, ex in pbar:
            output_format = f"Format your response as follows: \"The correct answer is (insert answer here)\""
            user_message = f"{prompt}\n{ex['text']}\n{output_format}"
            pred = utils.chatgpt(
                user_message,
                temperature=0.0,
                n=1,
                task=True
            )[0]
            # preds.append(bbh_mcq_postprocess(pred))
            processed_pred = gpqa_process_pred(pred)
            preds.append(processed_pred)    
            labels.append(ex['label'])
            texts.append(ex['text'])
            accuracy = gpqa_eval_fn(pred, ex['label'])
            acc_cnt += accuracy
            pbar.set_description(f'acc_score: {acc_cnt / (i + 1)}')
        acc_score = acc_cnt / min(n, len(test_exs))
        return acc_score, texts, labels, preds
    
    def stringify_prediction(self, pred, *args, **kwargs):
        return pred

class MathGroup(DataProcessor):
    def __init__(self, data_dir, max_threads, *args, **kwargs):
        super().__init__(data_dir, max_threads)
        self.gaokao_math_dir = Path(data_dir) / "AGIEval_Gaokao"
        self.aqua_dir = Path(data_dir) / "AGIEval_AQUA"

    def load_examples(self, gaokao_math_dir, aqua_dir, split="train"):
        gaokao_math_path = gaokao_math_dir / f"gaokao_math_{split}.jsonl"
        gaokao_math_data = load_jsonl(gaokao_math_path)
        aqua_path = aqua_dir / f"aqua_{split}.jsonl"
        aqua_data = load_jsonl(aqua_path)
        min_length = min(len(gaokao_math_data), len(aqua_data))
        exs = []
        for sample in gaokao_math_data[:min_length]:
            exs.append({
                "text": sample['question'], "label": sample['answer'], "source": "AGIEval_Gaokao_Math"
            })
        for sample in aqua_data[:min_length]:
            exs.append({
                "text": sample['question'], "label": sample['answer'], "source": "AGIEval_AQUA"
            })
        random.shuffle(exs)
        return exs
    
    def get_train_examples(self, *args, **kwargs):
        return self.load_examples(self.gaokao_math_dir, self.aqua_dir, split="train")
    
    def get_eval_examples(self, *args, **kwargs):
        return self.load_examples(self.gaokao_math_dir, self.aqua_dir, split="validation")
    
    def get_test_examples(self, *args, **kwargs):
        return self.load_examples(self.gaokao_math_dir, self.aqua_dir, split="test")

    def evaluate(self, model, prompt, test_exs, n=None, *args, **kwargs):
        if n is None:
            n = len(test_exs)
        texts, preds, labels = [], [], []
        acc_cnt = 0
        pbar = tqdm(enumerate(test_exs[:n]), total=min(n, len(test_exs)), desc='Evaluating')
        for i, ex in pbar:
            output_format = f"Format your response as follows: \"The correct answer is (insert answer here)\""
            user_message = f"{prompt}\n{ex['text']}\n{output_format}"
            pred = utils.chatgpt(
                user_message,
                temperature=0.0,
                n=1,
                # max_tokens=4096,
                task=True
            )[0]
            processed_pred = str(gpqa_process_pred(pred))
            preds.append(processed_pred)    
            labels.append(ex['label'])
            texts.append(ex['text'])
            accuracy = gpqa_eval_fn(pred, ex['label'])
            acc_cnt += accuracy
            pbar.set_description(f'acc_score: {acc_cnt / (i + 1)}')
        acc_score = acc_cnt / min(n, len(test_exs))
        return acc_score, texts, labels, preds

    def stringify_prediction(self, pred, *args, **kwargs):
        return pred