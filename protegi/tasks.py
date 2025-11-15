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

    def evaluate(self, model, prompt, test_exs, n=None, *args, **kwargs):
        if n is None:
            n = len(test_exs)
        texts, preds, labels = [], [], []
        acc_cnt = 0
        pbar = tqdm(enumerate(test_exs[:n]), total=min(n, len(test_exs)), desc='Evaluating')
        for i, ex in pbar:
            user_message = f"{prompt}\n{ex['text']}"
            pred = utils.chatgpt(
                user_message,
                temperature=0.0,
                n=1,
                max_tokens=4096
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