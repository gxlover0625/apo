import utils
from collections import defaultdict
import numpy as np
from liquid import Template
from tqdm import tqdm
import concurrent.futures
import utils
import os
from tasks import bbh_freeform_postprocess, bbh_mcq_postprocess, bbeh_mcq_eval_fn, preprocess_sample, gpqa_eval_fn, gpqa_process_pred

def predict_on_example(inputs):
    ex, predictor, prompt = inputs
    if os.environ['TASK'] in ["logical_deduction_seven_objects", "causal_judgement", "geometric_shapes", "WSC"]:
        output_format = """You must give your final answer by starting with 'So the answer is'"""
        user_message = f"{prompt}\n{ex['text']}\n{output_format}"
    elif os.environ['TASK'] in ["bbeh_geometric_shapes", "Geo_Group", "Logical_Group"]:
        output_format = """When you provide the final answer, please use the prefix \"The answer is:\" without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\""""
        user_message = f"{prompt}\n{ex['text']}\n{output_format}"
    elif os.environ['TASK'] in ["gpqa"]:
        output_format = f"Format your response as follows: \"The correct answer is (insert answer here)\""
        user_message = f"{prompt}\n{ex['text']}\n{output_format}"
    pred = utils.chatgpt(
        user_message,
        temperature=0.0,
        n=1,
        # max_tokens=4096,
        task=True
    )[0]
    if os.environ['TASK'] in ['causal_judgement']:
        pred = bbh_freeform_postprocess(pred)
    elif os.environ['TASK'] in ['geometric_shapes', 'logical_deduction_seven_objects', "WSC"]:
        pred = bbh_mcq_postprocess(pred)
        if len(pred) == 1 and pred.isupper():
            pred = f"({pred})"
    elif os.environ['TASK'] in ["bbeh_geometric_shapes"]:
        pred = preprocess_sample(pred)
        pred = pred.upper()
        if len(pred) == 1 and pred.isupper():
            pred = f"({pred})"
    elif os.environ['TASK'] in ["Geo_Group", "Logical_Group"]:
        pred = preprocess_sample(pred)
        if ex['source'] == "BBEH_BoardGame":
            pass
        else:
            pred = pred.upper()
            if len(pred) == 1 and pred.isupper():
                pred = f"({pred})"
    elif os.environ['TASK'] in ["gpqa"]:
        pred = gpqa_process_pred(pred)

    # pred = predictor.inference(ex, prompt)
    return prompt, ex, pred

class Cached01Scorer:

    def __init__(self):
        self.cache = {}

    def __call__(self, predictor, prompts, data, agg='mean', max_threads=1):
        def compute_scores(prompts_exs):
            out_scores = {}
            inputs = [(ex, predictor, prompt) for prompt, ex in prompts_exs]
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(predict_on_example, ex) for ex in inputs]
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='01 scorer'):
                    prompt, ex, pred = future.result()            
                    if pred == ex['label']:
                        out_scores[f'{ex}-{prompt}'] = 1
                    else:
                        out_scores[f'{ex}-{prompt}'] = 0
            return out_scores

        cached_scores = defaultdict(list)
        prompts_exs_to_compute = []
        for ex, prompt in [(ex, prompt) for ex in data for prompt in prompts]:
            if f'{ex}-{prompt}' in self.cache:
                cached_scores[prompt].append(self.cache[f'{ex}-{prompt}'])
            else:
                prompts_exs_to_compute.append((prompt, ex))
        computed_scores = compute_scores(prompts_exs_to_compute)
        for prompt, ex in prompts_exs_to_compute:
            self.cache[f'{ex}-{prompt}'] = computed_scores[f'{ex}-{prompt}']
            cached_scores[prompt].append(computed_scores[f'{ex}-{prompt}'])

        if agg == 'mean':
            return [np.mean(cached_scores[prompt]) for prompt in prompts]
        else:
            raise Exception('Unk agg: '+ agg)


def logprob_on_example(inputs):
    ex, predictor, base_prompt, prompt, temperature = inputs
    lps = utils.instructGPT_logprobs(prompt, temperature=temperature)
    # last log prob is the log prob of answer (assuming single token responses)
    return base_prompt, ex, lps[0]['logprobs']['token_logprobs'][-1]


class CachedLogLikelihoodScorer:

    def __init__(self):
        self.cache = {}

    def __call__(self, predictor, prompts, data, agg='mean', max_threads=1):
        def compute_scores(prompts_exs):
            out_scores = {}
            inputs = []
            for prompt, ex in prompts_exs:
                inputs.append((
                    ex,
                    predictor,
                    prompt,
                    Template(
                        prompt + ' ' + predictor.categories[ex['label']]
                        ).render(text=ex['text']),
                            predictor.opt['temperature']
                ))
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
                futures = [executor.submit(logprob_on_example, input) for input in inputs]
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)
                                                ), total=len(futures), desc='ll scorer'):
                    prompt, ex, pred = future.result()            
                    out_scores[f'{ex}-{prompt}'] = pred
            return out_scores


        cached_scores = defaultdict(list)
        prompts_exs_to_compute = []
        for ex, prompt in [(ex, prompt) for ex in data for prompt in prompts]:
            if f'{ex}-{prompt}' in self.cache:
                cached_scores[prompt].append(self.cache[f'{ex}-{prompt}'])
            else:
                prompts_exs_to_compute.append((prompt, ex))

        computed_scores = compute_scores(prompts_exs_to_compute)
        for prompt, ex in prompts_exs_to_compute:
            self.cache[f'{ex}-{prompt}'] = computed_scores[f'{ex}-{prompt}']
            cached_scores[prompt].append(computed_scores[f'{ex}-{prompt}'])

        if agg == 'mean':
            return [np.mean(cached_scores[prompt]) for prompt in prompts]
        else:
            raise Exception('Unk agg: '+ agg)
