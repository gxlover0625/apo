import os
import argparse
import concurrent
from dotenv import load_dotenv
from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
import numpy as np
import random

from openai import OpenAI
from textgrad.engine.local_model_openai_api import ChatExternalClient
load_dotenv(override=True)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def eval_sample(item, eval_fn, model):
    """
    This function allows us to evaluate if an answer to a question in the prompt is a good answer.

    """
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        return int(eval_output_variable.value)
    except:
        eval_output_variable = eval_fn([x, y, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)
    
def eval_dataset(test_set, eval_fn, model, max_samples: int=None):
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for _, sample in enumerate(test_set):
            
            future = executor.submit(eval_sample, sample, eval_fn, model)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list 

def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set):
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model))
    previous_performance = np.mean(results["validation_acc"][-1])
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]
    
    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)

set_seed(42)
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, default="BBH_tracking_shuffled_objects_seven_objects")
parser.add_argument("--task_model", type=str, required=True, default="Qwen3-4B-Instruct-2507")
parser.add_argument("--opt_model", type=str, required=True, default="Qwen3-4B-Instruct-2507")
args = parser.parse_args()
# llm_api_eval = tg.get_engine(engine_name="experimental:Qwen3-4B-Instruct-2507")
# llm_api_test = tg.get_engine(engine_name="experimental:Qwen3-4B-Instruct-2507")
base_url = os.environ['OPENAI_BASE_URL']
api_key = os.environ['OPENAI_API_KEY']
eval_client = OpenAI(base_url=base_url, api_key=api_key)
test_client = OpenAI(base_url=base_url, api_key=api_key)
llm_api_eval = ChatExternalClient(
    client=eval_client,
    model_string=args.opt_model,
    temperature=0.7
)
llm_api_test = ChatExternalClient(
    client=test_client,
    model_string=args.task_model,
    temperature=0.
)
tg.set_backward_engine(llm_api_eval, override=True)

# Load the data and the evaluation function
# train_set, val_set, test_set, eval_fn = load_task("BBH_object_counting", evaluation_api=llm_api_eval)
train_set, val_set, test_set, eval_fn = load_task(args.dataset, evaluation_api=llm_api_eval)
print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
# STARTING_SYSTEM_PROMPT = train_set.get_task_description()
STARTING_SYSTEM_PROMPT = """You must give your final answer by starting with 'So the answer is'"""

print(STARTING_SYSTEM_PROMPT)

train_loader = tg.tasks.DataLoader(train_set, batch_size=3, shuffle=True)


# Testing the 0-shot performance of the evaluation engine
system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                            requires_grad=True, 
                            role_description="system prompt to the language model")
model_evaluation = tg.BlackboxLLM(llm_api_eval, system_prompt)

system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                            requires_grad=True,
                            role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")
model = tg.BlackboxLLM(llm_api_test, system_prompt)

optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])

results = {"test_acc": [], "prompt": [], "validation_acc": []}
results["test_acc"].append(eval_dataset(test_set, eval_fn, model))
results["validation_acc"].append(eval_dataset(val_set, eval_fn, model))
results["prompt"].append(system_prompt.get_value())

for epoch in range(3):
    for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
        pbar.set_description(f"Training step {steps}. Epoch {epoch}")
        optimizer.zero_grad()
        losses = []
        for (x, y) in zip(batch_x, batch_y):
            x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
            y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
            response = model(x)
            try:
                eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
            except:
                eval_output_variable = eval_fn([x, y, response])
            losses.append(eval_output_variable)
        total_loss = tg.sum(losses)
        total_loss.backward()
        optimizer.step()
        
        run_validation_revert(system_prompt, results, model, eval_fn, val_set)
        
        print("sys prompt: ", system_prompt)
        test_acc = eval_dataset(test_set, eval_fn, model)
        results["test_acc"].append(test_acc)
        results["prompt"].append(system_prompt.get_value())
        if steps == 3:
            break

# 根据验证集的结果，选择最佳prompt
best_idx = np.argmax([
    np.mean(acc) for acc in results["validation_acc"]
])
best_prompt = results["prompt"][best_idx]
best_test_acc = np.mean(results["test_acc"][best_idx])
print("Best Prompt")
print(best_prompt)
print("Best Test Accuracy")
print(best_test_acc)