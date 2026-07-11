#!/usr/bin/env python3
import os
import json
import argparse
import threading
from .data_processor import DataProcessor, load_data

from ace import ACE
from utils import initialize_clients
from openai.resources.chat.completions import Completions


class TokenMeter:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.cnt = 0
        self.lock = threading.Lock()

    def update(self, usage=None):
        if usage is None:
            return
        with self.lock:
            self.input_tokens += getattr(usage, "prompt_tokens", 0)
            self.output_tokens += getattr(usage, "completion_tokens", 0)
            self.total_tokens += getattr(usage, "total_tokens", 0)

    def report(self, verbose=True):
        with self.lock:
            if verbose:
                print(f"Total API calls: {self.cnt}")
                print(f"Total tokens: {self.total_tokens}")
                print(f"Input tokens: {self.input_tokens}")
                print(f"Output tokens: {self.output_tokens}")


token_meter = TokenMeter()
_original_create = Completions.create


def patched_create(self, *args, **kwargs):
    is_stream = kwargs.get("stream", False)
    if not is_stream:
        response = _original_create(self, *args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage:
            token_meter.update(usage)
        with token_meter.lock:
            token_meter.cnt += 1
        return response
    else:
        response_stream = _original_create(self, *args, **kwargs)

        def stream_wrapper():
            final_usage = None
            for chunk in response_stream:
                if chunk.usage:
                    final_usage = chunk.usage
                yield chunk
            token_meter.update(final_usage)
            with token_meter.lock:
                token_meter.cnt += 1
        return stream_wrapper()


Completions.create = patched_create


def parse_args():
    parser = argparse.ArgumentParser(description='Run ACE on human_group')
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--mode", type=str, default="offline",
                        choices=['offline', 'online', 'eval_only'],
                        help="Run mode: 'offline' for offline training, "
                             "'online' for online training, 'eval_only' for evaluation only")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--initial_playbook_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default="./eval/humangroup/data/sample_config.json")
    parser.add_argument("--api_provider", type=str, default="openai",
                        choices=["sambanova", "together", "openai", "commonstack"],
                        help="API provider")
    parser.add_argument("--model", type=str, default="DeepSeek-V3.1",
                        help="Model used for generator, reflector, and curator")
    return parser.parse_args()


def preprocess_data(task_name, config, mode):
    processor = DataProcessor(task_name=task_name)

    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None

        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            raise ValueError(f"{mode} mode requires test data in config.")

        if mode == "online":
            print(f"Online mode: Training and testing on {len(test_samples)} examples")
        else:
            print(f"Eval only mode: Testing on {len(test_samples)} examples")

    else:
        train_samples = load_data(config["train_data"])
        val_samples = load_data(config["val_data"])
        train_samples = processor.process_task_data(train_samples)
        val_samples = processor.process_task_data(val_samples)

        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            test_samples = []

        print(f"Offline mode: Training on {len(train_samples)} examples, "
              f"validating on {len(val_samples)}, testing on {len(test_samples)}")

    return train_samples, val_samples, test_samples, processor


def load_initial_playbook(path):
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return None


def main():
    args = parse_args()

    with open(args.config_path, 'r') as f:
        task_config = json.load(f)

    train_samples, val_samples, test_samples, data_processor = \
        preprocess_data(args.task_name, task_config[args.task_name], args.mode)

    initial_playbook = load_initial_playbook(args.initial_playbook_path)
    if initial_playbook:
        print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
    else:
        print("Using empty playbook as initial playbook\n")

    api_provider = args.api_provider
    ace_system = ACE(
        api_provider=api_provider,
        generator_model=args.model,
        reflector_model=args.model,
        curator_model=args.model,
        max_tokens=4096,
        initial_playbook=initial_playbook
    )

    config = {
        'num_epochs': 1,
        'max_num_rounds': 3,
        'curator_frequency': 1,
        'eval_steps': 100,
        'online_eval_frequency': 15,
        'save_steps': 50,
        'playbook_token_budget': 80000,
        'task_name': args.task_name,
        'mode': args.mode,
        'json_mode': False,
        'no_ground_truth': False,
        'save_dir': args.save_path,
        'test_workers': 20,
        'initial_playbook_path': args.initial_playbook_path,
        'use_bulletpoint_analyzer': False,
        'api_provider': api_provider
    }

    results = ace_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=data_processor,
        config=config
    )

    token_meter.report()


if __name__ == "__main__":
    main()
