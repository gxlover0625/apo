import json
import numpy as np
import random
import os
import threading

from dataclasses import dataclass, asdict
from transformers import HfArgumentParser
from pathlib import Path
from openai.resources.chat.completions import Completions
from collections import defaultdict

from utils import get_logger, get_timestamp
from memapo_v2 import MemAPO
from retriever import Retriever
from evaluator import Evaluator
from updater import Updater
from ctm import CorrectTemplateMemory
from epm import ErrorPatternMemory
from task import get_eval_fn

from textgrad.tasks import load_task

class TokenMeter:
    def __init__(self):
        self.lock = threading.Lock()
        self._stats = defaultdict(lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cnt": 0,
        })

    def update(self, model: str, usage=None):
        if usage is None:
            return
        with self.lock:
            s = self._stats[model]
            s["input_tokens"] += getattr(usage, "prompt_tokens", 0)
            s["output_tokens"] += getattr(usage, "completion_tokens", 0)
            s["total_tokens"] += getattr(usage, "total_tokens", 0)

    def inc_cnt(self, model: str):
        with self.lock:
            self._stats[model]["cnt"] += 1

    def report(self, verbose=True):
        with self.lock:
            if verbose:
                print("=" * 50)
                for model, s in sorted(self._stats.items()):
                    print(f"[TokenMeter] [{model}] calls={s['cnt']} | input={s['input_tokens']} | output={s['output_tokens']} | total={s['total_tokens']}")
                print("=" * 50)

token_meter = TokenMeter()
_original_create = Completions.create

def patched_create(self, *args, **kwargs):
    model = kwargs.get("model", "unknown")
    is_stream = kwargs.get("stream", False)
    if not is_stream:
        response = _original_create(self, *args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage:
            token_meter.update(model, usage)
        token_meter.inc_cnt(model)
        return response
    else:
        response_stream = _original_create(self, *args, **kwargs)
        def stream_wrapper():
            final_usage = None
            for chunk in response_stream:
                if chunk.usage:
                    final_usage = chunk.usage
                yield chunk
            token_meter.update(model, final_usage)
            token_meter.inc_cnt(model)
        return stream_wrapper()
    
Completions.create = patched_create

@dataclass
class MemAPOArgs:
    dataset:str = ""

    correct_threshold:float = 0.7
    correct_topk:int = 3
    max_templates:int = 30

    error_threshold:float = 0.7
    error_topk:int = 1

    max_retries:int = 3
    embed_model:str = ""
    llm_model:str = ""
    llm_temperature:float = 0.7
    
    disable_logging:bool = False
    log_dir: str = "./logs"
    db_dir: str = "./db"

    exp_name:str = ""
    seed:int = 42

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    parser = HfArgumentParser(MemAPOArgs)
    args = parser.parse_args_into_dataclasses()[0]

    # 设置随机数种子, 获取时间
    seed_everything(args.seed)
    timestamp = get_timestamp()

    # 配置logger
    os.environ["disable_logging"] = "1" if args.disable_logging else "0"

    # 路径全部转绝对路径
    log_dir = str(Path(args.log_dir).resolve()) + f"/{args.exp_name}_{timestamp}"
    db_dir = str(Path(args.db_dir).resolve())
    data_dir = Path(__file__).resolve().parent.parent / "data"

    if not args.disable_logging:
        log_file = f"{log_dir}/train.log"
        logger = get_logger(log_file)
        logger.info("Args:\n%s", json.dumps(asdict(args), indent=2, ensure_ascii=False, sort_keys=True))

    ctm_collection = f"{args.exp_name}_{timestamp}_correct_template_memory"
    epm_collection = f"{args.exp_name}_{timestamp}_error_pattern_memory"

    print(f"log_dir:          {log_dir}")
    print(f"db_dir:           {db_dir}")
    print(f"data_dir:         {data_dir}")
    print(f"ctm_collection:   {ctm_collection}")
    print(f"epm_collection:   {epm_collection}")

    correct_template_memory = CorrectTemplateMemory(
        db_dir, 
        ctm_collection,
        args.embed_model,
        args.correct_threshold,
        args.correct_topk,
        args.max_templates,
    )
    error_pattern_memory = ErrorPatternMemory(
        db_dir,
        epm_collection,
        args.embed_model,
        args.error_threshold,
        args.error_topk,
    )
    retriever = Retriever(
        correct_template_memory,
        error_pattern_memory,
    )
    evaluator = Evaluator()
    updater = Updater(
        correct_template_memory,
        error_pattern_memory,
    )

    # 准备数据
    train_set, val_set, test_set, _ = load_task(args.dataset, evaluation_api=None, data_dir=data_dir)
    eval_fn = get_eval_fn(args.dataset)
    print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
    if args.dataset in ["Geo_Group", "BBH_geometric_shapes", "bbeh_geometric_shapes"]:
        init_instruction = """Identify geometric shapes from their SVG paths."""
        output_format = "When you provide the final answer, please use the prefix \"The answer is:\" without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\""
    elif args.dataset in ["Logical_Group", "BBH_logical_deduction_seven_objects", "bbeh_boardgame_qa"]:
        init_instruction = """Let's solve the problem."""
        output_format = "When you provide the final answer, please use the prefix \"The answer is:\" without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\""
    elif args.dataset in ["gpqa"]:
        init_instruction = """Let's solve the problem."""
        output_format = f"Format your response as follows: \"The correct answer is (insert answer here)\""
    elif args.dataset in ["agieval_aqua", "agieval_gaokao_math", "agieval_sat", "math_group", "gaokao_group"]:
        init_instruction = """Let's solve the problem."""
        output_format = f"Format your response as follows: \"The correct answer is (insert answer here)\""
    elif args.dataset in ["agieval_gaokao_history", "agieval_gaokao_chinese", "agieval_gaokao_geography", "human_group"]:
        init_instruction = """Let's solve the problem."""
        output_format = f"Format your response as follows: \"The correct answer is (insert answer here)\""

    memapo = MemAPO(
        retriever,
        evaluator,
        updater,
        args.llm_model,
        args.llm_temperature,
        init_instruction,
        output_format,
        eval_fn,
        args.max_retries,
    )
    memapo.train(train_set)

    # 报告训练阶段 token 消耗
    token_meter.report()

    # 保存两个memory的dict
    ctm_save_path = f"{log_dir}/ctm.json"
    epm_save_path = f"{log_dir}/epm.json"
    correct_template_memory.save_to_json(ctm_save_path)
    error_pattern_memory.save_to_json(epm_save_path)
    print(f"CTM saved to:     {ctm_save_path}")
    print(f"EPM saved to:     {epm_save_path}")

    # 保存 checkpoint 配置，方便推理时 reuse
    checkpoint_config = {
        "ctm_json_path": ctm_save_path,
        "epm_json_path": epm_save_path,
        "db_dir": db_dir,
        "ctm_collection": ctm_collection,
        "epm_collection": epm_collection,
        "embed_model": args.embed_model,
        "correct_threshold": args.correct_threshold,
        "correct_topk": args.correct_topk,
        "max_templates": args.max_templates,
        "error_threshold": args.error_threshold,
        "error_topk": args.error_topk,
        "llm_model": args.llm_model,
        "llm_temperature": args.llm_temperature,
        "init_instruction": init_instruction,
        "output_format": output_format,
        "dataset": args.dataset,
        "max_retries": args.max_retries,
    }
    ckpt_config_path = f"{log_dir}/checkpoint.json"
    with open(ckpt_config_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_config, f, ensure_ascii=False, indent=2)
    print(f"Checkpoint config: {ckpt_config_path}")

    test_save_path = f"{log_dir}/test_results.json"
    memapo.test(test_set, test_save_path)
    print("Done!")