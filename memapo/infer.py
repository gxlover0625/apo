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

    def report(self, verbose=True, logger=None):
        with self.lock:
            if verbose:
                lines = ["=" * 50]
                for model, s in sorted(self._stats.items()):
                    lines.append(f"[TokenMeter] [{model}] calls={s['cnt']} | input={s['input_tokens']} | output={s['output_tokens']} | total={s['total_tokens']}")
                lines.append("=" * 50)
                for line in lines:
                    print(line, flush=True)
                if logger:
                    logger.info("\n".join(lines))

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
class InferArgs:
    dataset:str = ""
    category:str = ""

    correct_threshold:float = 0.1 # 第一次尝试的时候效果好，不一定是最好的
    correct_topk:int = 3
    max_templates:int = 30

    error_threshold:float = 0.7
    error_topk:int = 1

    max_retries:int = 3
    embed_model:str = ""
    llm_model:str = ""
    llm_temperature:float = 0.7
    num_workers:int = 1  # 推理并发线程数，>1 时启用多线程

    judge_model:str = ""
    judge_threshold:float = 7.0

    disable_logging:bool = False
    log_dir: str = "./logs"
    db_dir: str = "./db"

    exp_name:str = ""
    seed:int = 42

    # ---- reuse 相关 ----
    checkpoint:str = ""            # checkpoint.json 路径，传了就走 reuse
    ctm_json_path:str = ""         # 也可以手动指定各个路径
    epm_json_path:str = ""
    ctm_collection:str = ""
    epm_collection:str = ""

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def get_instruction_and_format(dataset:str):
    if dataset in ["Geo_Group", "BBH_geometric_shapes", "bbeh_geometric_shapes"]:
        init_instruction = """Identify geometric shapes from their SVG paths."""
        output_format = "When you provide the final answer, please use the prefix \"The answer is:\" without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\""
    elif dataset in ["Logical_Group", "BBH_logical_deduction_seven_objects", "bbeh_boardgame_qa"]:
        init_instruction = """Let's solve the problem."""
        output_format = "When you provide the final answer, please use the prefix \"The answer is:\" without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\""
    elif dataset in ["gpqa"]:
        init_instruction = """Let's solve the problem."""
        output_format = f"Format your response as follows: \"The correct answer is (insert answer here)\""
    elif dataset in ["agieval_aqua", "agieval_gaokao_math", "agieval_sat", "math_group", "gaokao_group"]:
        init_instruction = """Let's solve the problem."""
        output_format = f"Format your response as follows: \"The correct answer is (insert answer here)\""
    elif dataset in ["agieval_gaokao_history", "agieval_gaokao_chinese", "agieval_gaokao_geography", "human_group"]:
        init_instruction = """Let's solve the problem."""
        output_format = f"Format your response as follows: \"The correct answer is (insert answer here)\""
    elif dataset in ["mt_bench"]:
        init_instruction = ""
        output_format = ""
    else:
        init_instruction = """Let's solve the problem."""
        output_format = f"Format your response as follows: \"The correct answer is (insert answer here)\""
    return init_instruction, output_format

if __name__ == "__main__":
    parser = HfArgumentParser(InferArgs)
    args = parser.parse_args_into_dataclasses()[0]

    seed_everything(args.seed)
    timestamp = get_timestamp()

    os.environ["disable_logging"] = "1" if args.disable_logging else "0"
    log_dir = str(Path(args.log_dir).resolve()) + f"/{args.exp_name}_{timestamp}"
    db_dir = str(Path(args.db_dir).resolve())
    data_dir = Path(__file__).resolve().parent.parent / "data"

    if not args.disable_logging:
        log_file = f"{log_dir}/infer.log"
        logger = get_logger(log_file)
        logger.info("Args:\n%s", json.dumps(asdict(args), indent=2, ensure_ascii=False, sort_keys=True))

    # 如果传了 checkpoint，从中加载所有 reuse 配置
    if args.checkpoint:
        with open(args.checkpoint, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        ctm_json_path = args.ctm_json_path or ckpt["ctm_json_path"]
        epm_json_path = args.epm_json_path or ckpt["epm_json_path"]
        ctm_collection = args.ctm_collection or ckpt["ctm_collection"]
        epm_collection = args.epm_collection or ckpt["epm_collection"]
        db_dir = db_dir if args.db_dir != "./db" else ckpt.get("db_dir", db_dir)
        embed_model = args.embed_model or ckpt.get("embed_model", "")
        dataset = args.dataset or ckpt["dataset"]
        llm_model = args.llm_model or ckpt.get("llm_model", "")
        llm_temperature = args.llm_temperature if args.llm_temperature != 0.7 else ckpt.get("llm_temperature", 0.7)
        correct_threshold = args.correct_threshold if args.correct_threshold != 0.7 else ckpt.get("correct_threshold", 0.7)
        correct_topk = args.correct_topk if args.correct_topk != 3 else ckpt.get("correct_topk", 3)
        max_templates = args.max_templates if args.max_templates != 30 else ckpt.get("max_templates", 30)
        error_threshold = args.error_threshold if args.error_threshold != 0.7 else ckpt.get("error_threshold", 0.7)
        error_topk = args.error_topk if args.error_topk != 1 else ckpt.get("error_topk", 1)
        max_retries = args.max_retries if args.max_retries != 3 else ckpt.get("max_retries", 3)
        if args.dataset and args.dataset != ckpt.get("dataset"):
            init_instruction, output_format = get_instruction_and_format(dataset)
        else:
            init_instruction = ckpt.get("init_instruction") or get_instruction_and_format(dataset)[0]
            output_format = ckpt.get("output_format") or get_instruction_and_format(dataset)[1]
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        assert args.ctm_json_path and args.epm_json_path, \
            "必须传 --checkpoint 或同时传 --ctm_json_path 和 --epm_json_path"
        assert args.ctm_collection and args.epm_collection, \
            "必须传 --checkpoint 或同时传 --ctm_collection 和 --epm_collection"
        ctm_json_path = args.ctm_json_path
        epm_json_path = args.epm_json_path
        ctm_collection = args.ctm_collection
        epm_collection = args.epm_collection
        embed_model = args.embed_model
        dataset = args.dataset
        llm_model = args.llm_model
        llm_temperature = args.llm_temperature
        correct_threshold = args.correct_threshold
        correct_topk = args.correct_topk
        max_templates = args.max_templates
        error_threshold = args.error_threshold
        error_topk = args.error_topk
        max_retries = args.max_retries
        init_instruction, output_format = get_instruction_and_format(dataset)

    print(f"db_dir:           {db_dir}")
    print(f"ctm_collection:   {ctm_collection}")
    print(f"epm_collection:   {epm_collection}")
    print(f"ctm_json_path:    {ctm_json_path}")
    print(f"epm_json_path:    {epm_json_path}")

    memapo = MemAPO.from_checkpoint(
        ctm_json_path=ctm_json_path,
        epm_json_path=epm_json_path,
        db_dir=db_dir,
        ctm_collection=ctm_collection,
        epm_collection=epm_collection,
        emb_model=embed_model,
        correct_threshold=correct_threshold,
        correct_topk=correct_topk,
        max_templates=max_templates,
        error_threshold=error_threshold,
        error_topk=error_topk,
        model_name=llm_model,
        temperature=llm_temperature,
        init_instruction=init_instruction,
        output_format=output_format,
        eval_fn=get_eval_fn(dataset, judge_model=args.judge_model, judge_threshold=args.judge_threshold),
        max_attempts=max_retries,
    )

    _, _, test_set, _ = load_task(dataset, evaluation_api=None, data_dir=data_dir, category=args.category)
    print(f"Test set size: {len(test_set)}")

    test_save_path = f"{log_dir}/infer_results.json"
    print(f"num_workers: {args.num_workers}")
    summary, results = memapo.test(test_set, test_save_path, num_workers=args.num_workers)
    print(f"Accuracy: {summary['correct']}/{summary['total']} = {summary['accuracy']:.4f}")
    print(f"Results saved to: {test_save_path}")

    # 当 dataset 为 mt_bench 时，额外保存 MT-Bench 标准 jsonl 格式
    if dataset == "mt_bench":
        import time
        import uuid
        mt_data_path = data_dir / "mt_bench" / f"{args.category}_test.jsonl"
        raw_items = []
        with open(mt_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_items.append(json.loads(line))

        model_id = f"memapo-{llm_model}"
        if args.exp_name:
            model_id = f"memapo-{args.exp_name}"
        mtbench_jsonl_path = f"{log_dir}/{model_id}.jsonl"
        with open(mtbench_jsonl_path, "w", encoding="utf-8") as f:
            for item, record in zip(raw_items, results):
                ans = {
                    "question_id": item["question_id"],
                    "answer_id": uuid.uuid4().hex[:22],
                    "model_id": model_id,
                    "choices": [{"index": 0, "turns": [record["pred"]]}],
                    "tstamp": time.time(),
                }
                f.write(json.dumps(ans, ensure_ascii=False) + "\n")
        print(f"MT-Bench answer file saved to: {mtbench_jsonl_path}")

    _logger = get_logger() if not args.disable_logging else None
    token_meter.report(logger=_logger)
    print("Done!", flush=True)