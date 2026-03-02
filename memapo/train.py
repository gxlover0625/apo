import json
import numpy as np
import random
import os

from dataclasses import dataclass, asdict
from transformers import HfArgumentParser
from pathlib import Path

from utils import get_logger, get_timestamp
from memapo_v2 import MemAPO
from retriever import Retriever
from evaluator import Evaluator
from updater import Updater
from ctm import CorrectTemplateMemory
from epm import ErrorPatternMemory
from task import get_eval_fn

from textgrad.tasks import load_task

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
    log_dir_abs = str(Path(args.log_dir).resolve())
    if not args.disable_logging:
        log_file = f"{log_dir_abs}/{args.exp_name}_{timestamp}_train.log"
        logger = get_logger(log_file)
        logger.info("Args:\n%s", json.dumps(asdict(args), indent=2, ensure_ascii=False, sort_keys=True))
    
    # collection_name = f"{args.exp_name}_{timestamp}"
    # db = VectorStore(args.db_dir, collection_name, emb_model=args.embed_model, threshold=args.correct_threshold, topk=args.correct_topk)
    # db.add(doc_content="hello", doc_metadata={"meta": "test"})
    # print(db.query_topk("hello world"))
    exp_timestamp = get_timestamp()

    # 路径全部转绝对路径
    log_dir = str(Path(args.log_dir).resolve())
    db_dir = str(Path(args.db_dir).resolve())
    data_dir = Path(__file__).resolve().parent.parent / "data"

    ctm_collection = f"{args.exp_name}_{exp_timestamp}_correct_template_memory"
    epm_collection = f"{args.exp_name}_{exp_timestamp}_error_pattern_memory"

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

    # 保存两个memory的dict
    ctm_save_path = f"{log_dir}/{args.exp_name}_{exp_timestamp}_ctm.json"
    epm_save_path = f"{log_dir}/{args.exp_name}_{exp_timestamp}_epm.json"
    correct_template_memory.save_to_json(ctm_save_path)
    error_pattern_memory.save_to_json(epm_save_path)
    print(f"CTM saved to:     {ctm_save_path}")
    print(f"EPM saved to:     {epm_save_path}")

    test_save_path = f"{log_dir}/{args.exp_name}_{exp_timestamp}_test_results.json"
    memapo.test(test_set, test_save_path)
    print("Done!")