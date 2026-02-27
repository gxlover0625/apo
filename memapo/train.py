import json
import numpy as np
import random
import os

from dataclasses import dataclass, asdict
from datetime import datetime
from transformers import HfArgumentParser

from utils import get_logger, get_timestamp
from storage import VectorStore
from agent import Agent

@dataclass
class MemAPOArgs:
    correct_threshold:float = 0.7
    correct_topk:int = 3

    max_retries:int = 3
    embed_model:str = ""
    llm_model:str = ""
    
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
    if not args.disable_logging:
        log_file = f"{args.log_dir}/{args.exp_name}_{timestamp}_train.log"
        logger = get_logger(log_file)
        logger.info("Args:\n%s", json.dumps(asdict(args), indent=2, ensure_ascii=False, sort_keys=True))
    
    # collection_name = f"{args.exp_name}_{timestamp}"
    # db = VectorStore(args.db_dir, collection_name, emb_model=args.embed_model, threshold=args.correct_threshold, topk=args.correct_topk)
    # db.add(doc_content="hello", doc_metadata={"meta": "test"})
    # print(db.query_topk("hello world"))

    agent = Agent(args.llm_model, temperature=0., role_description="testing")
    response = agent.chat("请介绍一下自己。")
    print(response)