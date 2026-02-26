import json
import numpy as np
import random

from dataclasses import dataclass, asdict
from datetime import datetime
from transformers import HfArgumentParser

from utils import get_logger

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

    # 配置logger
    if not args.disable_logging:
        log_file = f"{args.log_dir}/{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_train.log"
        logger = get_logger(log_file)
        logger.info("Args:\n%s", json.dumps(asdict(args), indent=2, ensure_ascii=False, sort_keys=True))
    
    seed_everything(args.seed)