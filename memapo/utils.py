import logging

from uuid import uuid4
from datetime import datetime
from pathlib import Path

def get_logger(log_file:str=None):
    logger = logging.getLogger("memapo")
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_id(prefix:str=None):
    uid = str(uuid4())
    if prefix:
        return f"{prefix}_{uid}"
    return uid