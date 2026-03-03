import json
import re
import logging

from uuid import uuid4
from datetime import datetime
from pathlib import Path

class _MultiLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        header = f"{self.formatTime(record, self.datefmt)} | {record.levelname} | {record.filename}:{record.lineno}"
        msg = record.getMessage()
        parts = msg.split(" | ")
        body = "\n  ".join(parts)
        return f"{header}\n{body}"

def get_logger(log_file: str = None):
    logger = logging.getLogger("memapo")
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    formatter = _MultiLineFormatter(datefmt="%Y-%m-%d %H:%M:%S")

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

def _fix_latex_escapes(json_str: str) -> str:
    return re.sub(
        r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})',
        r'\\\\',
        json_str,
    )

def extract_json(text: str) -> dict:
    code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1)
    
    start = text.find("{")
    if start == -1:
        return None
    
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                raw = text[start:i + 1]
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    pass
                # 尝试修复 LaTeX 转义后重新解析
                try:
                    return json.loads(_fix_latex_escapes(raw))
                except json.JSONDecodeError as e:
                    print(e)
                    return None
    return None