import json
import os
import random

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
from uuid import uuid4

from utils import get_id, get_timestamp, extract_json, get_logger
from storage import VectorStore
from prompts import (
    build_update_error_pattern_sys_prompt,
    build_update_error_pattern_user_prompt,
)

@dataclass
class BadCase:
    question: str
    ground_truth: str
    wrong_pred: str
    reflection: str = None
    idx: str = field(default_factory=lambda: get_id(prefix="bad_case"))

    def to_dict(self):
        return {
            "idx": self.idx,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "wrong_pred": self.wrong_pred,
            "reflection": self.reflection,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BadCase":
        obj = cls(
            question=d["question"],
            ground_truth=d["ground_truth"],
            wrong_pred=d["wrong_pred"],
            reflection=d.get("reflection"),
            idx=d["idx"],
        )
        return obj

class ErrorPattern:
    def __init__(self, pattern:str, bad_cases:List[BadCase]):
        self.idx = get_id(prefix="error_pattern")
        self.pattern = pattern
        self.bad_cases = bad_cases
        self.created_timestamp = None
        self.updated_timestamp = None

    def to_dict(self):
        return {
            "idx": self.idx,
            "pattern": self.pattern,
            "bad_cases": [bc.to_dict() for bc in self.bad_cases],
            "created_timestamp": self.created_timestamp,
            "updated_timestamp": self.updated_timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ErrorPattern":
        bad_cases = [BadCase.from_dict(bc) for bc in d["bad_cases"]]
        obj = cls(pattern=d["pattern"], bad_cases=bad_cases)
        obj.idx = d["idx"]
        obj.created_timestamp = d.get("created_timestamp")
        obj.updated_timestamp = d.get("updated_timestamp")
        return obj
        
class ErrorPatternMemory:
    def __init__(self, restore_path:str="./db", collection_name:str=None, emb_model:str=None, threshold:float=None, topk:int=None):
        self.all_bad_cases:Dict[str, BadCase] = {}
        self.error_pattern_clusters = {}
        self.sample_k = 3
        self.db = VectorStore(restore_path, collection_name, emb_model, threshold, topk)

    def add_bad_case(self, bad_case:BadCase, client=None):
        bad_case_id = bad_case.idx
        self.all_bad_cases[bad_case_id] = bad_case
        _log = not int(os.environ.get('disable_logging', '0'))
        if _log:
            logger = get_logger()
            logger.info("[EPM] add_bad_case | id=%s | question=%s | reflection=%s",
                        bad_case_id, bad_case.question[:100], (bad_case.reflection or '')[:100])
        retrieved_results = self.db.query_topk_threshold(query=bad_case.reflection)
        if len(retrieved_results) == 0:
            # TODO 直接拿反思作为簇的描述，先这样写吧，看后续会不会改
            new_pattern_description = bad_case.reflection
            new_bad_cases = [bad_case]
            new_error_pattern = ErrorPattern(new_pattern_description, new_bad_cases)
            if _log:
                logger.info("[EPM] add_bad_case | no matching pattern found -> creating new error_pattern | pattern=%s",
                            new_pattern_description[:100] if new_pattern_description else 'None')
            self.add_error_pattern(new_error_pattern)
        else:
            matched_pattern_id = retrieved_results[0]["metadata"]["id"]
            matched_pattern = self.error_pattern_clusters[matched_pattern_id]
            if _log:
                logger.info("[EPM] add_bad_case | matched existing pattern | pattern_id=%s | similarity=%.4f | pattern=%s -> updating",
                            matched_pattern_id, retrieved_results[0]["similarity"], matched_pattern.pattern[:100])
            self.update_error_pattern(matched_pattern, bad_case, client)

    def add_error_pattern(self, error_pattern:ErrorPattern):
        doc_id = error_pattern.idx
        doc_content = error_pattern.pattern
        now_timestamp = get_timestamp()
        doc_metadata = {
            "id": doc_id,
            "created_timestamp": now_timestamp,
            "updated_timestamp": now_timestamp,
            "type": "error_pattern",
            "pattern": error_pattern.pattern,
        }
        error_pattern.created_timestamp = now_timestamp
        error_pattern.updated_timestamp = now_timestamp
        self.db.add(doc_id, doc_content, doc_metadata)
        self.error_pattern_clusters[doc_id] = error_pattern
        if not int(os.environ.get('disable_logging', '0')):
            logger = get_logger()
            logger.info("[EPM] add_error_pattern | id=%s | pattern=%s | bad_cases=%d | total_clusters=%d",
                        doc_id, error_pattern.pattern[:100] if error_pattern.pattern else 'None',
                        len(error_pattern.bad_cases), len(self.error_pattern_clusters))

    def update_error_pattern(self, error_pattern:ErrorPattern, bad_case:BadCase, client=None):
        _log = not int(os.environ.get('disable_logging', '0'))
        historical = list(error_pattern.bad_cases)
        sampled = random.sample(historical, min(self.sample_k, len(historical)))
        if _log:
            logger = get_logger()
            logger.info("[EPM] update_error_pattern START | pattern_id=%s | old_pattern=%s | historical_bad_cases=%d | sampled=%d | new_bad_case_id=%s",
                        error_pattern.idx, error_pattern.pattern[:100], len(historical), len(sampled), bad_case.idx)

        sys_prompt = build_update_error_pattern_sys_prompt()
        user_prompt = build_update_error_pattern_user_prompt(
            current_pattern=error_pattern.pattern,
            new_bad_case=bad_case,
            historical_bad_cases=sampled,
        )
        raw = client.generate(user_prompt, sys_prompt)
        result = extract_json(raw)
        if all(existing.idx != bad_case.idx for existing in error_pattern.bad_cases):
            error_pattern.bad_cases.append(bad_case)

        need_update = False
        if result:
            need_update = result.get("updated", False)
        
        if need_update:
            new_pattern = result.get("pattern", raw)
            old_pattern = error_pattern.pattern
            error_pattern.pattern = new_pattern
            error_pattern.updated_timestamp = get_timestamp()
            doc_metadata = {
                "id": error_pattern.idx,
                "created_timestamp": error_pattern.created_timestamp,
                "updated_timestamp": error_pattern.updated_timestamp,
                "type": "error_pattern",
                "pattern": new_pattern,
            }
            self.db.update(error_pattern.idx, new_pattern, doc_metadata)
            if _log:
                logger.info("[EPM] update_error_pattern | pattern_id=%s | pattern UPDATED | old='%s' | new='%s' | bad_cases_count=%d",
                            error_pattern.idx, old_pattern[:80], new_pattern[:80], len(error_pattern.bad_cases))
        else:
            if _log:
                logger.info("[EPM] update_error_pattern | pattern_id=%s | LLM decided NO UPDATE needed (updated=False) | bad_cases_count=%d",
                            error_pattern.idx, len(error_pattern.bad_cases))

    def retrieve(self, question:str, *args, **kwargs)->List[ErrorPattern]:
        # 当前只考虑最简单的实现，召回所有的error pattern
        patterns = list(self.error_pattern_clusters.values())
        if not int(os.environ.get('disable_logging', '0')):
            logger = get_logger()
            logger.info("[EPM] retrieve | question=%s | returned_all_patterns=%d | pattern_ids=%s",
                        question[:100], len(patterns), [p.idx for p in patterns])
        return patterns
    
    def save_to_json(self, path: str):
        data = {
            "all_bad_cases": {k: v.to_dict() for k, v in self.all_bad_cases.items()},
            "error_pattern_clusters": {k: v.to_dict() for k, v in self.error_pattern_clusters.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.all_bad_cases = {
            k: BadCase.from_dict(v) for k, v in data["all_bad_cases"].items()
        }
        self.error_pattern_clusters = {
            k: ErrorPattern.from_dict(v) for k, v in data["error_pattern_clusters"].items()
        }

    @classmethod
    def from_checkpoint(cls, json_path: str, restore_path: str = "./db", collection_name: str = None,
                        emb_model: str = None, threshold: float = None, topk: int = None):
        """从之前保存的 JSON + 持久化 DB 恢复 ErrorPatternMemory。"""
        obj = cls(restore_path, collection_name, emb_model, threshold, topk)
        obj.load_from_json(json_path)
        return obj

    def update(self, question:str, ground_truth:str, wrong_pred:str, reflection:str=None, client=None, *args, **kwargs):
        if not int(os.environ.get('disable_logging', '0')):
            logger = get_logger()
            logger.info("[EPM] update START | question=%s | reflection=%s",
                        question[:100], (reflection or '')[:100])
        bad_case = BadCase(
            question=question,
            ground_truth=ground_truth,
            wrong_pred=wrong_pred,
            reflection=reflection
        )
        self.add_bad_case(bad_case, client)