import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from uuid import uuid4

from storage import VectorStore
from utils import get_id, get_timestamp, extract_json
from prompts import (
    build_create_template_sys_prompt,
    build_create_template_user_prompt,
    build_update_templates_sys_prompt,
    build_update_templates_user_prompt,
)

@dataclass
class GoodCase:
    question: str
    ground_truth: str
    correct_pred: str
    idx: str = field(default_factory=lambda: get_id(prefix="good_case"))

    def to_dict(self):
        return {
            "idx": self.idx,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "correct_pred": self.correct_pred,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GoodCase":
        return cls(
            question=d["question"],
            ground_truth=d["ground_truth"],
            correct_pred=d["correct_pred"],
            idx=d["idx"],
        )

class Template:
    def __init__(self, when_to_use:str, strategy:str, good_cases:List[GoodCase]):
        self.idx = get_id(prefix="template")
        self.when_to_use = when_to_use
        self.strategy = strategy
        self.good_cases = good_cases
        self.created_timestamp = None
        self.updated_timestamp = None

    def to_dict(self):
        return {
            "idx": self.idx,
            "when_to_use": self.when_to_use,
            "strategy": self.strategy,
            "good_cases": [gc.to_dict() for gc in self.good_cases],
            "created_timestamp": self.created_timestamp,
            "updated_timestamp": self.updated_timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Template":
        good_cases = [GoodCase.from_dict(gc) for gc in d["good_cases"]]
        obj = cls(when_to_use=d["when_to_use"], strategy=d["strategy"], good_cases=good_cases)
        obj.idx = d["idx"]
        obj.created_timestamp = d.get("created_timestamp")
        obj.updated_timestamp = d.get("updated_timestamp")
        return obj

class CorrectTemplateMemory:
    def __init__(self, restore_path:str="./db", collection_name:str=None, emb_model:str=None, threshold:float=0.7, topk:int=3):
        self.all_templates = {}
        self.db = VectorStore(restore_path, collection_name, emb_model, threshold, topk)
    
    def add_template(self, template:Template):
        doc_id = template.idx
        doc_content = template.when_to_use
        now_timestamp = get_timestamp()
        template.created_timestamp = now_timestamp
        template.updated_timestamp = now_timestamp
        doc_metadata = {
            "id": doc_id,
            "created_timestamp": now_timestamp,
            "updated_timestamp": now_timestamp,
            "type": "template",
            "when_to_use": template.when_to_use,
            "strategy": template.strategy,
        }
        self.db.add(doc_id, doc_content, doc_metadata)
        self.all_templates[doc_id] = template

    def delete_template(self, template_id:str):
        if template_id in self.all_templates:
            del self.all_templates[template_id]
            self.db.delete(template_id)

    def update_template(self, template_id:str, when_to_use:str=None, strategy:str=None, good_case:GoodCase=None):
        if template_id not in self.all_templates:
            return
        template = self.all_templates[template_id]
        
        need_db_update = False
        if when_to_use is not None and when_to_use != "":
            template.when_to_use = when_to_use
            need_db_update = True
        if strategy is not None and strategy != "":
            template.strategy = strategy
            need_db_update = True
        if good_case is not None:
            template.good_cases.append(good_case)

        if need_db_update:
            template.updated_timestamp = get_timestamp()
            doc_metadata = {
                "id": template_id,
                "created_timestamp": template.created_timestamp,
                "updated_timestamp": template.updated_timestamp,
                "type": "template",
                "when_to_use": template.when_to_use,
                "strategy": template.strategy,
            }
            self.db.update(template_id, template.when_to_use, doc_metadata)

    def retrieve(self, question:str, *args, **kwargs)->List[Template]:
        # threshold, topk的参数是在初始化向量数据库的时候传入的
        retrieved_results = self.db.query_topk_threshold(query=question)
        if len(retrieved_results) == 0:
            return []
        template_ids = [res["metadata"]["id"] for res in retrieved_results]
        templates = [self.all_templates[tid] for tid in template_ids if tid in self.all_templates]
        return templates
    
    def save_to_json(self, path: str):
        data = {
            "all_templates": {k: v.to_dict() for k, v in self.all_templates.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.all_templates = {
            k: Template.from_dict(v) for k, v in data["all_templates"].items()
        }

    def update(self, used_templates, question:str=None, ground_truth:str=None, correct_pred:str=None, reflections:list=None, client=None, *args, **kwargs):
        if len(used_templates) == 0:
            sys_prompt = build_create_template_sys_prompt()
            user_prompt = build_create_template_user_prompt(question, correct_pred, reflections)
            raw = client.generate(user_prompt, sys_prompt)
            result = extract_json(raw)

            if result:
                when_to_use = result.get("when_to_use", "")
                strategy = result.get("strategy", "")
            else:
                when_to_use = raw
                strategy = raw

            good_case = GoodCase(
                question=question,
                ground_truth=ground_truth,
                correct_pred=correct_pred,
            )
            new_template = Template(
                when_to_use=when_to_use,
                strategy=strategy,
                good_cases=[good_case],
            )
            self.add_template(new_template)
        else:
            # 有模板被召回，让 LLM 决定对召回的模板执行什么操作
            sys_prompt = build_update_templates_sys_prompt()
            user_prompt, id_mapping = build_update_templates_user_prompt(
                recalled_templates=used_templates,
                question=question,
                correct_pred=correct_pred,
                reflections=reflections,
            )
            raw = client.generate(user_prompt, sys_prompt)
            result = extract_json(raw)
            if not result:
                return

            actions = result.get("actions", [])
            if not isinstance(actions, list):
                return

            good_case = GoodCase(
                question=question,
                ground_truth=ground_truth,
                correct_pred=correct_pred,
            )
            processed_ids = set()

            for item in actions:
                if not isinstance(item, dict):
                    continue
                action = item.get("action", "")

                if action == "add":
                    wtu = item.get("when_to_use", "")
                    stg = item.get("strategy", "")
                    if not wtu or not stg:
                        continue
                    new_template = Template(
                        when_to_use=wtu,
                        strategy=stg,
                        good_cases=[good_case],
                    )
                    self.add_template(new_template)

                elif action == "update":
                    seq_id = str(item.get("template_id", ""))
                    template_id = id_mapping.get(seq_id, "")
                    if not template_id or template_id not in self.all_templates:
                        continue
                    if template_id in processed_ids:
                        continue
                    processed_ids.add(template_id)
                    self.update_template(
                        template_id=template_id,
                        when_to_use=item.get("when_to_use",""),
                        strategy=item.get("strategy",""),
                        good_case=good_case,
                    )

                elif action == "delete":
                    seq_id = str(item.get("template_id", ""))
                    template_id = id_mapping.get(seq_id, "")
                    if not template_id or template_id not in self.all_templates:
                        continue
                    if template_id in processed_ids:
                        continue
                    processed_ids.add(template_id)
                    self.delete_template(template_id)

                elif action == "none":
                    seq_id = str(item.get("template_id", ""))
                    template_id = id_mapping.get(seq_id, "")
                    if template_id:
                        processed_ids.add(template_id)