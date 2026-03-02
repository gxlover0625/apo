import json
import os
import random

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from uuid import uuid4

from storage import VectorStore
from utils import get_id, get_timestamp, extract_json, get_logger
from prompts import (
    build_create_template_sys_prompt,
    build_create_template_user_prompt,
    build_update_templates_sys_prompt,
    build_update_templates_user_prompt,
    build_generation_sys_prompt,
    build_generation_user_prompt,
    build_merge_templates_sys_prompt,
    build_merge_templates_user_prompt,
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
    def __init__(self, restore_path:str="./db", collection_name:str=None, emb_model:str=None, threshold:float=0.7, topk:int=3, max_templates:int=30):
        self.all_templates = {}
        self.max_templates = max_templates
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
        if not int(os.environ.get('disable_logging', '0')):
            logger = get_logger()
            logger.info("[CTM] add_template | id=%s | when_to_use=%s | total_templates=%d", doc_id, template.when_to_use[:100], len(self.all_templates))

    def delete_template(self, template_id:str):
        if template_id in self.all_templates:
            template = self.all_templates[template_id]
            del self.all_templates[template_id]
            self.db.delete(template_id)
            if not int(os.environ.get('disable_logging', '0')):
                logger = get_logger()
                logger.info("[CTM] delete_template | id=%s | when_to_use=%s | total_templates=%d", template_id, template.when_to_use[:100], len(self.all_templates))

    def update_template(self, template_id:str, when_to_use:str=None, strategy:str=None):
        if template_id not in self.all_templates:
            return
        template = self.all_templates[template_id]
        _log = not int(os.environ.get('disable_logging', '0'))
        
        need_db_update = False
        changed_fields = []
        if when_to_use is not None and when_to_use != "":
            if _log:
                changed_fields.append(f"when_to_use: '{template.when_to_use[:60]}' -> '{when_to_use[:60]}'")
            template.when_to_use = when_to_use
            need_db_update = True
        if strategy is not None and strategy != "":
            if _log:
                changed_fields.append(f"strategy: '{template.strategy[:60]}' -> '{strategy[:60]}'")
            template.strategy = strategy
            need_db_update = True

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
            if _log:
                logger = get_logger()
                logger.info("[CTM] update_template | id=%s | changes=[%s]", template_id, '; '.join(changed_fields))
        else:
            if _log:
                logger = get_logger()
                logger.info("[CTM] update_template | id=%s | no fields changed, skipping DB update", template_id)

    def _validate_template(self, template:Template, client, eval_fn, init_instruction:str, output_format:str, sample_k:int=3) -> bool:
        if not template.good_cases or eval_fn is None:
            return True
        samples = random.sample(template.good_cases, min(sample_k, len(template.good_cases)))
        gen_sys_prompt = build_generation_sys_prompt(init_instruction, [])
        for gc in samples:
            gen_user_prompt = build_generation_user_prompt(gc.question, output_format, [template], [])
            pred = client.generate(gen_user_prompt, gen_sys_prompt)
            if not eval_fn(pred, gc.ground_truth):
                if not int(os.environ.get('disable_logging', '0')):
                    logger = get_logger()
                    logger.info("[CTM] _validate_template FAILED | template_id=%s | question=%s", template.idx, gc.question[:100])
                return False
        if not int(os.environ.get('disable_logging', '0')):
            logger = get_logger()
            logger.info("[CTM] _validate_template PASSED | template_id=%s | validated=%d samples", template.idx, len(samples))
        return True

    def retrieve(self, question:str, *args, **kwargs)->List[Template]:
        # threshold, topk的参数是在初始化向量数据库的时候传入的
        retrieved_results = self.db.query_topk_threshold(query=question)
        if len(retrieved_results) == 0:
            if not int(os.environ.get('disable_logging', '0')):
                logger = get_logger()
                logger.info("[CTM] retrieve | question=%s | recalled=0", question[:100])
            return []
        template_ids = [res["metadata"]["id"] for res in retrieved_results]
        templates = [self.all_templates[tid] for tid in template_ids if tid in self.all_templates]
        if not int(os.environ.get('disable_logging', '0')):
            logger = get_logger()
            logger.info("[CTM] retrieve | question=%s | recalled=%d | ids=%s", question[:100], len(templates), [t.idx for t in templates])
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

    def update(self, used_templates, question:str=None, ground_truth:str=None, correct_pred:str=None, reflections:list=None, client=None, eval_fn=None, init_instruction:str=None, output_format:str=None, *args, **kwargs):
        _log = not int(os.environ.get('disable_logging', '0'))
        if _log:
            logger = get_logger()
            logger.info("[CTM] update START | question=%s | recalled_templates=%d | has_reflections=%d",
                        question[:100] if question else 'None', len(used_templates), len(reflections) if reflections else 0)
        if len(used_templates) == 0:
            if _log:
                logger.info("[CTM] update | no templates recalled -> creating new template via LLM")
            sys_prompt = build_create_template_sys_prompt()
            user_prompt = build_create_template_user_prompt(question, correct_pred, reflections)
            raw = client.generate(user_prompt, sys_prompt)
            result = extract_json(raw)

            if result:
                when_to_use = result.get("when_to_use", "")
                strategy = result.get("strategy", "")
                if _log:
                    logger.info("[CTM] update | LLM returned valid JSON | when_to_use=%s | strategy=%s", when_to_use[:100], strategy[:100])
            else:
                when_to_use = raw
                strategy = raw
                if _log:
                    logger.info("[CTM] update | LLM returned non-JSON, using raw text as when_to_use and strategy")

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
            if _log:
                logger.info("[CTM] update | %d templates recalled -> asking LLM for update actions | recalled_ids=%s",
                            len(used_templates), [t.idx for t in used_templates])
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
                if _log:
                    logger.info("[CTM] update | LLM returned no valid JSON for actions, aborting update")
                return

            actions = result.get("actions", [])
            if not isinstance(actions, list):
                if _log:
                    logger.info("[CTM] update | 'actions' field is not a list (got %s), aborting update", type(actions).__name__)
                return

            good_case = GoodCase(
                question=question,
                ground_truth=ground_truth,
                correct_pred=correct_pred,
            )
            processed_ids = set()
            if _log:
                logger.info("[CTM] update | LLM proposed %d actions: %s",
                            len(actions), [a.get('action', '?') if isinstance(a, dict) else '?' for a in actions])

            for item in actions:
                if not isinstance(item, dict):
                    continue
                action = item.get("action", "")

                if action == "add":
                    wtu = item.get("when_to_use", "")
                    stg = item.get("strategy", "")
                    if not wtu or not stg:
                        if _log:
                            logger.info("[CTM] update | action=add skipped: missing when_to_use or strategy")
                        continue
                    new_template = Template(
                        when_to_use=wtu,
                        strategy=stg,
                        good_cases=[good_case],
                    )
                    if _log:
                        logger.info("[CTM] update | action=add | new_template_id=%s | when_to_use=%s", new_template.idx, wtu[:100])
                    self.add_template(new_template)

                elif action == "update":
                    seq_id = str(item.get("template_id", ""))
                    template_id = id_mapping.get(seq_id, "")
                    if not template_id or template_id not in self.all_templates:
                        if _log:
                            logger.info("[CTM] update | action=update skipped: seq_id=%s -> template_id=%s not found", seq_id, template_id)
                        continue
                    if template_id in processed_ids:
                        if _log:
                            logger.info("[CTM] update | action=update skipped: template_id=%s already processed", template_id)
                        continue
                    processed_ids.add(template_id)

                    # 先构造修改后的临时 template 做验证
                    original = self.all_templates[template_id]
                    new_wtu = item.get("when_to_use", "") or original.when_to_use
                    new_stg = item.get("strategy", "") or original.strategy
                    temp_template = Template(when_to_use=new_wtu, strategy=new_stg, good_cases=list(original.good_cases))
                    temp_template.idx = original.idx
                    if _log:
                        logger.info("[CTM] update | action=update | template_id=%s | validating proposed changes: when_to_use=%s | strategy=%s",
                                    template_id, new_wtu[:100], new_stg[:100])

                    if self._validate_template(temp_template, client, eval_fn, init_instruction, output_format):
                        # 验证通过，执行更新
                        if _log:
                            logger.info("[CTM] update | action=update | template_id=%s | validation PASSED -> applying update", template_id)
                        self.update_template(
                            template_id=template_id,
                            when_to_use=item.get("when_to_use", ""),
                            strategy=item.get("strategy", ""),
                        )
                    else:
                        # 验证失败，不动原 template，创建新 template
                        if _log:
                            logger.info("[CTM] update | action=update | template_id=%s | validation FAILED -> creating fallback new template", template_id)
                        fallback = Template(
                            when_to_use=new_wtu,
                            strategy=new_stg,
                            good_cases=[good_case],
                        )
                        self.add_template(fallback)

                elif action == "delete":
                    seq_id = str(item.get("template_id", ""))
                    template_id = id_mapping.get(seq_id, "")
                    if not template_id or template_id not in self.all_templates:
                        if _log:
                            logger.info("[CTM] update | action=delete skipped: seq_id=%s -> template_id=%s not found", seq_id, template_id)
                        continue
                    if template_id in processed_ids:
                        if _log:
                            logger.info("[CTM] update | action=delete skipped: template_id=%s already processed", template_id)
                        continue
                    processed_ids.add(template_id)
                    if _log:
                        logger.info("[CTM] update | action=delete | template_id=%s | when_to_use=%s",
                                    template_id, self.all_templates[template_id].when_to_use[:100])
                    self.delete_template(template_id)

                elif action == "none":
                    seq_id = str(item.get("template_id", ""))
                    template_id = id_mapping.get(seq_id, "")
                    if template_id:
                        processed_ids.add(template_id)
                        if _log:
                            logger.info("[CTM] update | action=none | template_id=%s | no changes needed", template_id)

            for t in used_templates:
                if t.idx in self.all_templates:
                    self.all_templates[t.idx].good_cases.append(good_case)
                    if _log:
                        logger.info("[CTM] update | appended good_case to template_id=%s | total_good_cases=%d",
                                    t.idx, len(self.all_templates[t.idx].good_cases))

        # 每次 update 后检查是否需要合并
        if _log:
            logger.info("[CTM] update END | total_templates=%d | max_templates=%d | merge_needed=%s",
                        len(self.all_templates), self.max_templates, len(self.all_templates) > self.max_templates)
        if len(self.all_templates) > self.max_templates:
            self.merge_templates(client, eval_fn, init_instruction, output_format)

    def merge_templates(self, client, eval_fn=None, init_instruction:str=None, output_format:str=None):
        """当模板数量超过 max_templates 时，调用 LLM 识别可合并的模板组并执行合并。"""
        _log = not int(os.environ.get('disable_logging', '0'))
        if len(self.all_templates) <= self.max_templates:
            if _log:
                logger = get_logger()
                logger.info("[CTM] merge_templates skipped | total=%d <= max=%d", len(self.all_templates), self.max_templates)
            return

        if _log:
            logger = get_logger()
            logger.info("[CTM] merge_templates START | total=%d | max=%d", len(self.all_templates), self.max_templates)

        sys_prompt = build_merge_templates_sys_prompt()
        user_prompt, id_mapping = build_merge_templates_user_prompt(self.all_templates, self.max_templates)
        raw = client.generate(user_prompt, sys_prompt)
        result = extract_json(raw)

        if not result:
            if _log:
                logger.info("[CTM] merge_templates | LLM returned no valid JSON, aborting merge")
            return

        merge_groups = result.get("merge_groups", [])
        if not isinstance(merge_groups, list) or len(merge_groups) == 0:
            if _log:
                logger.info("[CTM] merge_templates | no merge groups proposed by LLM")
            return

        if _log:
            logger.info("[CTM] merge_templates | LLM proposed %d merge groups", len(merge_groups))
        merged_count = 0
        consumed_ids = set()  # 已被前面的组合并掉的模板，避免交叉引用
        for group in merge_groups:
            if not isinstance(group, dict):
                continue

            seq_ids = group.get("template_ids", [])
            if not isinstance(seq_ids, list) or len(seq_ids) < 2:
                continue

            # 映射序号到真实 ID，过滤无效的和已被消费的
            real_ids = []
            for sid in seq_ids:
                real_id = id_mapping.get(str(sid), "")
                if real_id and real_id in self.all_templates and real_id not in consumed_ids:
                    real_ids.append(real_id)
            if len(real_ids) < 2:
                if _log:
                    logger.info("[CTM] merge_templates | group skipped: only %d valid templates after filtering (need >= 2) | seq_ids=%s", len(real_ids), seq_ids)
                continue

            merged_wtu = group.get("merged_when_to_use", "")
            merged_stg = group.get("merged_strategy", "")
            if not merged_wtu or not merged_stg:
                if _log:
                    logger.info("[CTM] merge_templates | group skipped: missing merged_when_to_use or merged_strategy | real_ids=%s", real_ids)
                continue

            # 收集所有 good_cases
            all_good_cases = []
            for rid in real_ids:
                all_good_cases.extend(self.all_templates[rid].good_cases)

            # 构造临时模板做验证
            temp_template = Template(when_to_use=merged_wtu, strategy=merged_stg, good_cases=list(all_good_cases))

            if _log:
                logger.info("[CTM] merge_templates | validating merged template | source_ids=%s | merged_when_to_use=%s | merged_good_cases=%d",
                            real_ids, merged_wtu[:100], len(all_good_cases))
            if eval_fn and not self._validate_template(temp_template, client, eval_fn, init_instruction, output_format):
                if _log:
                    logger.info("[CTM] merge_templates | validation FAILED for group %s -> skipping merge", real_ids)
                continue

            if _log:
                logger.info("[CTM] merge_templates | validation PASSED -> deleting %d source templates and creating merged template", len(real_ids))
            # 验证通过，删除原模板，添加合并后的新模板
            for rid in real_ids:
                self.delete_template(rid)
                consumed_ids.add(rid)

            new_template = Template(
                when_to_use=merged_wtu,
                strategy=merged_stg,
                good_cases=all_good_cases,
            )
            self.add_template(new_template)
            merged_count += 1

            if _log:
                logger.info("[CTM] merge_templates | merged %s -> new_template_id=%s | merged_when_to_use=%s",
                            real_ids, new_template.idx, merged_wtu[:100])

        if _log:
            logger.info("[CTM] merge_templates DONE | groups_merged=%d | total_templates=%d", merged_count, len(self.all_templates))