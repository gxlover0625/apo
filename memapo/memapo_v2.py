from retriever import Retriever
from evaluator import Evaluator
from updater import Updater
from ctm import CorrectTemplateMemory
from epm import ErrorPatternMemory
from llm import LLMFactory
from utils import extract_json, get_logger
from prompts import (
    build_generation_sys_prompt, 
    build_generation_user_prompt,
    build_reflection_sys_prompt,
    build_reflection_user_prompt,
    build_summarize_reflection_sys_prompt,
    build_summarize_reflection_user_prompt,
)

class MemAPO:
    def __init__(
        self, 
        retriever:Retriever, 
        evaluator:Evaluator, 
        updater:Updater,
        model_name:str=None,
        temperature:float=0.,
        init_instruction:str=None,
        output_format:str=None,
        eval_fn=None,
        max_attempts:int=3,
        **kwargs
    ):
        self.retriever = retriever
        self.evaluator = evaluator
        self.updater = updater
        self.client = LLMFactory.get_llm(model_name, temperature)
        self.init_instruction = init_instruction
        self.output_format = output_format
        self.eval_fn = eval_fn
        self.max_attempts = max_attempts

    @classmethod
    def from_checkpoint(
        cls,
        ctm_json_path: str,
        epm_json_path: str,
        db_dir: str = "./db",
        ctm_collection: str = None,
        epm_collection: str = None,
        emb_model: str = None,
        correct_threshold: float = 0.7,
        correct_topk: int = 3,
        max_templates: int = 30,
        error_threshold: float = 0.7,
        error_topk: int = 1,
        model_name: str = None,
        temperature: float = 0.,
        init_instruction: str = None,
        output_format: str = None,
        eval_fn=None,
        max_attempts: int = 3,
        **kwargs,
    ) -> "MemAPO":
        """从之前训练保存的 checkpoint 恢复完整的 MemAPO 实例（含 memory + DB）。"""
        ctm = CorrectTemplateMemory.from_checkpoint(
            ctm_json_path, db_dir, ctm_collection, emb_model,
            correct_threshold, correct_topk, max_templates,
        )
        epm = ErrorPatternMemory.from_checkpoint(
            epm_json_path, db_dir, epm_collection, emb_model,
            error_threshold, error_topk,
        )
        retriever = Retriever(ctm, epm)
        evaluator = Evaluator()
        updater = Updater(ctm, epm)
        return cls(
            retriever, evaluator, updater,
            model_name, temperature,
            init_instruction, output_format,
            eval_fn, max_attempts, **kwargs,
        )

    def process_train_sample(self, sample, *args, **kwargs):
        import os
        _log = not int(os.environ.get('disable_logging', '0'))
        if _log:
            logger = get_logger()

        # Stage-1: Memory Retrieve
        question, ground_truth = sample
        templates, error_patterns = self.retriever.retrieve(question, *args, **kwargs)
        if _log:
            logger.info("[MemAPO] Stage-1 Retrieve | question=%s | retrieved_templates=%d | retrieved_error_patterns=%d",
                        question[:100], len(templates), len(error_patterns))

        # Stage-2: Evaluate Prompt
        gen_sys_prompt = build_generation_sys_prompt(self.init_instruction, error_patterns)
        reflect_sys_prompt = build_reflection_sys_prompt()
        reflections = []
        correct_update = False
        for attempt in range(1, self.max_attempts + 1):
            gen_user_prompt = build_generation_user_prompt(
                question, self.output_format, templates, reflections
            )
            pred = self.client.generate(gen_user_prompt, gen_sys_prompt)
            is_correct = self.eval_fn(pred, ground_truth)
            if _log:
                logger.info("[MemAPO] Stage-2 Attempt %d/%d | question=%s | pred=%s | is_correct=%s",
                            attempt, self.max_attempts, question[:80], pred[:80], is_correct)
            if is_correct:
                correct_update = True
                if _log:
                    logger.info("[MemAPO] Stage-2 | attempt %d CORRECT -> proceeding to update correct memory", attempt)
                break

            reflection_user_prompt = build_reflection_user_prompt(question, pred, reflections)
            reflection_raw = self.client.generate(reflection_user_prompt, reflect_sys_prompt)
            reflection_json = extract_json(reflection_raw)
            reflection_text = reflection_json["reflection"] if reflection_json else reflection_raw
            if _log:
                logger.info("[MemAPO] Stage-2 | attempt %d WRONG -> reflection extracted (json=%s): %s",
                            attempt, reflection_json is not None, reflection_text[:100])
            reflections.append({
                "attempt": attempt,
                "wrong_pred": pred,
                "reflection": reflection_text,
            })

        # Stage-3: Update Memory
        if correct_update:
            if _log:
                logger.info("[MemAPO] Stage-3 Update | path=CORRECT_MEMORY | question=%s | correct_pred=%s | reflections_before_correct=%d",
                            question[:80], pred[:80], len(reflections))
            self.updater.update_correct_memory(
                used_templates=templates,
                question=question,
                ground_truth=ground_truth,
                correct_pred=pred,
                reflections=reflections,
                client=self.client,
                eval_fn=self.eval_fn,
                init_instruction=self.init_instruction,
                output_format=self.output_format,
            )
        else:
            if _log:
                logger.info("[MemAPO] Stage-3 Update | path=ERROR_MEMORY | question=%s | all %d attempts FAILED | summarizing reflections",
                            question[:80], self.max_attempts)
            sum_reflect_sys_prompt = build_summarize_reflection_sys_prompt()
            sum_reflect_user_prompt = build_summarize_reflection_user_prompt(
                question=question,
                ground_truth=ground_truth,
                reflections=reflections,
            )
            summary_raw = self.client.generate(sum_reflect_user_prompt, sum_reflect_sys_prompt)
            summary_json = extract_json(summary_raw)
            final_reflection = summary_json["reflection"] if summary_json else summary_raw
            if _log:
                logger.info("[MemAPO] Stage-3 | final_reflection (json=%s): %s -> updating error memory",
                            summary_json is not None, final_reflection[:100])

            self.updater.update_error_memory(
                question=question,
                ground_truth=ground_truth,
                wrong_pred=pred,
                reflection=final_reflection,
                client=self.client
            )
    
    def train(self, train_set:list, *args, **kwargs):
        for sample in train_set:
            self.process_train_sample(sample, *args, **kwargs)

    def process_test_sample(self, sample, *args, **kwargs):
        import os
        _log = not int(os.environ.get('disable_logging', '0'))
        if _log:
            logger = get_logger()

        question, ground_truth = sample
        templates, error_patterns = self.retriever.retrieve(question, *args, **kwargs)

        gen_sys_prompt = build_generation_sys_prompt(self.init_instruction, error_patterns)
        gen_user_prompt = build_generation_user_prompt(question, self.output_format, templates, [])
        pred = self.client.generate(gen_user_prompt, gen_sys_prompt)
        is_correct = self.eval_fn(pred, ground_truth)

        if _log:
            logger.info("[MemAPO-Test] question=%s | ground_truth=%s | pred=%s | is_correct=%s | templates=%d | error_patterns=%d",
                        question[:100], ground_truth[:100], pred[:100], is_correct,
                        len(templates), len(error_patterns))

        return {
            "question": question,
            "ground_truth": ground_truth,
            "pred": pred,
            "is_correct": is_correct,
            "sys_prompt": gen_sys_prompt,
            "user_prompt": gen_user_prompt,
            "retrieved_templates": len(templates),
            "retrieved_error_patterns": len(error_patterns),
        }

    def test(self, test_set:list, save_path:str=None, *args, **kwargs):
        import json
        import os
        from pathlib import Path
        _log = not int(os.environ.get('disable_logging', '0'))
        if _log:
            logger = get_logger()

        results = []
        correct_count = 0
        total_count = 0

        for sample in test_set:
            record = self.process_test_sample(sample, *args, **kwargs)
            results.append(record)
            total_count += 1
            if record["is_correct"]:
                correct_count += 1

            if _log:
                logger.info("[MemAPO-Test] progress=%d/%d | running_accuracy=%.4f",
                            total_count, len(test_set), correct_count / total_count)

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        summary = {
            "total": total_count,
            "correct": correct_count,
            "wrong": total_count - correct_count,
            "accuracy": accuracy,
        }

        if _log:
            logger.info("[MemAPO-Test] DONE | total=%d | correct=%d | wrong=%d | accuracy=%.4f",
                        total_count, correct_count, total_count - correct_count, accuracy)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)
            if _log:
                logger.info("[MemAPO-Test] results saved to %s", save_path)

        return summary, results