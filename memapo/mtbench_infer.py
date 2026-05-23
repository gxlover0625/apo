"""MT-Bench direct inference & pairwise comparison.

Subcommands:
  infer    — Generate model answers (turn-1) in MT-Bench jsonl format.
  pairwise — Compare two answer files using an LLM judge and report win/loss/tie.

Usage:
    python mtbench_infer.py infer --model gpt-4o-mini-0718 --parallel 4
    python mtbench_infer.py pairwise --answer-a model_a.jsonl --answer-b model_b.jsonl --judge-model gpt-4o-mini-0718
"""
import argparse
import json
import os
import time
import threading
import concurrent.futures

import uuid
import tqdm

from llm import LLMFactory

TEMPERATURE_CONFIG = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "test", "llm_judge", "data")

PAIRWISE_SYS_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two "
    "AI assistants to the user question displayed below. You should choose the assistant that "
    "follows the user's instructions and answers the user's question better. Your evaluation "
    "should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, "
    "and level of detail of their responses. Begin your evaluation by comparing the two responses "
    "and provide a short explanation. Avoid any position biases and ensure that the order in which "
    "the responses were presented does not influence your decision. Do not allow the length of the "
    "responses to influence your evaluation. Do not favor certain names of the assistants. Be as "
    "objective as possible. After providing your explanation, output your final verdict by strictly "
    "following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, "
    "and \"[[C]]\" for a tie."
)

PAIRWISE_USER_TEMPLATE = """[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]"""

_file_lock = threading.Lock()


def load_questions(question_file, begin=None, end=None):
    questions = []
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions[begin:end]


def load_answers(answer_file):
    answers = {}
    with open(answer_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                answers[obj["question_id"]] = obj
    return answers


def reorg_answer_file(answer_file):
    answers = {}
    with open(answer_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                answers[obj["question_id"]] = line
    with open(answer_file, "w", encoding="utf-8") as f:
        for qid in sorted(answers.keys()):
            f.write(answers[qid] + "\n")


# ======================== infer ========================

def get_answer(question, llm, model_id, max_tokens, answer_file, force_temperature=None):
    if force_temperature is not None:
        temperature = force_temperature
    elif question["category"] in TEMPERATURE_CONFIG:
        temperature = TEMPERATURE_CONFIG[question["category"]]
    else:
        temperature = 0.7

    try:
        output = llm.generate(question["turns"][0], temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        print(f"ERROR question_id={question['question_id']}: {e}")
        output = "ERROR"

    ans = {
        "question_id": question["question_id"],
        "answer_id": uuid.uuid4().hex[:22],
        "model_id": model_id,
        "choices": [{"index": 0, "turns": [output]}],
        "tstamp": time.time(),
    }

    with _file_lock:
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(answer_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(ans, ensure_ascii=False) + "\n")


def cmd_infer(args):
    os.environ.setdefault("disable_logging", "1")

    model_id = args.model_id or args.model
    question_file = args.question_file or os.path.join(DATA_DIR, args.bench_name, "question.jsonl")
    output_dir = args.output_dir or os.path.join(DATA_DIR, args.bench_name, "model_answer")

    questions = load_questions(question_file, args.question_begin, args.question_end)

    categories = [c.strip() for c in args.categories.split(",")] if args.categories else None
    if categories:
        questions = [q for q in questions if q["category"] in categories]

    print(f"Loaded {len(questions)} questions from {question_file}")
    if categories:
        print(f"Categories: {categories}")

    llm = LLMFactory.get_llm(args.model, temperature=0.7)

    # 按 category 分组推理，分开保存
    from collections import defaultdict
    by_category = defaultdict(list)
    for q in questions:
        by_category[q["category"]].append(q)

    for category, cat_questions in by_category.items():
        answer_file = os.path.join(output_dir, f"{model_id}_{category}.jsonl")
        if os.path.exists(answer_file):
            os.remove(answer_file)

        print(f"\n[{category}] {len(cat_questions)} questions → {answer_file}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [
                executor.submit(get_answer, q, llm, model_id, args.max_tokens, answer_file, args.force_temperature)
                for q in cat_questions
            ]
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                future.result()

        reorg_answer_file(answer_file)

    print(f"\nDone! Answers saved to {output_dir}/")


# ======================== pairwise ========================

def judge_one(question, answer_a_text, answer_b_text, judge_llm):
    user_prompt = PAIRWISE_USER_TEMPLATE.format(
        question=question,
        answer_a=answer_a_text,
        answer_b=answer_b_text,
    )
    judgment = judge_llm.generate(user_prompt, PAIRWISE_SYS_PROMPT, temperature=0, max_tokens=2048)

    if "[[A]]" in judgment:
        winner = "A"
    elif "[[B]]" in judgment:
        winner = "B"
    elif "[[C]]" in judgment:
        winner = "tie"
    else:
        winner = "error"
    return winner, judgment


def cmd_pairwise(args):
    os.environ.setdefault("disable_logging", "1")

    question_file = args.question_file or os.path.join(DATA_DIR, args.bench_name, "question.jsonl")
    questions = load_questions(question_file, None, None)
    q_map = {q["question_id"]: q for q in questions}

    answers_a = load_answers(args.answer_a)
    answers_b = load_answers(args.answer_b)

    model_a = next(iter(answers_a.values()))["model_id"]
    model_b = next(iter(answers_b.values()))["model_id"]
    print(f"Pairwise: {model_a} (A) vs {model_b} (B)")
    print(f"Judge model: {args.judge_model}")

    common_qids = sorted(set(answers_a.keys()) & set(answers_b.keys()))
    print(f"Common questions: {len(common_qids)}")

    judge_llm = LLMFactory.get_llm(args.judge_model, temperature=0.0)

    results = []
    stats = {"A": 0, "B": 0, "tie": 0, "error": 0}

    def _judge_worker(qid):
        q = q_map[qid]
        text_a = answers_a[qid]["choices"][0]["turns"][0]
        text_b = answers_b[qid]["choices"][0]["turns"][0]
        # game 1: A vs B
        w1, j1 = judge_one(q["turns"][0], text_a, text_b, judge_llm)
        # game 2: B vs A (swap position to reduce position bias)
        w2, j2 = judge_one(q["turns"][0], text_b, text_a, judge_llm)
        # resolve
        w2_mapped = {"A": "B", "B": "A", "tie": "tie", "error": "error"}[w2]
        if w1 == w2_mapped:
            final = w1
        else:
            final = "tie"
        return qid, final, {"g1": w1, "g2": w2, "g1_judgment": j1, "g2_judgment": j2}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(_judge_worker, qid) for qid in common_qids]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            qid, final, detail = future.result()
            stats[final] += 1
            results.append({"question_id": qid, "winner": final, **detail})

    total = stats["A"] + stats["B"] + stats["tie"]
    print(f"\n{'='*50}")
    print(f"Results: {model_a} (A) vs {model_b} (B)")
    print(f"{'='*50}")
    print(f"  A wins: {stats['A']} ({stats['A']/total*100:.1f}%)")
    print(f"  B wins: {stats['B']} ({stats['B']/total*100:.1f}%)")
    print(f"  Tie:    {stats['tie']} ({stats['tie']/total*100:.1f}%)")
    if stats["error"]:
        print(f"  Error:  {stats['error']}")
    win_rate_a = (stats["A"] + 0.5 * stats["tie"]) / total
    print(f"  Win rate (A, adjusted): {win_rate_a:.4f}")
    print(f"  Win rate (B, adjusted): {1 - win_rate_a:.4f}")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump({"model_a": model_a, "model_b": model_b, "stats": stats, "details": results},
                      f, ensure_ascii=False, indent=2)
        print(f"Details saved to: {args.output_file}")


# ======================== main ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MT-Bench direct inference & pairwise comparison")
    subparsers = parser.add_subparsers(dest="command")

    # infer subcommand
    p_infer = subparsers.add_parser("infer", help="Generate model answers (turn-1 only)")
    p_infer.add_argument("--model", type=str, required=True)
    p_infer.add_argument("--model-id", type=str, default=None)
    p_infer.add_argument("--bench-name", type=str, default="mt_bench")
    p_infer.add_argument("--question-file", type=str, default=None)
    p_infer.add_argument("--question-begin", type=int, default=None)
    p_infer.add_argument("--question-end", type=int, default=None)
    p_infer.add_argument("--output-dir", type=str, default=None, help="Output directory for answer files")
    p_infer.add_argument("--categories", type=str, default="humanities,roleplay,writing",
                         help="Comma-separated categories to infer (default: humanities,roleplay,writing)")
    p_infer.add_argument("--max-tokens", type=int, default=1024)
    p_infer.add_argument("--parallel", type=int, default=1)
    p_infer.add_argument("--force-temperature", type=float, default=None)

    # pairwise subcommand
    p_pair = subparsers.add_parser("pairwise", help="Pairwise comparison of two answer files")
    p_pair.add_argument("--answer-a", type=str, required=True, help="Answer jsonl file for model A")
    p_pair.add_argument("--answer-b", type=str, required=True, help="Answer jsonl file for model B")
    p_pair.add_argument("--judge-model", type=str, required=True, help="Judge LLM model name")
    p_pair.add_argument("--bench-name", type=str, default="mt_bench")
    p_pair.add_argument("--question-file", type=str, default=None)
    p_pair.add_argument("--parallel", type=int, default=1)
    p_pair.add_argument("--output-file", type=str, default=None, help="Save detailed results to file")

    args = parser.parse_args()

    if args.command == "infer":
        cmd_infer(args)
    elif args.command == "pairwise":
        cmd_pairwise(args)
    else:
        parser.print_help()
