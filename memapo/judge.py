import re
from llm import LLMFactory

one_score_pattern = re.compile(r"\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile(r"\[(\d+\.?\d*)\]")

JUDGE_SYS_PROMPT = "You are a helpful assistant."

JUDGE_USER_PROMPT_NO_REF = """[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

JUDGE_USER_PROMPT_WITH_REF = """[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Reference Answer]
{reference}
[The End of Reference Answer]

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""


class LLMJudge:
    def __init__(self, judge_model: str, threshold: float = 7.0, temperature: float = 0.0):
        self.client = LLMFactory.get_llm(judge_model, temperature)
        self.threshold = threshold
        self._current_question = None

    def set_question(self, question: str):
        self._current_question = question

    def score(self, question: str, prediction: str, reference: str = None) -> float:
        if reference:
            user_prompt = JUDGE_USER_PROMPT_WITH_REF.format(
                question=question, answer=prediction, reference=reference
            )
        else:
            user_prompt = JUDGE_USER_PROMPT_NO_REF.format(
                question=question, answer=prediction
            )
        raw = self.client.generate(user_prompt, JUDGE_SYS_PROMPT)
        return self._parse_score(raw)

    def _parse_score(self, judgment: str) -> float:
        match = re.search(one_score_pattern, judgment)
        if not match:
            match = re.search(one_score_pattern_backup, judgment)
        if match:
            return float(match.group(1))
        return -1.0

    def __call__(self, prediction: str, ground_truth: str = None) -> bool:
        question = self._current_question or ""
        s = self.score(question, prediction, ground_truth)
        return s >= self.threshold
