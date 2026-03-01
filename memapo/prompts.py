def join_sections(*parts: str) -> str:
    return "\n\n".join(p.strip() for p in parts if p and p.strip()).strip()

SYS_HEADER = """You are an expert assistant for question answering with retrieval-augmented memory.
Your job: answer the user's question by leveraging retrieved TEMPLATES as guidance and strictly following RULES derived from historical errors.
"""

RULES_BLOCK = """## RULES
The following rules are summarized from historical errors. You MUST follow them strictly:
{rules}
"""

def render_rules_block(error_patterns) -> str:
    if not error_patterns:
        return ""
    rules = "\n".join(f"R{i}. {ep.pattern}" for i, ep in enumerate(error_patterns, 1))
    return RULES_BLOCK.format(rules=rules)

def build_generation_sys_prompt(init_instruction: str, error_patterns) -> str:
    return join_sections(
        SYS_HEADER,
        init_instruction,
        render_rules_block(error_patterns),
    )

TEMPLATES_BLOCK = """<TEMPLATES>
Below are retrieved templates. Use their strategies as guidance. Each template includes one verified good case.
{templates}
</TEMPLATES>
"""

REFLECTIONS_BLOCK = """<REFLECTIONS>
You have attempted this question before and failed. Learn from your mistakes and do NOT repeat them.
{reflections}
</REFLECTIONS>
"""

QUESTION_BLOCK = """<QUESTION>
{question}
</QUESTION>
"""

OUTPUT_FORMAT_BLOCK = """<OUTPUT_FORMAT>
{output_format}
</OUTPUT_FORMAT>
"""

def render_templates_block(templates) -> str:
    if not templates:
        return ""
    blocks = [
        "\n".join([
            f"[TEMPLATE {i}]",
            f"strategy: {t.strategy}",
            f"example_question: {t.good_cases[0].question}",
            f"example_answer: {t.good_cases[0].correct_pred}",
        ])
        for i, t in enumerate(templates, 1)
    ]
    return TEMPLATES_BLOCK.format(templates="\n\n".join(blocks))

def render_reflections_block(reflections: list) -> str:
    """reflections: list of dict, each with keys 'attempt', 'wrong_pred', 'reflection'"""
    if not reflections:
        return ""
    blocks = [
        "\n".join([
            f"[Attempt {r['attempt']}]",
            f"wrong_answer: {r['wrong_pred']}",
            f"reflection: {r['reflection']}",
        ])
        for r in reflections
    ]
    return REFLECTIONS_BLOCK.format(reflections="\n\n".join(blocks))

def build_generation_user_prompt(question: str, output_format: str, templates, reflections: list = None) -> str:
    return join_sections(
        render_templates_block(templates),
        render_reflections_block(reflections or []),
        QUESTION_BLOCK.format(question=question),
        OUTPUT_FORMAT_BLOCK.format(output_format=output_format),
    )

REFLECTION_SYS_PROMPT = """You are a precise self-reflection assistant.
Your job: analyze why the previous answer was wrong and extract a concise, actionable lesson.
"""

REFLECTION_USER_PROMPT = """Your previous answer to the following question is likely WRONG. Re-examine your reasoning and find the mistake.

<QUESTION>
{question}
</QUESTION>

<YOUR_ANSWER>
{wrong_pred}
</YOUR_ANSWER>
{prior_reflections}
Instructions:
1. Re-read the question carefully and check whether your answer actually addresses all constraints and conditions.
2. Trace through your reasoning step by step — identify any logical gaps, unjustified assumptions, or calculation errors.
3. Consider alternative interpretations or approaches you may have overlooked.
4. If prior reflections exist, do NOT repeat the same analysis — dig deeper or try a completely different angle.

You MUST respond with a JSON object in exactly this format and nothing else:
{{
    "analysis": "your detailed step-by-step analysis of what went wrong", 
    "reflection": "one-sentence actionable lesson to avoid this mistake next time"
}}
"""

PRIOR_REFLECTIONS_BLOCK = """
<PRIOR_REFLECTIONS>
The following are reflections from previous failed attempts. Your new reflection MUST provide a different perspective.
{prior_reflections}
</PRIOR_REFLECTIONS>
"""

def _render_prior_reflections(reflections: list) -> str:
    if not reflections:
        return ""
    blocks = [
        f"[Attempt {r['attempt']}] {r['reflection']}"
        for r in reflections
    ]
    return PRIOR_REFLECTIONS_BLOCK.format(prior_reflections="\n".join(blocks))

def build_reflection_sys_prompt() -> str:
    return REFLECTION_SYS_PROMPT
    
def build_reflection_user_prompt(question: str, wrong_pred: str, reflections: list = None) -> str:
    return REFLECTION_USER_PROMPT.format(
        question=question,
        wrong_pred=wrong_pred,
        prior_reflections=_render_prior_reflections(reflections or []),
    ).strip()