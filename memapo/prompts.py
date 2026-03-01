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

SUMMARIZE_REFLECTION_SYS_PROMPT = """You are an expert error-pattern analyst.
Your job: given a question, its correct answer, and multiple failed attempts with reflections, synthesize ONE concise, generalizable error-pattern description that can prevent similar mistakes in the future.
"""

SUMMARIZE_REFLECTION_USER_PROMPT = """A model attempted the following question multiple times and FAILED every time. Analyze all attempts holistically and extract the root-cause error pattern.

<QUESTION>
{question}
</QUESTION>

<CORRECT_ANSWER>
{ground_truth}
</CORRECT_ANSWER>

<FAILED_ATTEMPTS>
{failed_attempts}
</FAILED_ATTEMPTS>

Instructions:
1. Compare all failed attempts — identify the COMMON root cause, not just surface-level symptoms.
2. Consider whether the errors stem from misunderstanding the question, flawed reasoning, calculation mistakes, or missing domain knowledge and so on.
3. Abstract the lesson into a generalizable rule that applies beyond this specific question.

You MUST respond with a JSON object in exactly this format and nothing else:
{{
    "root_cause": "brief description of the common root cause across all attempts",
    "reflection": "one-sentence generalizable rule to prevent this category of error in the future"
}}
"""

def _render_failed_attempts(reflections: list) -> str:
    blocks = [
        "\n".join([
            f"[Attempt {r['attempt']}]",
            f"wrong_answer: {r['wrong_pred']}",
            f"reflection: {r['reflection']}",
        ])
        for r in reflections
    ]
    return "\n\n".join(blocks)

def build_summarize_reflection_sys_prompt() -> str:
    return SUMMARIZE_REFLECTION_SYS_PROMPT

def build_summarize_reflection_user_prompt(question: str, ground_truth: str, reflections: list) -> str:
    return SUMMARIZE_REFLECTION_USER_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        failed_attempts=_render_failed_attempts(reflections),
    ).strip()

UPDATE_ERROR_PATTERN_SYS_PROMPT = """You are an expert error-pattern analyst.
Your job: refine an existing error-pattern description by incorporating new evidence from bad cases.
The updated pattern must be concise, generalizable, and actionable — it will be used as a RULE to prevent similar mistakes in the future.
"""

UPDATE_ERROR_PATTERN_USER_PROMPT = """An existing error pattern needs to be updated because a new bad case has been added to its cluster.

<CURRENT_PATTERN>
{current_pattern}
</CURRENT_PATTERN>

<HISTORICAL_BAD_CASES>
The following are existing bad cases already in this error pattern cluster.
{historical_bad_cases}
</HISTORICAL_BAD_CASES>

<NEW_BAD_CASE>
This is the new bad case that triggered the update.
question: {new_question}
correct_answer: {new_ground_truth}
wrong_answer: {new_wrong_pred}
reflection: {new_reflection}
</NEW_BAD_CASE>

Instructions:
1. Read the current pattern and historical bad cases to understand the existing error pattern.
2. Analyze the NEW bad case — determine whether it introduces a genuinely new dimension to the pattern.
3. If the new bad case is very similar to the historical ones and the current pattern already covers it well, you may keep the current pattern UNCHANGED.
4. Otherwise, produce a refined, generalizable one-sentence pattern description that:
   - Covers BOTH the new bad case and the historical ones
   - Is more precise or more general than the current pattern if the new evidence warrants it
   - Is actionable — it should clearly state what to do or avoid

You MUST respond with a JSON object in exactly this format and nothing else:
{{
    "analysis": "brief reasoning about whether the new bad case changes or confirms the pattern",
    "updated": true or false (whether the pattern needs to be updated),
    "pattern": "one-sentence error-pattern description (refined if updated=true, or the original current pattern if updated=false)"
}}
"""

def _render_bad_cases(bad_cases) -> str:
    if not bad_cases:
        return "(none)"
    blocks = [
        "\n".join([
            f"[BadCase {i}]",
            f"question: {bc.question}",
            f"correct_answer: {bc.ground_truth}",
            f"wrong_answer: {bc.wrong_pred}",
            f"reflection: {bc.reflection}",
        ])
        for i, bc in enumerate(bad_cases, 1)
    ]
    return "\n\n".join(blocks)

def build_update_error_pattern_sys_prompt() -> str:
    return UPDATE_ERROR_PATTERN_SYS_PROMPT

def build_update_error_pattern_user_prompt(current_pattern: str, new_bad_case, historical_bad_cases) -> str:
    return UPDATE_ERROR_PATTERN_USER_PROMPT.format(
        current_pattern=current_pattern,
        new_question=new_bad_case.question,
        new_ground_truth=new_bad_case.ground_truth,
        new_wrong_pred=new_bad_case.wrong_pred,
        new_reflection=new_bad_case.reflection,
        historical_bad_cases=_render_bad_cases(historical_bad_cases),
    ).strip()