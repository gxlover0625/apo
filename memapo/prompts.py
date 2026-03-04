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
You have attempted this question before and FAILED. The following are mandatory checks distilled from your previous mistakes.
Treat each item as a MUST-FOLLOW checklist — verify every point before producing your final answer.
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
    """reflections: list of dict, each with keys 'attempt', 'analysis', 'reflection'"""
    if not reflections:
        return ""
    blocks = [
        "\n".join([
            f"[Check {i}]",
            f"error_summary: {r.get('analysis', '')}",
            f"action: {r['reflection']}",
        ])
        for i, r in enumerate(reflections, 1)
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
Your job: diagnose the root cause of a wrong answer and produce a concrete, executable check that would have caught the mistake.
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
2. Trace through your reasoning step by step — pinpoint the EXACT step where the error occurs (e.g., misread a condition, wrong formula, arithmetic slip, overlooked edge case).
3. Consider alternative interpretations or approaches you may have overlooked.
4. If prior reflections exist, do NOT repeat the same diagnosis — dig deeper or try a completely different angle.

You MUST respond with a JSON object in exactly this format and nothing else:
{{
    "analysis": "The wrong answer is <your_wrong_answer>, then diagnose the root cause — identify WHICH step went wrong and WHY",
    "reflection": "a concrete, executable check-action that can be directly applied before answering (e.g., 'Verify that the denominator is non-zero before dividing', 'Re-read the question to confirm whether it asks for the minimum or maximum')"
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

CREATE_TEMPLATE_SYS_PROMPT = """You are an expert at abstracting reusable problem-solving templates.
Your job: given a question and its correct answer, extract a generalizable template that can guide solving similar problems in the future.
"""

CREATE_TEMPLATE_USER_PROMPT = """A model answered the following question correctly. Abstract this success into a reusable template.

<QUESTION>
{question}
</QUESTION>

<CORRECT_ANSWER>
{correct_pred}
</CORRECT_ANSWER>
{reflections_block}
Instructions:
1. Analyze the question type, scenario characteristics, and what makes this kind of problem recognizable.
2. Abstract the general solution procedure based on the successful reasoning trajectory. Describe the key reasoning or analysis steps needed to solve this type of problem. Each step should represent a high-level reasoning operation and can be directly reusable for a new problem that matches the same scenario. Keep the steps minimal, non-redundant, and sufficient for a single-pass solution attempt.
3. If prior failed attempts and reflections are provided, incorporate the lessons learned as pitfalls to avoid in the strategy.
4. Produce:
   - "when_to_use": a concise description of WHEN this template should be applied (what kind of question/scenario triggers it)
   - "strategy": the abstracted step-by-step reasoning procedure for this type of problem, including pitfalls to avoid if reflections are available

You MUST respond with a JSON object in exactly this format and nothing else:
{{
    "when_to_use": "one-sentence description of the applicable scenario",
    "strategy": "concise general reasoning strategy for this type of problem"
}}
"""

def build_create_template_sys_prompt() -> str:
    return CREATE_TEMPLATE_SYS_PROMPT

CREATE_TEMPLATE_REFLECTIONS_BLOCK = """
<PRIOR_FAILED_ATTEMPTS>
The model failed on earlier attempts before getting it right. Use these reflections to identify pitfalls.
{reflections}
</PRIOR_FAILED_ATTEMPTS>
"""

def _render_create_template_reflections(reflections: list) -> str:
    if not reflections:
        return ""
    blocks = [
        f"[Attempt {r['attempt']}] wrong_answer: {r['wrong_pred']} | reflection: {r['reflection']}"
        for r in reflections
    ]
    return CREATE_TEMPLATE_REFLECTIONS_BLOCK.format(reflections="\n".join(blocks))

def build_create_template_user_prompt(question: str, correct_pred: str, reflections: list = None) -> str:
    return CREATE_TEMPLATE_USER_PROMPT.format(
        question=question,
        correct_pred=correct_pred,
        reflections_block=_render_create_template_reflections(reflections or []),
    ).strip()

UPDATE_TEMPLATES_SYS_PROMPT = """You are an expert template manager for a retrieval-augmented problem-solving system.
Your job: given a new successfully solved case and the recalled templates, decide the best action to keep the template library accurate, non-redundant, and maximally useful.
"""

UPDATE_TEMPLATES_USER_PROMPT = """A model just answered a question correctly. Review the recalled templates and decide what actions to take.

<RECALLED_TEMPLATES>
{recalled_templates}
</RECALLED_TEMPLATES>

<NEW_GOOD_CASE>
question: {question}
correct_answer: {correct_pred}
</NEW_GOOD_CASE>
{reflections_block}
You must decide an action for EACH recalled template, and optionally add new templates. Rules:
- Each recalled template must appear EXACTLY ONCE in the actions list.
- Each action targets ONE template.

Available actions:

1. **none**: The recalled template already covers this case well (semantically equivalent). Keep it unchanged. Specify the template_id.
2. **update**: The recalled template is relevant but its when_to_use or strategy can be enriched / made more comprehensive with information from the new case. Specify the template_id and provide updated fields.
   - "when_to_use": a concise description of WHEN this template should be applied (what kind of question/scenario triggers it). Set to null to keep unchanged.
   - "strategy": the abstracted step-by-step reasoning procedure for this type of problem, including pitfalls to avoid if reflections are available. Set to null to keep unchanged.
3. **delete**: The recalled template conflicts with the new case (e.g. wrong strategy, contradictory advice) or is fully superseded. Specify the template_id.
4. **add**: The new case represents a genuinely new problem type not covered by ANY recalled template. Create a new template. (Use sparingly — only when none of the recalled templates can be updated to cover this case.)
   - "when_to_use": a concise description of WHEN this template should be applied (what kind of question/scenario triggers it).
   - "strategy": the abstracted step-by-step reasoning procedure for this type of problem, including pitfalls to avoid if reflections are available.

IMPORTANT:
- Only use template_id values that appear in RECALLED_TEMPLATES above. Do NOT invent template_ids.
- Every recalled template_id must appear exactly once across all actions.

You MUST respond with a JSON object containing an "actions" list:

{{
    "actions": [
        {{"action": "none", "template_id": "..."}},
        {{"action": "update", "template_id": "...", "when_to_use": "... or null", "strategy": "... or null"}},
        {{"action": "delete", "template_id": "..."}},
        {{"action": "add", "when_to_use": "...", "strategy": "..."}}
    ]
}}
"""

def _render_existing_templates(templates):
    if not templates:
        return "(none)", {}
    id_mapping = {}
    blocks = []
    for i, t in enumerate(templates, 1):
        id_mapping[str(i)] = t.idx
        blocks.append("\n".join([
            f"[Template {i}]",
            f"template_id: {i}",
            f"when_to_use: {t.when_to_use}",
            f"strategy: {t.strategy}",
        ]))
    return "\n\n".join(blocks), id_mapping

def build_update_templates_sys_prompt() -> str:
    return UPDATE_TEMPLATES_SYS_PROMPT

def build_update_templates_user_prompt(recalled_templates, question: str, correct_pred: str, reflections: list = None):
    rendered, id_mapping = _render_existing_templates(recalled_templates)
    prompt = UPDATE_TEMPLATES_USER_PROMPT.format(
        recalled_templates=rendered,
        question=question,
        correct_pred=correct_pred,
        reflections_block=_render_create_template_reflections(reflections or []),
    ).strip()
    return prompt, id_mapping

MERGE_TEMPLATES_SYS_PROMPT = """You are an expert template librarian for a retrieval-augmented problem-solving system.
Your job: given a full list of templates, identify groups of templates that are semantically similar or overlapping and can be merged into a single, more general template without losing coverage.
"""

MERGE_TEMPLATES_USER_PROMPT = """The template library has grown too large ({total} templates, limit is {limit}). Identify groups of templates that can be merged to reduce the total count.

<ALL_TEMPLATES>
{all_templates}
</ALL_TEMPLATES>

Instructions:
1. Read ALL templates carefully. Identify groups where the when_to_use scenarios overlap significantly or the strategies are highly similar / complementary.
2. Only merge templates that are truly related — do NOT force-merge unrelated templates just to reduce count.
3. For each merge group, produce a single merged template that covers all the scenarios and combines the best parts of each strategy.
4. Templates NOT included in any merge group will be kept as-is.
5. Try to reduce the total template count to at most {target} through merging.

You MUST respond with a JSON object in exactly this format and nothing else:
{{
    "merge_groups": [
        {{
            "template_ids": ["1", "3"],
            "reason": "brief explanation of why these templates should be merged",
            "merged_when_to_use": "combined scenario description covering all merged templates",
            "merged_strategy": "combined strategy incorporating the best of each template"
        }}
    ]
}}

If no templates can be reasonably merged, return: {{"merge_groups": []}}
"""

def _render_all_templates_for_merge(templates_dict):
    """Render all templates with sequential IDs for the merge prompt."""
    id_mapping = {}
    blocks = []
    for i, (real_id, t) in enumerate(templates_dict.items(), 1):
        id_mapping[str(i)] = real_id
        gc_count = len(t.good_cases) if t.good_cases else 0
        blocks.append("\n".join([
            f"[Template {i}]",
            f"template_id: {i}",
            f"when_to_use: {t.when_to_use}",
            f"strategy: {t.strategy}",
            f"good_cases_count: {gc_count}",
        ]))
    return "\n\n".join(blocks), id_mapping

def build_merge_templates_sys_prompt() -> str:
    return MERGE_TEMPLATES_SYS_PROMPT

def build_merge_templates_user_prompt(templates_dict, max_templates: int):
    rendered, id_mapping = _render_all_templates_for_merge(templates_dict)
    total = len(templates_dict)
    target = max(max_templates - 5, int(max_templates * 0.8))
    prompt = MERGE_TEMPLATES_USER_PROMPT.format(
        total=total,
        limit=max_templates,
        target=target,
        all_templates=rendered,
    ).strip()
    return prompt, id_mapping