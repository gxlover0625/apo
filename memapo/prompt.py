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

def build_generation_user_prompt(question: str, output_format: str, templates) -> str:
    return join_sections(
        render_templates_block(templates),
        QUESTION_BLOCK.format(question=question),
        OUTPUT_FORMAT_BLOCK.format(output_format=output_format),
    )