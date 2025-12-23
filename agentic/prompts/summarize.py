summarize_prompt = """# Task Description
You are given a solved instance consisting of a problem and its correct reasoning trajectory.

question:
```
{question}
```

reasoning trajectory:
```
{reasoning_trajectory}
```

Your task is to abstract this instance into a reusable problem-solving prototype that can be applied to other problems sharing the same underlying structure but different surface content.
The prototype should capture how the problem is cognitively structured and how it is strategically solved, rather than what the problem is about.

## Output Requirements:
### Context (Structural Archetype)
Describe the type of cognitive task involved.
- Focus on the information processing pattern, structural constraints, and reasoning demands.
- Characterize what kind of thinking is required (e.g., transformation, alignment, selection under constraints, abstraction from evidence).
- Do not reference specific domains, entities, values, or terminology from the original instance.

### Solution Steps (Strategic Workflow)
Provide a generalized solution strategy.
- Each step should represent a high-level reasoning operation, not a low-level procedure.
- Replace all instance-specific details with abstract placeholders (e.g., `<Input>`, `<Condition>`, `<Candidate>`, `<Outcome>`).
- The steps should be directly reusable for a new problem that matches the same Context.
- Keep the steps minimal, non-redundant, and sufficient for a single-pass solution attempt.

## Output Format
Provide the `Context` and `Solution Steps` in the following json format:
{{
    "context": "Concise description of the problem's structural type",
    "solution_steps": [
        "1. High-level reasoning operation",
        "2. High-level reasoning operation",
        ...
    ]
}}
"""