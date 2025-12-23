summarize_prompt = """# Task Description
You are given a solved instance consisting of a problem and its correct reasoning trajectory.
Your task is to abstract this instance into a reusable problem-solving prototype, separating what kind of problem this is from how such problems are solved.

question:
```
{question}
```

reasoning trajectory:
```
{reasoning_trajectory}
```

## Output Requirements:
### Context (Problem Scenario Signature)
Describe the specific scenario reflected by the question.
- Focus primarily on observable task details and scenario characteristics derived from the problem itself like input–output form, information organization, constraintion, the goal, etc.
- Characterize what kind of thinking is required (e.g., transformation, alignment, selection under constraints, abstraction from evidence).
- Do not reference specific domains, entities, values, or terminology from the original instance.
- Keep the Context concise and discriminative, not explanatory.

### Solution Steps (Strategic Workflow)
Provide a generalized solution strategy.
- Each step should represent a high-level reasoning operation, not a low-level procedure.
- Replace all instance-specific details with abstract placeholders (e.g., `<Input>`, `<Condition>`, `<Candidate>`, `<Outcome>`).
- The steps should be directly reusable for a new problem that matches the same Context.
- Keep the steps minimal, non-redundant, and sufficient for a single-pass solution attempt.

## Output Format
Provide the `Context` and `Solution Steps` in the following json format:
{{
    "context": "Concise description of the problem scenario",
    "solution_steps": [
        "1. High-level reasoning operation",
        "2. High-level reasoning operation",
        ...
    ]
}}
"""