reflection_prompt = """
You are an advanced reasoning agent. You answered a question incorrectly.

Question: {question}

Your Incorrect Answer: {wrong_trajectory}

Reflections on past failures:
{past_reflections}

Please carefully examine question and your incorrect answer step by step, and provide a concise reflection to guide the next attempt. Do not answer the question here, just analyze why it was wrong.
Output your final reflection between <reflection> and </reflection>.
"""

summarize_prompt = """
You are a summarization agent. Your goal is to abstract the specific problem and its correct solution trajectory into a reusable skill template.

Question: {question}

Correct Solution: {trajectory}

Reflections on past failures: {past_reflections}

Please perform the following analysis:
1. Context
Describe the specific scenario reflected by the question. Focus primarily on observable task details and scenario characteristics derived from the problem itself like input–output form, provided-information, constraintion etc.
2. Solution Steps
Abstract the general solution procedure based on the successful reasoning trajectory. Describe the key reasoning or analysis steps needed to solve this type of problem.
Each step should represent a high-level reasoning operation and can be directly reusable for a new problem that matches the same Context. Keep the steps minimal, non-redundant, and sufficient for a single-pass solution attempt.
3. Pitfalls
Summarize specific mistakes from the reflections; otherwise, identify the implicit critical checks or common traps that the correct solution successfully handled.

Output Format:
Using XML tags to encapsulate your response.
<context>
description of the problem scenario.
</context>

<solution_steps>
1. high-level reasoning operation
2. ...
</solution_steps>

<pitfalls>
specific mistakes or common traps.
</pitfalls>
"""