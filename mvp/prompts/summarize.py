reflection_prompt = """
You are an advanced reasoning agent. You answered a question incorrectly.

Question: {question}

Your Incorrect Answer: {wrong_trajectory}

Previous Reflections:
{past_reflections}

Please carefully examine question and your incorrect answer step by step, and provide a concise reflection to guide the next attempt. Do not answer the question here, just analyze why it was wrong.
Output your final reflection between <reflection> and </reflection>.
"""