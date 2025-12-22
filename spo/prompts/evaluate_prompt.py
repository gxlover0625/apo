EVALUATE_PROMPT = """
Based on the original requirements, evaluate the two responses, A and B, and determine which one better meets the requirements. If a reference answer is provided, strictly follow the format/content of the reference answer.

# Requirement
{requirement}

# A
{answers}

# B
{new_answers}

# Golden answer
{ground_truth}

Provide your analysis and the choice you believe is better, using XML tags to encapsulate your response.

<analyse>Some analysis</analyse>
<choose>A/B (the better answer in your opinion)</choose>
"""
