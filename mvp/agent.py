import re
from prompts.summarize import reflection_prompt, summarize_prompt
from prototype import Prototype

class Agent:
    def __init__(self, model:str, temperature:float=0., call_fn=None):
        self.model = model
        self.temperature = temperature
        self.call_fn = call_fn
    
    def direct_answer(self, user_prompt, sys_prompt="You are a helpful assistant."):
        return self.call_fn(user_prompt=user_prompt, sys_prompt=sys_prompt, model=self.model, temperature=self.temperature)
    
    def extract_tag_content(self, tag_name, text):
        pattern = f"<{tag_name}>\s*(.*?)\s*</{tag_name}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            return text

    def __call__(self, user_prompt):
        return self.direct_answer(user_prompt=user_prompt)

class AnswerAgent(Agent):    
    def _generate_reflection(self, question, wrong_trajectory, memory):
        past_reflections = "\n".join([f"- {m}" for m in memory]) if memory else "None"
        meta_prompt = reflection_prompt.format(
            question=question,
            wrong_trajectory=wrong_trajectory,
            past_reflections=past_reflections
        )
        reflection_result = self.direct_answer(user_prompt=meta_prompt)
        return self.extract_tag_content("reflection", reflection_result)
    
    def answer_without_reflection(self, instruction, output_format, question, gt, eval_fn, *args, **kwargs):
        sys_prompt = f"{instruction} {output_format}"
        user_prompt = f"{question}"
        is_success = False
        prediction = self.direct_answer(user_prompt=user_prompt, sys_prompt=sys_prompt)
        if eval_fn(prediction, gt):
            is_success = True
            return is_success, prediction
        else:
            return is_success, prediction

    def reflexion_answer(self, instruction, output_format, question, gt, eval_fn, *args, **kwargs):
        memory = []
        t = 0
        max_trials = 3
        is_success = False

        while t < max_trials:
            past_reflections = "\n".join([f"- {m}" for m in memory]) if memory else "None"
            if t == 0:
                # prompt = f"{instruction}\n{question}\n{output_format}"
                sys_prompt = f"{instruction} {output_format}"
                user_prompt = f"{question}"
            else:
                sys_prompt = f"{instruction} {output_format}"
                user_prompt = (
                    f"## Question\n{question}\n\n"
                    f"## Past Failure Reflections\n{past_reflections}\n\n"
                    f"## Previous Attempt\n{last_trajectory}\n\n"
                    f"Answer the question again, explicitly avoiding the mistakes described above."
                )

            # Generate trajectory
            trajectory = self.direct_answer(user_prompt=user_prompt, sys_prompt=sys_prompt)
            last_trajectory = trajectory

            # Evaluate trajectory
            if eval_fn(trajectory, gt):
                is_success = True
                return is_success, trajectory, memory
            else:
                # self-reflection
                reflection = self._generate_reflection(
                    question=question,
                    wrong_trajectory=trajectory,
                    memory=memory
                )
                memory.append(reflection)
            t += 1
        return is_success, trajectory, memory
    
    def answer_with_prototype(self, instruction, output_format, question, gt, eval_fn, prototype, *args, **kwargs):
        demos_str = ""
        for idx, demo in enumerate(prototype.demos, 1):
            demos_str += (
                f"### Example {idx}\n"
                f"Question: {demo.question}\n"
                f"Answer: {demo.trajectory}\n"
            )
        sys_prompt = (
            f"## Task\n{instruction}\n\n"
            f"## Success Solution Steps\n{prototype.strategy.solution_steps}\n\n"
            f"## Success Cases:\n{demos_str}\n\n"
            f"## Common Pitfalls:\n{prototype.strategy.pitfalls}\n\n"
        )
        user_prompt = f"{question}\n{output_format}"
        final_response = self.direct_answer(user_prompt=user_prompt, sys_prompt=sys_prompt)
        is_success = False
        if eval_fn(final_response, gt):
            is_success = True
            return is_success, final_response
        else:
            return is_success, final_response
    
class SummaryAgent(Agent):
    def summary(self, question, trajectory, past_reflections, *args, **kwargs):
        summary_prompt = summarize_prompt.format(
            question=question,
            trajectory=trajectory,
            past_reflections=past_reflections
        )
        summary_result = self.direct_answer(user_prompt=summary_prompt)
        context = self.extract_tag_content("context", summary_result)
        solution_steps = self.extract_tag_content("solution_steps", summary_result)
        pitfalls = self.extract_tag_content("pitfalls", summary_result)
        return context, solution_steps, pitfalls