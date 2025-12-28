import re
from prompts.summarize import reflection_prompt

class Agent:
    def __init__(self, model:str, temperature:float=0., call_fn=None):
        self.model = model
        self.temperature = temperature
        self.call_fn = call_fn
    
    def direct_answer(self, prompt):
        return self.call_fn(prompt, model=self.model, temperature=self.temperature)

    def __call__(self, prompt):
        return self.direct_answer(prompt)

class AnswerAgent(Agent):    
    def _generate_reflection(self, question, wrong_trajectory, memory):
        past_reflections = "\n".join([f"- {m}" for m in memory]) if memory else "None"
        meta_prompt = reflection_prompt.format(
            question=question,
            wrong_trajectory=wrong_trajectory,
            past_reflections=past_reflections
        )
        reflection_result = self.direct_answer(meta_prompt)
        pattern = r"<reflection>(.*?)</reflection>"
        match = re.search(pattern, reflection_result, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return reflection_result
    
    def reflexion_answer(self, instruction, output_format, question, gt, eval_fn, *args, **kwargs):
        memory = []
        t = 0
        max_trials = 3
        is_success = False

        while t < max_trials:
            past_reflections = "\n".join([f"- {m}" for m in memory]) if memory else "None"
            if t == 0:
                prompt = f"{instruction}\n{question}\n{output_format}"
            else:
                prompt = (
                    f"Instruction: {instruction}\n"
                    f"Question: {question}\n\n"
                    f"Previous reflections:\n{past_reflections}\n\n"
                    f"Previous Attempt: {last_trajectory}\n\n"
                    f"Based on the instruction and previous reflections above, carefully answer the question again.\n"
                    f"{output_format}"
                )

            # Generate trajectory
            trajectory = self.direct_answer(prompt)
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
    
class SummaryAgent(Agent):
    def summary(self, *args, **kwargs):
        pass