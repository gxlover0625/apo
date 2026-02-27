from llm import LLMFactory

class Agent:
    def __init__(self, model_name:str, temperature:float=0., role_description:str=None):
        self.model_name = model_name
        self.temperature = temperature
        self.role_description = role_description
        self.llm = LLMFactory.get_llm(model_name, temperature)

    def chat(self, user_prompt, sys_prompt=None):
        return self.llm.generate(user_prompt, sys_prompt)