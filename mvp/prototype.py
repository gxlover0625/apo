from typing import List

class Demonstration:
    def __init__(self, question, trajectory, *args, **kwargs):
        self.question = question
        self.trajectory = trajectory

class Strategy:
    def __init__(self, solution_steps, pitfalls, *args, **kwargs):
        self.solution_steps = solution_steps
        self.pitfalls = pitfalls

class Prototype:
    def __init__(self, prototype_id:str, context:str, demos:List[Demonstration], strategy:Strategy, max_demos=3, *args, **kwargs):
        self.prototype_id = prototype_id
        self.context = context
        self.demos = demos
        self.strategy = strategy
        self.max_demos = max_demos

    def update_demo(self, demo:Demonstration):
        self.demos.append(demo)
        if len(self.demos) > self.max_demos:
            self.demos = self.demos[-self.max_demos:]
        
