import warnings
import random
import re
import logging
import os
import concurrent.futures
from typing import List

from rich import print
import time
# from cohere import Client
from typing import Any

from pb.mutation_operators import mutate
from pb import gsm
from pb.types import EvolutionUnit, Population

logger = logging.getLogger(__name__)

gsm8k_examples = gsm.read_jsonl('pb/data/gsm.jsonl')

def create_population(tp_set: List, mutator_set: List, problem_description: str) -> Population:
    """samples the mutation_prompts and thinking_styles and returns a 'Population' object.

    Args:
        'size' (int): the size of the population to create.
        'problem_description (D)' (str): the problem description we are optimizing for.
    """
    data = {
        'size': len(tp_set)*len(mutator_set),
        'age': 0,
        'problem_description' : problem_description,
        'elites' : [],
        'units': [EvolutionUnit(**{
            'T' : t, 
            'M' : m,
            'P' : '',
            'fitness' : 0,
            'history' : []
            }) for t in tp_set for m in mutator_set]
    }

    return Population(**data)

def init_run(population: Population, opt_model: Any, task_model: Any, num_evals: int, train_set: Any, eval_fn: Any):
    """ The first run of the population that consumes the prompt_description and 
    creates the first prompt_tasks.
    
    Args:
        population (Population): A population created by `create_population`.
    """

    start_time = time.time()

    prompts = []

    for unit in population.units:    
        # template= f"{unit.T} {unit.M} INSTRUCTION: {population.problem_description} INSTRUCTION MUTANT = "
        template = (
            f"{unit.M} {unit.T}\n"
            f"Current INSTRUCTION (need to be mutated, do not solve): {population.problem_description}\n"
            f"Generate a mutated instruction and wrap it in <MUTANT> and </MUTANT> tags."
        )
        prompts.append(template)
    
 
    results = opt_model.batch_generate(prompts)

    end_time = time.time()

    logger.info(f"Prompt initialization done. {end_time - start_time}s")

    assert len(results) == population.size, "size of google response to population is mismatched"
    for i, item in enumerate(results):
        try:
            population.units[i].P = item[0].text.split("<MUTANT>")[1].split("</MUTANT>")[0]
        except:
            population.units[i].P = item[0].text

    _evaluate_fitness(population, task_model, num_evals, train_set, eval_fn)
    
    return population

def run_for_n(n: int, population: Population, opt_model: Any, task_model: Any, num_evals: int, train_set: Any, eval_fn: Any):
    """ Runs the genetic algorithm for n generations.
    """     
    p = population
    for i in range(n):  
        print(f"================== Population {i} ================== ")
        mutate(p, opt_model, train_set)
        print("done mutation")
        _evaluate_fitness(p, task_model, num_evals, train_set, eval_fn)
        print("done evaluation")
        print(p.elites)

    return p

def _evaluate_fitness(population: Population, model: Any, num_evals: int, train_set:Any, eval_fn: Any) -> Population:
    """ Evaluates each prompt P on a batch of Q&A samples, and populates the fitness values.
    """
    # need to query each prompt, and extract the answer. hardcoded 4 examples for now.
    
    logger.info(f"Starting fitness evaluation...")
    start_time = time.time()

    #batch = random.sample(gsm8k_examples, num_evals)
    # instead of random, its better for reproducibility 
    # batch = gsm8k_examples[:num_evals]
    batch = train_set[:num_evals]

    elite_fitness = -1
    current_elite = None
    examples = []
    for unit in population.units:
        # set the fitness to zero from past run.
        unit.fitness = 0
        # todo. model.batch this or multithread
        if os.environ['TASK'] in ["causal_judgement", "logical_deduction_seven_objects", "geometric_shapes"]:
            examples.append([unit.P + ' \n' + example['input'] for example in batch])

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # map() preserves order: results[i] corresponds to examples[i]
        results = list(executor.map(lambda batch: model.batch_generate(batch, temperature=0), examples))


    # https://arxiv.org/pdf/2309.16797.pdf#page=5, P is a task-prompt to condition 
    # the LLM before further input Q.
    for unit_index, fitness_results in enumerate(results):
        for i, x in enumerate(fitness_results):
            # valid = re.search(gsm.gsm_extract_answer(batch[i]['answer']), x[0].text)
            if os.environ['TASK'] in ["causal_judgement", "logical_deduction_seven_objects", "geometric_shapes"]:
                valid = eval_fn(x[0].text, batch[i]['target'])
            if valid:
                # 0.25 = 1 / 4 examples
                population.units[unit_index].fitness += (1 / num_evals)
        
        # Check if this unit is the best after evaluating all samples
        if population.units[unit_index].fitness > elite_fitness:
            # I am copying this bc I don't know how it might get manipulated by future mutations.
            current_elite = population.units[unit_index].model_copy()
            elite_fitness = population.units[unit_index].fitness
    
    # append best unit of generation to the elites list.
    population.elites.append(current_elite)
    end_time = time.time()
    logger.info(f"Done fitness evaluation. {end_time - start_time}s")

    return population