import json
import sys

sys.path.append("./human-eval")
import re
import random
from itertools import combinations
from pathlib import Path

from human_eval import data, execution
from openai import OpenAI
import tqdm

import settings

# Define the problem-specific parameters
GENE_LENGTH = 10  # Length of each individual (chromosome)
POPULATION_SIZE = 20  # Number of individuals in the population
GENERATIONS = 20  # Number of generations to run
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.4


OPENAI_CLIENT = OpenAI(base_url=settings.OPENAI_BASE_URL)

def save(output, name):
    output_file = Path(__file__).parent / name

    with output_file.open("a") as name:
        json.dump(output, name)
        name.write("\n")

def _ask_model(msg: str):
    # n has to stay 1 for this to work
    return (
        OPENAI_CLIENT.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": msg,
                }
            ],
            model=settings.OPENAI_MODEL,
        )
        .choices[0]
        .message.content
    )


def initialize_population(population: list[str], training_data):
    print("Generating population 1")

    for _ in tqdm.tqdm(range(POPULATION_SIZE)):
        examples = random.sample(training_data, 10)
        prompt = ""

        for example in examples:
            prompt += f"task:{example["prompt"]}\nanswer:\n{example["canonical_solution"]}\n\n"

        prompt += (
            "\n...\n The sentence that give instructions for solving these tasks is"
        )

        response = _ask_model(prompt)
        population.append(response)


def extract_first_python_code_block(text):
    # Regular expression to match a Python code block
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(
            1
        ).strip()  # Extract the code block content and strip any extra whitespace
    return text


def fitness_function(individual, test_data):
    print("Collecting responses")
    responses = {}
    for sample in tqdm.tqdm(test_data):
        prompt = f"{individual}\n{sample["prompt"]}"
        responses[sample["task_id"]] = extract_first_python_code_block(
            _ask_model(prompt)
        )

    print("Evaluating fitness")
    success = 0
    for sample in tqdm.tqdm(test_data):
        result = execution.check_correctness(sample, responses[sample["task_id"]], 20.0)
        if result["passed"]:
            success += 1

    return success / len(test_data)


def selection(population, tournament_size=2):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x[1])


def crossover(parent1, parent2):
    template = """
Summarize the following sentences

### before:
Explain the step-by-step solution to this problem, ensuring clarity and correctness. Provide the most efficient method to solve this problem, including the final answer.
### after:
Provide a clear, step-by-step solution to this problem, including the most efficient method and the final answer.

### before:
Expand on this poem while maintaining its tone and style. Refine this poem to enhance its rhythm and imagery.
### after:
Expand and refine this poem, enhancing its tone, style, and imagery.
...

### before:
{parents}
### after:
"""
    candidate1 = template.format(parents=parent1 + parent2)
    candidate2 = template.format(parents=parent2 + parent1)

    return _ask_model(candidate1), _ask_model(candidate2)


def mutate(individual):
    template = """
Paraphrase the following sentences

### before:
She went to the store.
### after:
She headed to the store.

### before:
She smiled warmly at him.
### after:
She gave him a warm smile.
...

### before:
{individual}
### after:
"""

    return _ask_model(template.format(individual=individual))


def split_data(data, train_ratio=0.2, val_ratio=0.2, test_ratio=0.6, seed=None):
    print("Generating data splits")

    keys = list(data.keys())

    if seed is not None:
        random.seed(seed)

    random.shuffle(keys)

    total_len = len(keys)
    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)

    train_keys = keys[:train_end]
    val_keys = keys[train_end:val_end]
    test_keys = keys[val_end:]

    train_set = list(map(lambda k: data[k], train_keys))
    val_set = list(map(lambda k: data[k], val_keys))
    test_set = list(map(lambda k: data[k], test_keys))

    print(f"Generated {len(train_set)} | {len(val_set)} | {len(test_set)}")

    return train_set, val_set, test_set


def run():
    problems = data.read_problems()
    train_set, val_set, test_set = split_data(problems)

    save(train_set, "train.json")
    save(val_set, "val.json")
    save(test_set, "test.json")

    population = []

    initialize_population(population, train_set)

    population = [(i, fitness_function(i, test_set)) for i in population]

    assert len(population) == POPULATION_SIZE

    save(population, "populations.json")

    print("Starting GA...")
    g = 0
    while g < tqdm.tqdm(GENERATIONS):
        population_next = []
        crossover_targets = [i for i in population if random.random() < CROSSOVER_RATE]
        mutation_targets = [i for i in population if random.random() < MUTATION_RATE]

        print("Crossover...")
        results = [y
                   for x in tqdm.tqdm(combinations(crossover_targets))
                   for y in crossover(x[0], x[1])]
        print("Mutating...")
        results += [mutate(i) for i in tqdm.tqdm(mutation_targets)]
        results = [(i, fitness_function(i, test_set)) for i in results]

        for _ in range(POPULATION_SIZE - 1):
            population_next.append(selection(results))
        population_next.append(max(population, key=lambda x: x[1]))

        save(population, "populations.json")
        population = population_next
        g += 1
