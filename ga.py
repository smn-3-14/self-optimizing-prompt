import json
import sys
import logging
import re
import random
import traceback
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from tenacity import retry, wait_exponential, after_log

sys.path.append("./human-eval")
from human_eval import data, execution
from openai import OpenAI
import requests
import tqdm

import settings

# Define the problem-specific parameters
POPULATION_SIZE = 20  # Number of individuals in the population
GENERATIONS = 20  # Number of generations to run
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.4


tmp_dir = TemporaryDirectory(delete=False)
tmp_dir_path = Path(tmp_dir.name)
logger = logging.getLogger("ga")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("[%(levelname)s|%(asctime)s] %(message)s")

# std = logging.StreamHandler(sys.stdout)
# std.setLevel(logging.DEBUG)
# std.setFormatter(formatter)
# logger.addHandler(std)

fh = logging.FileHandler(str((tmp_dir_path / "logs.txt").resolve()))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

OPENAI_CLIENT = OpenAI(base_url=settings.OPENAI_BASE_URL)
# HF_CLIENT = InferenceClient("google/gemma-2-2b-it", token=settings.TOKEN, timeout=15.0)


def save(output, name):
    output_file = Path(tmp_dir_path) / name
    logger.info(f"Writing to {output_file.resolve()}")

    with output_file.open("a") as name:
        json.dump(output, name)
        name.write("\n")

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), after=after_log(logger, logging.ERROR))
def _ask_model(msg: str, max_new_tokens = 64, do_sample=False, top_p=None, temp=None):
    try:
        API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"
        headers = {"Authorization": "Bearer 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", "x-use-cache": "false", "x-wait-for-model": "true"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            return response.json()

        params = {
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
        }

        if not top_p is None:
            params["top_p"] = top_p

        if not temp is None:
            params["temperature"] = temp

        output = query({
            "inputs": msg,
            "parameters": params,
            "return_full_text": False,
        })


        response = output[0]["generated_text"].replace(msg, "")
        # response = response.split("```")[0]

        return response
    except Exception as e:
        logger.error(traceback.format_exc())
        raise e from None


    # return (
    #     HF_CLIENT.text_generation(
    #         msg,
    #         top_p = top_p,
    #         temperature=temp,
    #         do_sample = do_sample,
    #         max_new_tokens = max_new_tokens,
    #     )
    #     .choices[0]
    #     .message.content
    # )


def initialize_population(population: list[str], training_data):
    logger.info("Generating population 1")

    for _ in tqdm.tqdm(range(POPULATION_SIZE)):
        examples = random.sample(training_data, 10)
        prompt = ""

        for example in examples:
            prompt += f"task:{example["prompt"]}\nanswer:\n{example["canonical_solution"]}\n\n"

        prompt += (
            "\n...\n The sentence that gives a instruction for solving these kind of tasks is"
        )

        response = _ask_model(prompt, max_new_tokens=64, do_sample=True, top_p=0.95, temp=0.9)
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
    logger.info("Collecting responses")
    responses = {}
    for sample in tqdm.tqdm(test_data):
        prompt = f"{sample["prompt"]}\n{individual}"
        responses[sample["task_id"]] = extract_first_python_code_block(
            _ask_model(prompt, max_new_tokens=200)
        )

    logger.info("Evaluating fitness")
    success = 0
    for sample in tqdm.tqdm(test_data):
        result = execution.check_correctness(sample, responses[sample["task_id"]], 20.0)
        if result["passed"]:
            success += 1

    return success / len(test_data)


def selection(population, test_data, tournament_size=2):
    tournament = random.sample(population, tournament_size)
    tournament = [i() for i in tournament]
    tournament = [(i, fitness_function(i, test_data)) for i in tournament]
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
    child_source = random.sample([candidate1, candidate2], 1)[0]

    return _ask_model(child_source, max_new_tokens=64, do_sample=True, top_p=0.95, temp=0.9)


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

    return _ask_model(template.format(individual=individual), max_new_tokens=64, do_sample=True, top_p=0.95, temp=0.9)


def split_data(data, train_ratio=0.2, val_ratio=0.2, test_ratio=0.6, seed=None):
    logger.info("Generating data splits")

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

    logger.info(f"Generated {len(train_set)} | {len(val_set)} | {len(test_set)}")

    return train_set, val_set, test_set


def run():
    problems = data.read_problems()
    train_set, val_set, test_set = split_data(problems)

    save(train_set, "train.json")
    save(val_set, "val.json")
    save(test_set, "test.json")

    population = []

    initialize_population(population, train_set)

    logger.info("Evaluating first population")
    population = [(i, fitness_function(i, val_set)) for i in population]

    save(population, "population_0.json")

    for i in range(GENERATIONS):
        assert len(population) == POPULATION_SIZE

        population_next = []
        crossover_targets = [i for i in population if random.random() < CROSSOVER_RATE]
        crossover_targets = [i for i in combinations(crossover_targets, 2)]
        mutation_targets = [i for i in population if random.random() < MUTATION_RATE]

        p = [lambda: i for i in population]
        c = [lambda: crossover(i[0], i[1]) for i in crossover_targets]
        m = [lambda: mutate(i) for i in mutation_targets]

        logger.info(f"Generatin population... {i + 1}")
        for _ in tqdm.tqdm(range(POPULATION_SIZE - 1)):
            population_next.append(selection(p + c + m, val_set))
        population_next.append(max(population, key=lambda x: x[1]))

        logger.info(
            f"Max: {max(population_next, key=lambda x: x[1])[1]} | Min: {min(population_next, key=lambda x: x[1])[1]} | Avrg: {sum(map(lambda x: x[1], population_next)) / len(population_next)}"
        )

        population = population_next
        save(population, f"population_{i + 1}.json")
