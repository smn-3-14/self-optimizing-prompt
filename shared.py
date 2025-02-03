import json
import random
import logging
from itertools import combinations

import concurrent
import tqdm

from args import args
from model import ask_model, get_text_from_response


# Define the problem-specific parameters
POPULATION_SIZE = 15  # Number of individuals in the population
GENERATIONS = 8  # Number of generations to run
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.4
LIMIT_EVALS = 25


logger = logging.getLogger(__name__)
output_dir = None  # This is horrible... TODO: fix


def init(output_directory):
    global output_dir
    output_dir = output_directory


def save(output, name):
    output_file = output_dir / name
    logger.info(f"Writing to {output_file.resolve()}")

    with output_file.open("a") as fh:
        json.dump(output, fh)
        fh.write("\n")


def load(name):
    file = output_dir / name
    with file.open() as f:
        return json.load(f)


def get_max(population):
    return max(population, key=lambda i: i["fitness"])


def get_min(population):
    return min(population, key=lambda i: i["fitness"])


def get_avg(population):
    return sum(map(lambda i: i["fitness"], population)) / len(population)


def initialize_population(training_data):
    population: list[dict[str, object]] = []
    for _ in tqdm.tqdm(range(POPULATION_SIZE), desc="Init First Population"):
        examples = random.sample(training_data, 10)
        prompt = ""

        for example in examples:
            prompt += f"task:{example["q"]}\nanswer:\n{example["a"]}\n\n"

        prompt += "\n...\n The sentence that gives instruction for solving these kind of tasks is"

        response = ask_model(
            prompt, max_new_tokens=64, do_sample=True, top_p=0.95, temp=0.9
        )
        population.append(response)

    return population


def selection(population, tournament_size=2):
    tournament = random.sample(population, tournament_size)
    tournament = [i() for i in tournament]
    return get_max(tournament)


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

    return ask_model(
        child_source, max_new_tokens=64, do_sample=True, top_p=0.95, temp=1.5
    )


def mutate(individual):
    template = """
Paraphrase the following sentences

### before:
The scientist conducted an experiment to test the effects of temperature on plant growth.
### after:
In an effort to study how temperature influences plant growth, the scientist carefully designed and carried out an experiment.

### before:
The city built a new park to provide more green spaces for residents.
### after:
To increase the availability of green spaces for its residents, the city decided to construct a brand-new park equipped with walking trails and playgrounds.
...

### before:
{individual}
### after:
"""

    return ask_model(
        template.format(individual=individual),
        max_new_tokens=64,
        do_sample=True,
        top_p=0.95,
        temp=1.5,
    )


def made_progress(old_best, new_best):
    if old_best["fitness"] < 0.2:
        return True
    return old_best["fitness"] < new_best["fitness"]


def final_eval(data, fitness_eval):
    api = "uni" if args["uni"] else "hf"
    logging.info(f"Generating final eval with {api} model {args["model"]}")

    for i in tqdm.tqdm(range(GENERATIONS + 1), desc="Eval Generation"):
        population = load(f"population_{i}.json")
        best = get_max(population)
        del best["fitness"]
        fitness_eval(best, data)
        save(best, f"eval_{i}.json")

    logging.info("Done!")


def exec_ga(train, val, test, fitness_eval):
    logger.info("Executing genetic algorithm")
    logger.info("Using")
    logger.info(f"Population size:\t{POPULATION_SIZE}")
    logger.info(f"Generations:\t{GENERATIONS}")
    logger.info(f"Crossover rate:\t{CROSSOVER_RATE}")
    logger.info(f"Mutation rate:\t{MUTATION_RATE}")

    save(train, "train.json")
    save(val, "val.json")
    save(test, "test.json")

    population = initialize_population(train)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(fitness_eval, i, random.sample(val, LIMIT_EVALS))
            for i in population
        ]
        futures = [
            future.result()
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures),
                desc="Eval First Generation",
                total=len(population),
            )
        ]

    save(population, "population_0.json")

    for i in tqdm.tqdm(range(GENERATIONS), desc="Generation"):
        assert len(population) == POPULATION_SIZE

        logger.info(
            f"Stats: {get_max(population)["fitness"]} | {get_avg(population)} | {get_min(population)["fitness"]}"
        )

        current_best = get_max(population)
        next_population = []
        crossover_targets = [i for i in population if random.random() < CROSSOVER_RATE]
        crossover_targets = [i for i in combinations(crossover_targets, 2)]
        mutation_targets = [i for i in population if random.random() < MUTATION_RATE]

        # delay evaluation until we need it
        options = []

        def curry_me(asdf):
            assert type(asdf) != tuple

            def re():
                assert type(asdf) != tuple
                return asdf

            return re

        for individual in population:
            options.append(curry_me(individual))

        def curry_me2(asdf):
            assert type(asdf) != tuple

            def re():
                resp = fitness_eval(
                    mutate(get_text_from_response(asdf)),
                    random.sample(val, LIMIT_EVALS),
                )
                assert type(resp) != tuple
                return resp

            return re

        for individual in mutation_targets:
            options.append(curry_me2(individual))

        def curry_me3(asdf):
            def re():
                resp = fitness_eval(
                    crossover(
                        get_text_from_response(asdf[0]), get_text_from_response(asdf[1])
                    ),
                    random.sample(val, LIMIT_EVALS),
                )
                assert type(resp) != tuple
                return resp

            return re

        for individual in crossover_targets:
            options.append(curry_me3(individual))

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(selection, options) for _ in range(POPULATION_SIZE - 1)
            ]
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures),
                desc="Selecting",
                total=POPULATION_SIZE - 1,
            ):
                next_population.append(future.result())
        next_population.append(current_best)

        save(next_population, f"population_{i + 1}.json")

        # if not made_progress(current_best, get_max(population)):
        #     logger.info("Early stop reached")
        #     break

        population = next_population

    logger.info("Done!")
