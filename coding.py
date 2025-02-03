import re
import sys
import logging
import random

import concurrent

import tqdm

sys.path.append("./human-eval")
from human_eval import data, execution

from args import args
from model import ask_model, get_text_from_response
from shared import save, exec_ga, final_eval


logger = logging.getLogger(__name__)


def init_question_answer(data):
    # bring question and answer into a good format to make init of population a bit easier
    for sample in data:
        sample["q"] = sample["prompt"]
        sample["a"] = sample["canonical_solution"]


def extract_first_code_block(text):
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    pattern = re.compile(r"```\n(.*?)```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return text


def fitness_eval(individual, val):
    if "fitness" in individual:
        return  # no need to check fitness twice
    logger.info("Collecting responses")
    responses = {}
    correct = 0
    for sample in val:
        response = ask_model(
            sample["prompt"],
            system=get_text_from_response(individual),
            max_new_tokens=200,
        )
        code = extract_first_code_block(get_text_from_response(response))
        result = execution.check_correctness(sample, code, 20.0)
        response["passed"] = result["passed"]
        responses[sample["task_id"]] = response
        if result["passed"]:
            correct += 1
    individual["responses"] = responses
    individual["fitness"] = correct / len(val)
    return individual


def fitness_eval_fast(individual, val):
    if "fitness" in individual:
        return  # no need to check fitness twice
    logger.info("Collecting responses")
    responses = {}
    correct = 0

    def _ask_model_and_eval(individual, sample):
        response = ask_model(
            sample["prompt"],
            system=get_text_from_response(individual),
            max_new_tokens=200,
        )
        code = extract_first_code_block(get_text_from_response(response))
        result = execution.check_correctness(sample, code, 20.0)
        response["passed"] = result["passed"]
        responses[sample["task_id"]] = response

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []

        for sample in val:
            futures.append(executor.submit(_ask_model_and_eval, individual, sample))

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), desc="Evaluating", total=len(val)
        ):
            ...

    for response in responses.values():
        if response["passed"]:
            correct += 1

    individual["responses"] = responses
    individual["fitness"] = correct / len(val)
    return individual


def split_data(data, train_ratio=0.2, val_ratio=0.2):
    logger.info("Generating data splits")
    keys = list(data.keys())
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


def base_line():
    api = "uni" if args["uni"] else "hf"
    logging.info(f"Generating coding baseline with {api} model {args["model"]}")

    base = {
        "choices": [{"message": {"content": "Write code for the problem statement."}}]
    }
    problems = list(data.read_problems().values())

    init_question_answer(problems)
    fitness_eval_fast(base, problems)
    save(base, "base_line.json")


def run_final():
    problems = list(data.read_problems().values())
    init_question_answer(problems)

    final_eval(problems, fitness_eval_fast)


def run():
    api = "uni" if args["uni"] else "hf"
    logging.info(f"Running coding challange with {api} model {args["model"]}")

    problems = data.read_problems()
    train, val, test = split_data(problems)

    init_question_answer(train)
    init_question_answer(val)
    init_question_answer(test)

    exec_ga(train, val, test, fitness_eval)
