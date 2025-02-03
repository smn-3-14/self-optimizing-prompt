import re
import random
import logging
from string import ascii_uppercase
import concurrent
from datasets import load_dataset
import tqdm

from args import args
from shared import exec_ga, final_eval, save
from model import ask_model, get_text_from_response


logger = logging.getLogger(__name__)


def init_question_answer(data):
    # bring question and answer into a good format to make init of population a bit easier
    for i in data:
        options = [
            f"{letter}: {word}" for letter, word in zip(ascii_uppercase, i["options"])
        ]
        i["q"] = i["question"] + "\n\n" + "\n".join(options)
        i["a"] = f"({i["answer"]})"


def extract_uppercase_char_in_parentheses(s):
    match = re.search(r"\(([A-Z])\)", s)
    return match.group(1) if match else None


def extract_uppercase_char_with_trailing_colon(s):
    match = re.search(r"([A-Z]):", s)
    return match.group(1) if match else None


def extract_answer(s):
    answer = extract_uppercase_char_in_parentheses(s)
    if answer is None:
        answer = extract_uppercase_char_with_trailing_colon(s)
    return answer


def check_answer(sample, response):
    response_str = response["choices"][0]["message"]["content"]
    response_selected = extract_answer(response_str)
    return response_selected == sample["answer"]


def fitness_eval(individual, val):
    if "fitness" in individual:
        return  # no need to check fitness twice
    logger.info("Collecting responses")
    responses = {}
    correct = 0
    for sample in val:
        response = ask_model(
            sample["q"], system=get_text_from_response(individual), max_new_tokens=200
        )
        response["passed"] = check_answer(sample, response)
        responses[sample["question_id"]] = response
        if check_answer(sample, response):
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
            sample["q"],
            system=get_text_from_response(individual),
            max_new_tokens=200,
        )
        response["passed"] = check_answer(sample, response)
        responses[sample["question_id"]] = response

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


def run_final():
    problems = load_dataset("TIGER-Lab/MMLU-Pro")["test"].to_list()
    problems = random.sample(problems, 200)
    init_question_answer(problems)

    final_eval(problems, fitness_eval_fast)



def base_line():
    api = "uni" if args["uni"] else "hf"
    logging.info(f"Generating multiple choice baseline with {api} model {args["model"]}")

    base = {"choices": [{"message": {"content": "Answer the following question."}}]}
    problems = load_dataset("TIGER-Lab/MMLU-Pro")["test"].to_list()
    problems = random.sample(problems, 200)

    init_question_answer(problems)
    fitness_eval_fast(base, problems)
    save(base, "base_line.json")


def run():
    api = "uni" if args["uni"] else "hf"
    logging.info(f"Running multiple choice challange with {api} model {args["model"]}")

    problems = load_dataset("TIGER-Lab/MMLU-Pro")

    test = problems["test"].to_list()
    split = problems["validation"].train_test_split(test_size=0.5)
    val = split["test"].to_list()
    train = split["train"].to_list()

    init_question_answer(test)
    init_question_answer(val)
    init_question_answer(train)

    logger.info(f"Generated {len(train)} | {len(val)} | {len(test)}")

    exec_ga(train, val, test, fitness_eval)
