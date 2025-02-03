import requests
import traceback
import logging

from openai import OpenAI
from tenacity import retry, wait_exponential, after_log

from args import args
import settings
import logging


logger = logging.getLogger(__name__)


def get_text_from_response(individual):
    return individual["choices"][0]["message"]["content"]


def ask_openai(
    msg: str, max_new_tokens=64, do_sample=False, top_p=None, temp=None
): ...  # currently replaced by uni


def ask_hf(
    msg: str,
    system: str = "",
    max_new_tokens=64,
    do_sample=False,
    top_p=None,
    temp=None,
):
    client = OpenAI(
        base_url="https://api-inference.huggingface.co/v1/", api_key=settings.HF_API_KEY
    )

    messages = []

    if system:
        messages.append({"role": "system", "content": system[:440]})

    messages.append({"role": "user", "content": msg})

    completion = client.chat.completions.create(
        model=args["model"],
        messages=messages,
        max_tokens=max_new_tokens,
        top_p=top_p,
        temperature=temp,
        extra_headers={"x-wait-for-model": "true"},
    )

    response = completion.to_dict()
    logger.info(f"Response:{response}")
    try:
        assert type(get_text_from_response(response)) == str
    except Exception as e:
        logger.info(msg)
        logger.info(response)
        raise e from None
    return response


def ask_uni(
    msg: str,
    system: str = "",
    max_new_tokens=64,
    do_sample=False,
    top_p=None,
    temp=None,
):
    messages = []

    if system:
        messages.append({"role": "system", "content": system[:440]})

    messages.append({"role": "user", "content": msg})

    body = {
        "model": args["model"],
        "messages": messages,
    }

    if not top_p is None:
        body["top_p"] = top_p

    if not max_new_tokens is None:
        body["max_tokens"] = max_new_tokens

    if not temp is None:
        body["temperature"] = temp

    response = requests.post(
        url="http://172.26.92.115/chat_completion",
        headers={"Authorization": settings.UNI_API_KEY},
        json=body,
        timeout=30,
    )
    response.raise_for_status()
    response = response.json()
    try:
        assert type(get_text_from_response(response)) == str
    except Exception as e:
        logger.info(msg)
        logger.info(response)
        raise e from None
    return response


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    after=after_log(logger, logging.INFO),
)
def ask_model(*_args, **_kwargs):
    try:
        if args["uni"]:
            return ask_uni(*_args, **_kwargs)

        return ask_hf(*_args, **_kwargs)

    except Exception as e:
        logger.error(traceback.format_exc())
        raise e from None
