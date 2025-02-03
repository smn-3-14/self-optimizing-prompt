import time
import random
import string
import logging
from pathlib import Path

import shared
import coding
import multiple_choice
from args import init_arguments, args

logger = logging.getLogger(__name__)

def init_dir():
    if args["dir"]:
        dir = Path(args["dir"])
        if not dir.exists() or not dir.is_dir():
            logging.error("dir error")
            exit(-1)
        return dir

    api = "uni" if args["uni"] else "hf"
    task = "code" if args["code"] else "qa"
    output_dir = (
        Path(__file__).parent
        / "results"
        / (
            str(int(time.time()))
            + f"_{api}_{task}_"
            + "".join(random.choices(string.ascii_letters + string.digits, k=8))
        )
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def init_logging(output_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s|%(asctime)s | %(name)s] %(message)s",
        filename=str((output_dir / "logs.txt").resolve()),
    )

    logging.getLogger().addHandler(logging.StreamHandler())


def main():
    init_arguments()
    output_dir = init_dir()
    shared.init(output_dir)
    init_logging(output_dir)

    logger.info(f"Running with args: {args}")

    if args["code"]:
        if args["baseline"]:
            coding.base_line()
        # coding.run()
        # coding.run_final()
    else:
        if args["baseline"]:
            multiple_choice.base_line()
        # multiple_choice.run()
        # multiple_choice.run_final()


if __name__ == "__main__":
    main()
