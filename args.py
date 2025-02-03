import argparse

# TODO: fix this up
args = {}

def init_arguments():
    global args
    parser = argparse.ArgumentParser(description="Testsuit for promp engineering")

    # Add arguments here
    parser.add_argument("--uni", action='store_true', help="Use uni api")
    # parser.add_argument("--hf", action='store_true', help="Use hf api")
    parser.add_argument("--code", action='store_true', help="Run code bench")
    # parser.add_argument("--qa", action='store_true', help="Run qa bench")
    parser.add_argument("--model", type=str, help="which model to use")
    parser.add_argument("--dir", type=str, help="output directory to use")
    parser.add_argument("--baseline", action='store_true', help="Create baseline")

    # Parse arguments and store them globally
    args.update(vars(parser.parse_args()))
