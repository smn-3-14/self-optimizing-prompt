import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)

OPENAI_MODEL= os.environ.get("OPENAI_MODEL")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
