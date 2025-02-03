import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)

UNI_API_KEY= os.environ.get("UNI_API_KEY")
UNI_MODEL = os.environ.get("UNI_MODEL")
HF_API_KEY = os.environ.get("HF_API_KEY")