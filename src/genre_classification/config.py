from dotenv import load_dotenv
from pathlib import Path
import os

ENV_PATH = None
aux = Path(__file__)

while aux.parent != aux:
	env_files = list(aux.parent.glob("*.env"))
	if env_files:
		ENV_PATH = aux.parent / env_files[0].name
		break
	else:
		aux = aux.parent

if ENV_PATH is None:
	raise FileNotFoundError("Cannot find any .env file in any parent directories.")

load_dotenv(ENV_PATH)

MAIN_NAME = os.getenv("MAIN_NAME", "main.py")
SEED = int(os.getenv("SEED", "42"))
KAGGLEHUB_USERNAME = os.getenv("KAGGLEHUB_USERNAME", None)
KAGGLEHUB_KEY = os.getenv("KAGGLEHUB_KEY", None)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)
