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

# Diccionario de mapeo de etiquetas para armonizarlas
LABELS_MAP = {
            'action': 'action_adventure',
            'adventure': 'action_adventure',
            'war': 'action_adventure',
            'western': 'action_adventure',
            
            'thriller': 'suspense_crime',
            'crime': 'suspense_crime',
            'mystery': 'suspense_crime',
            'film-noir': 'suspense_crime',
            
            'drama': 'drama_romance',
            'romance': 'drama_romance',
            'biography': 'drama_romance',
            'history': 'drama_romance',
            'short': 'drama_romance',
            'adult': 'drama_romance',
            
            'sci-fi': 'scifi_horror_fantasy',
            'scifi': 'scifi_horror_fantasy',
            'fantasy': 'scifi_horror_fantasy',
            'horror': 'scifi_horror_fantasy',
            
            'comedy': 'comedy_family',
            'family': 'comedy_family',
            'animation': 'comedy_family',
            'musical': 'comedy_family',
            'music': 'comedy_family',
            
            'documentary': 'documentary_factual',
            'news': 'documentary_factual',
            'reality-tv': 'documentary_factual',
            'talk-show': 'documentary_factual',
            'game-show': 'documentary_factual',
            'sport': 'documentary_factual',
            'sports': 'documentary_factual'
        }