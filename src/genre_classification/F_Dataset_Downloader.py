import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
import os
import shutil

from .config import MAIN_NAME, SEED, LABELS_MAP
from .Credentials import kagglehub_credentials, huggingface_credentials

class Kaggle_Error(Exception):
	pass

class HuggingFace_Error(Exception):
	pass


class Dataset_Downloader:

	def __init__(self):
		self.label_map = LABELS_MAP # Mapa de etiquetas creado en el archivo de configuracion
		path = Path(__file__).resolve()
		self.dataset_folder_path = None
		while True:
			if (path / MAIN_NAME).is_file():
				self.dataset_folder_path = path / "datasets"
				self.dataset_folder_path.mkdir(exist_ok=True)
				break

			if path.parent == path:
				raise FileNotFoundError(f"No se pudo encontrar '{MAIN_NAME}' en ningún directorio superior.")

			path = path.parent

	def __call__(self, overwrite = False) -> tuple[Path, Path]:
		return self.download_datasets(overwrite)

	def download_datasets(self, overwrite = False) -> tuple[Path, Path]:
		train_path = self.dataset_folder_path / "dataset_train.csv"
		test_path = self.dataset_folder_path / "dataset_test.csv"

		if not overwrite and train_path.exists() and test_path.exists():
			return train_path, test_path

		list_train, list_test = [], []

		for method_name in dir(self):
			if method_name.startswith("_download"):
				method = getattr(self, method_name)
				if callable(method):
					try:
						mini_train, mini_test = method()
						list_train.append(mini_train)
						list_test.append(mini_test)

					except Kaggle_Error:
						kagglehub_credentials()
						shutil.rmtree(self.dataset_folder_path)
						os.execv(sys.executable, [sys.executable] + sys.argv)

					except HuggingFace_Error:
						huggingface_credentials()
						shutil.rmtree(self.dataset_folder_path)
						os.execv(sys.executable, [sys.executable] + sys.argv)

		df_train = pd.concat(list_train, ignore_index=True)
		df_test = pd.concat(list_test, ignore_index=True)

		df_train.to_csv(train_path, index=False)
		df_test.to_csv(test_path, index=False)
		return train_path, test_path

	def _download_imdb_movie_genres(self):
		try:
			dataset_dict = load_dataset("adrienheymans/imdb-movie-genres")
		except Exception:
			raise HuggingFace_Error()

		df = dataset_dict['train'].to_pandas()

		df = df.rename(columns={'Title': 'title', 'Description': 'text', 'Genre': 'genre'})
		df['text'] = df['title'].astype(str) + ". " + df['text'].astype(str) # Añadimos el titulo al texto
		
		df = df[['title', 'text', 'genre']]

		df["genre"] = df["genre"].str.lower()
		df["genre"] = df["genre"].replace(self.label_map)
		df.drop(df[df['text'].str.contains("See full ", na=False)].index, inplace=True)

		df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['genre'], random_state=SEED) if SEED else train_test_split(df, test_size=0.2, stratify=df['genre'])
		return df_train, df_test

	def _download_imdb_genres(self):
		try:
			dataset_dict = load_dataset("jquigl/imdb-genres")
		except Exception:
			raise HuggingFace_Error()

		df: pd.DataFrame = pd.concat([
			dataset_dict['train'].to_pandas(),
			dataset_dict["validation"].to_pandas(),
			dataset_dict['test'].to_pandas()
		], ignore_index=True)

		df = df.drop(columns=['expanded-genres', "rating"])
		df = df.rename(columns={"movie title - year": "title", "description": "text"})

		df['text'] = df['title'].astype(str) + ". " + df['text'].astype(str) # Añadimos el titulo al texto
		
		df = df[['title', 'text', 'genre']]

		df["genre"] = df["genre"].str.lower()
		df["genre"] = df["genre"].replace(self.label_map)
		df.drop(df[df['text'].str.contains("See full ", na=False)].index, inplace=True)

		df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['genre'], random_state=SEED) if SEED else train_test_split(df, test_size=0.2, stratify=df['genre'])
		return df_train, df_test
