from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd

from .config import MAIN_NAME, SEED

class Dataset_Downloader:
	def __init__(self):
		path = Path(__file__)
		self.dataset_folder_path = None
		while not self.dataset_folder_path:
			if path.resolve() == "/":
				raise FileNotFoundError(f"Cannot find {MAIN_NAME} in any parent directories.")
			elif path.is_dir() and MAIN_NAME in [f.name for f in path.iterdir() if f.is_file()]:
				self.dataset_folder_path = path / "datasets"
				self.dataset_folder_path.mkdir(exist_ok=True)
			else:
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
					except Exception("kaggle"):
						pass
					except Exception("huggingface"):
						pass
					list_train.append(mini_train)
					list_test.append(mini_test)
		df_train = pd.concat(list_train, ignore_index=True)
		df_test = pd.concat(list_test, ignore_index=True)

		df_train.to_csv(train_path, index=False)
		df_test.to_csv(test_path, index=False)
		return train_path, test_path

	def _download_imdb_movie_genres(self):
		try:
			dataset_dict = load_dataset("adrienheymans/imdb-movie-genres")
		except:
			raise Exception("huggingface")
		df: pd.DataFrame = pd.concat([dataset_dict['train'].to_pandas(), dataset_dict['test'].to_pandas()], ignore_index=True)
		df = df.drop(columns=['label'])
		df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['genre'], random_state=SEED) if SEED else train_test_split(df, test_size=0.2, stratify=df['genre'])
		return df_train, df_test

	def _download_imdb_genres(self):
		try:
			dataset_dict = load_dataset("jquigl/imdb-genres")
		except:
			raise Exception("huggingface")
		df: pd.DataFrame = pd.concat([dataset_dict['train'].to_pandas(), dataset_dict["validation"].to_pandas(),dataset_dict['test'].to_pandas()], ignore_index=True)
		df = df.drop(columns=['expanded-genres', "rating"])
		df = df.rename(columns={"movie title - year": "title","description": "text"})
		df = df[['title', 'text', 'genre']]
		df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['genre'], random_state=SEED) if SEED else train_test_split(df, test_size=0.2, stratify=df['genre'])
		return df_train, df_test
