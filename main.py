from src.genre_classification.F_Dataset_Downloader import Dataset_Downloader
import pandas as pd

def main():
	d = Dataset_Downloader()
	train_path, test_path = d(overwrite = True)
	df_train = pd.read_csv(train_path)
	df_test = pd.read_csv(test_path)
	print(len(set(df_train['genre'])), set(df_train['genre']))

if __name__ == "__main__":
	main()
