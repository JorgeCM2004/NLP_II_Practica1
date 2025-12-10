from src.genre_classification.F_Dataset_Downloader import Dataset_Downloader
import pandas as pd

'''
Este script es el punto de entrada para el proyecto. Desde aqui se descarga el dataset 
y se muestra el numero de generos y los generos que hay en el dataset.
'''

def main():
	dataset = Dataset_Downloader()
	train_path, test_path = dataset(overwrite = True)
	df_train = pd.read_csv(train_path)
	df_test = pd.read_csv(test_path)
	print("Primeros ejemplos del dataset de train: \n")
	print(df_train.head(5))
	print("Primeros ejemplos del dataset de test: \n")
	print(df_test.head(5))
	print("Numero de generos distintos: \n", len(set(df_train['genre'])))
	print("Generos: \n", set(df_train['genre']))


if __name__ == "__main__":
	main()
