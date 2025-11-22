import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from typing import Literal, Tuple
from .config import SEED

class Preprocess_Data_2:

    def __init__(self):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Descargando stopwords de NLTK...")
            nltk.download('stopwords')
        
        self.stemmer = SnowballStemmer('english')
        self.stop_words = set(stopwords.words('english'))
        # Pre-compilamos el regex aquí una sola vez para toda la clase
        self.regex_pattern = re.compile(r'[^a-zA-Z\s]')

    def data_representation(self, data, 
                            weight_type: Literal['tf-idf', 'binary', 'frequency'] = 'tf-idf',
                            max_features: int = 5000,  # MEJORA: Parametrizable
                            ngram_range: Tuple[int, int] = (1, 2)): # MEJORA: Parametrizable
        
        weight_type = weight_type.lower()
        
        # Pasamos los parámetros dinámicos
        params = {
            'max_features': max_features, 
            'min_df': 5, 
            'ngram_range': ngram_range # Permite capturar contexto (ej: "ciencia ficcion")
        } 

        if weight_type == 'tf-idf':
            vectorizer = TfidfVectorizer(**params)
        elif weight_type == 'binary':
            vectorizer = CountVectorizer(binary=True, **params)
        elif weight_type == 'frequency':
            vectorizer = CountVectorizer(**params)
        else:
            raise ValueError(f"Weight type '{weight_type}' not supported.")

        # Ajustamos y transformamos
        matrix = vectorizer.fit_transform(data)
        vocabulary = vectorizer.vocabulary_
        
        return matrix, vocabulary, vectorizer

    def preprocessing_data(self, data: list[str]) -> list[str]:
        
        if hasattr(data, 'tolist'):
            data = data.tolist()

        processed_data = []
        
        for text in data:
            # 1. Minúsculas
            text = text.lower()
            
            # 2. MEJORA CRÍTICA: Reemplazar por ' ' (espacio), no por '' (nada)
            # "end.Start" -> "end Start" (antes era "endStart")
            text = self.regex_pattern.sub(' ', text)
            
            # 3. Tokenizar
            tokens = text.split()
            
            # 4. Stemming y Stopwords
            clean_tokens = [
                self.stemmer.stem(token) 
                for token in tokens 
                if token not in self.stop_words and len(token) > 2
            ]
            
            processed_data.append(" ".join(clean_tokens))

        return processed_data

    def split_data(self, data, test_size=0.2):
        train_set, test_set = train_test_split(
            data, test_size=test_size, random_state=SEED, shuffle=True, stratify=data['genre']
        )
        return train_set, test_set