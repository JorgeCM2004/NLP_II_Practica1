import re
import os
import spacy
import nltk
import pandas as pd
import numpy as np
from typing import Literal, List, Tuple, Union, Optional, Any
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

try:
    from .config import SEED
except ImportError:
    SEED = 42

# Clase que se encarga de la preprocesación de los datos, por defecto usaremos NLTK porque es el mas rapido y tienen casi los mismos resultados que SpaCy
class Preprocess_Data:
    def __init__(self, library: Literal['NLTK', 'Spacy', 'TRANSFORMER'] = 'NLTK'):
        self.library = library.upper()
        self.nlp = None
        self.stemmer = None
        self.stop_words = None
        
        if self.library == 'SPACY':
            try:
                self.nlp = spacy.load("en_core_web_lg", disable=["parser", "ner", "textcat"])
                print("SpaCy (en_core_web_lg) cargado exitosamente.")
            except OSError:
                raise RuntimeError(
                    "Error: El modelo 'en_core_web_lg' no está instalado.\n"
                    "Ejecuta en tu terminal: python -m spacy download en_core_web_lg"
                )
        
        elif self.library == 'NLTK':
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                print("Descargando stopwords de NLTK...")
                nltk.download('stopwords', quiet=True)
            
            self.stemmer = SnowballStemmer('english')
            self.stop_words = set(stopwords.words('english'))
            self.regex_pattern = re.compile(r'\b[a-zA-Z]{2,}\b')
            print("NLTK configurado exitosamente.")

        elif self.library == 'TRANSFORMER':
            print("Modo TRANSFORMER activado (limpieza mínima).")
        
        else:
            raise ValueError("Library must be 'NLTK', 'Spacy' or 'TRANSFORMER'")

    def preprocessing_data(self, data: Union[List[str], pd.Series, np.ndarray]) -> List[str]:
        if isinstance(data, pd.Series):
            data_list = data.fillna("").astype(str).tolist()
        elif isinstance(data, np.ndarray):
            data_list = data.tolist()
        else:
            data_list = [str(x) if x is not None else "" for x in data]

        if self.library == 'SPACY':
            return self._process_with_spacy(data_list)
        elif self.library == 'TRANSFORMER':
            return self._process_custom_transformer(data_list)
        else:
            return self._process_with_nltk(data_list)

    def _process_with_spacy(self, data: List[str]) -> List[str]:
        n_jobs = -1 if os.name != 'nt' else 1
        
        processed_data = []
        for doc in self.nlp.pipe(data, batch_size=2000, n_process=n_jobs):
            tokens = [
                token.lemma_.lower() 
                for token in doc 
                if not token.is_stop and not token.is_punct and token.is_alpha
            ]
            processed_data.append(" ".join(tokens))
            
        return processed_data

    def _process_custom_transformer(self, data: List[str]) -> List[str]:
        processed_data = []
        for text in data:
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)
            processed_data.append(text)
        return processed_data

    def _process_with_nltk(self, data: List[str]) -> List[str]:
        processed_data = []
        
        for text in data:
            text = text.lower()
            tokens = self.regex_pattern.findall(text)
            
            clean_tokens = [
                self.stemmer.stem(token) 
                for token in tokens 
                if token not in self.stop_words
            ]
            processed_data.append(" ".join(clean_tokens))
            
        return processed_data

    def data_representation(self, data: List[str], weight_type: Literal['tf-idf', 'binary', 'frequency'] = 'tf-idf'):
        # Tras probar varias configuraciones, esta es la que mejor funciona
        vectorizer_args = {
            'stop_words': 'english',
            'ngram_range': (1, 2),
            'min_df': 5,
            'strip_accents': 'unicode',
            'max_features': 25000
        }

        if weight_type == 'tf-idf':
            if 'sublinear_tf' not in vectorizer_args:
                vectorizer_args['sublinear_tf'] = True
            
            vectorizer = TfidfVectorizer(**vectorizer_args)
        elif weight_type == 'binary':
            vectorizer = CountVectorizer(binary=True, **vectorizer_args)
            
        elif weight_type == 'frequency':
            vectorizer = CountVectorizer(binary=False, **vectorizer_args)
            
        else:
            raise ValueError(f"Weight type '{weight_type}' no soportado.")

        print(f"Vectorizando con {weight_type.upper()} y n-grams {vectorizer_args['ngram_range']}...")
        matrix = vectorizer.fit_transform(data)
        vocabulary = vectorizer.vocabulary_

        return matrix, vocabulary, vectorizer
