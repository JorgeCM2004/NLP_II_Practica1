import re
import os
import spacy
import nltk
import pandas as pd
from typing import Literal, List, Tuple, Union, Optional
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from .config import SEED

class Preprocess_Data:

    def __init__(self, library: Literal['NLTK', 'Spacy'] = 'NLTK'):
        
        self.library = library.upper()
        
        if self.library == 'SPACY':
            try:
                self.nlp = spacy.load("en_core_web_lg", disable=["parser", "ner", "textcat"])
                print("SpaCy cargado exitosamente.")
            except IOError:
                raise RuntimeError(
                    "Error: El modelo 'en_core_web_lg' de spaCy no está instalado. "
                    "Ejecuta: python -m spacy download en_core_web_lg"
                )
        
        elif self.library == 'NLTK':
            self.nlp = None
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            
            self.stemmer = SnowballStemmer('english')
            self.stop_words = set(stopwords.words('english'))
            self.regex_pattern = re.compile(r'[^a-zA-Z\s]')
        
        else:
            raise ValueError("Library must be 'NLTK' or 'Spacy'")

    def preprocessing_data(self, data: Union[List[str], pd.Series]) -> List[str]:

        if hasattr(data, 'tolist'):
            data_list = data.tolist()
        else:
            data_list = data

        if self.library == 'SPACY':
            return self._process_with_spacy(data_list)
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

    def _process_with_nltk(self, data: List[str]) -> List[str]:
        processed_data = []
        
        for text in data:
            text = text.lower()
            text = self.regex_pattern.sub('', text)
            tokens = text.split()
            clean_tokens = [
                self.stemmer.stem(token) 
                for token in tokens 
                if token not in self.stop_words and len(token) > 1
            ]
            processed_data.append(" ".join(clean_tokens))
            
        return processed_data

    def data_representation(self, data: List[str], 
                          weight_type: Literal['tf-idf', 'binary', 'frequency'] = 'tf-idf', 
                          max_features: int = 5000, 
                          ngram_range: Tuple[int, int] = (1, 2)):

        weight_type = weight_type.lower()
        
        vectorizer_args = {
            'max_features': max_features, 
            'min_df': 5, 
            'ngram_range': ngram_range 
        } 

        if weight_type == 'tf-idf':
            vectorizer = TfidfVectorizer(**vectorizer_args)
        elif weight_type == 'binary':
            vectorizer = CountVectorizer(binary=True, **vectorizer_args)
        elif weight_type == 'frequency':
            vectorizer = CountVectorizer(binary=False, **vectorizer_args)
        else:
            raise ValueError(f"Weight type '{weight_type}' is not supported. Use 'binary', 'frequency', or 'tf-idf'.")

        matrix = vectorizer.fit_transform(data)
        vocabulary = vectorizer.vocabulary_

        return matrix, vocabulary, vectorizer

    def split_data(self, data: pd.DataFrame, target_col: str = 'genre', test_size: float = 0.2):

        if not isinstance(data, pd.DataFrame):
             raise TypeError("El método split_data tiene que ser un pandas DataFrame.")
             
        train_set, test_set = train_test_split(
            data, 
            test_size=test_size, 
            random_state=SEED, 
            shuffle=True, 
            stratify=data[target_col]
        )
        return train_set, test_set