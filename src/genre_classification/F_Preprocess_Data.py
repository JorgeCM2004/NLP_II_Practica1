from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy
from typing import Literal
from sklearn.model_selection import train_test_split
from .config import SEED

'''
añadir n-gramas y limite del vocabulario
'''

class Preprocess_Data:

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            print("Modelo de spaCy 'en_core_web_sm' cargado.")
        except IOError:
            print("Error: El modelo 'en_core_web_sm' de spaCy no está instalado.")
            print("Ejecuta: python -m spacy download en_core_web_sm")
            self.nlp = None

    def data_representation(self, data, weight_type: Literal['tf-idf', 'binary', 'frequency'] = 'tf-idf'):

        weight_type = weight_type.lower()
        vectorizer = None

        if weight_type == 'tf-idf':
            vectorizer = TfidfVectorizer()

        elif weight_type == 'binary':
            vectorizer = CountVectorizer(binary=True)

        elif weight_type == 'frequency':
            vectorizer = CountVectorizer()

        else:
            raise ValueError(f"The type of weight: '{weight_type}' is not soported." f"Please, use 'binary', 'frequency', or 'tf-idf'.")

        matrix = vectorizer.fit_transform(data)

        vocabulary = vectorizer.vocabulary_

        return matrix, vocabulary, vectorizer

    def preprocessing_data(self, data: list[str]) -> list[list[str]]:

        if self.nlp is None:
            raise RuntimeError("El modelo de spaCy no está cargado.")

        data_preprocesed = []

        for doc in self.nlp.pipe(data, batch_size=50):

            clean_tokens = []
            for token in doc:
                if (not token.is_stop and
                    not token.is_punct and
                    token.is_alpha):

                    clean_tokens.append(token.lemma_.lower())

            data_preprocesed.append(clean_tokens)

        text =  [" ".join(doc_tokens) for doc_tokens in data_preprocesed]

        return text

    def split_data(self, data, test_size=0.2):

        train_set, test_set = train_test_split(data, test_size=test_size, random_state=SEED, shuffle=True, stratify= data['genre'])
        return train_set, test_set
