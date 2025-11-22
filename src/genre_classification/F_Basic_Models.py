
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .F_Compute_Metrics import Compute_Metrics
#from .F_Preprocess_Data import Preprocess_Data
from .F_Preprocess_2 import Preprocess_Data_2
from .config import SEED

from typing import Literal

class Basic_Models(Preprocess_Data_2, Compute_Metrics):

    def __init__(self, model_type: Literal['Random_Forest', 'Naive_Bayes', 'LogReg', 'Linear_SVM']):

        self.model_type = model_type
        self.model = self._select_model()
        self.prep = Preprocess_Data_2()
        self.objectPrepro = None


    def _select_model(self):
        if self.model_type == 'Random_Forest':
            return RandomForestClassifier(random_state=SEED, n_jobs=-1, n_estimators=100)
        elif self.model_type == 'Naive_Bayes':
            return MultinomialNB()
        elif self.model_type == 'LogReg':
            return LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=SEED, max_iter=1000)
        elif self.model_type == 'Linear_SVM':
            return SVC(random_state=SEED, dual="auto")
        else:
            raise ValueError(f"Tipo de modelo desconocido: {self.model_type}")

    def fit(self, x, y):
        x['text'] = self.prep.preprocessing_data(x['text'])
        print("Ha terminado el preprocesado")
        texts_transformed,_,self.objectPrepro = self.prep.data_representation(x['text'], 'tf-idf')
        print("Ha terminado la transfromacion al embbeding")
        self.model.fit(texts_transformed, y)
        print("Ha terminado de entrenar")

    def transform(self, x_test):
        x_test['text'] = self.prep.preprocessing_data(x_test['text'])
        print("Ha terminado de preprocesar entrenamiento")

        texts_matrix = self.objectPrepro.transform(x_test['text'])
        print("Ha terminado de representar entrenamiento")

        y_hat = self.model.predict(texts_matrix)

        return y_hat


    def evaluate(self, y_true, y_hat, labels, evaluate_type : Literal["sk_learn_metrics", "all_metrics"]):
        metrics = Compute_Metrics(y_hat, y_true, labels)

        if evaluate_type.lower() == "all_metrics":
            return metrics.compute_all()

        elif evaluate_type.lower() == "sk_learn_metrics":
            return metrics.full_report()

        else:
            raise ValueError(f"Not metric type found: {evaluate_type}")



