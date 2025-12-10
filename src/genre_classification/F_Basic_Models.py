import os
import joblib
import pandas as pd
from typing import Literal

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC 

from .F_Compute_Metrics import Compute_Metrics
from .F_Preprocess_Data import Preprocess_Data
from .config import SEED

class Basic_Models(Preprocess_Data, Compute_Metrics):

    def __init__(self, model_type: Literal['Random_Forest', 'Naive_Bayes', 'LogReg', 'Linear_SVM']):
        self.model_type = model_type
        self.prep = Preprocess_Data()
        self.objectPrepro = None 
        self.model = self._select_model()

    def _select_model(self):
        if self.model_type == 'Random_Forest':
            return RandomForestClassifier(random_state=SEED, n_estimators=200,min_samples_leaf=2,max_features='sqrt', class_weight='balanced_subsample', n_jobs=-1)
        
        elif self.model_type == 'LogReg':
            return LogisticRegression(class_weight='balanced', solver='saga', C=1.2, max_iter=2000, n_jobs=-1)
        
        elif self.model_type == 'Linear_SVM':
            return LinearSVC(class_weight='balanced', dual="auto", C=0.5, random_state=SEED)
        
        elif self.model_type == 'Naive_Bayes':
            return ComplementNB(alpha=0.8)
        
        else:
            raise ValueError(f"Tipo de modelo desconocido: {self.model_type}")

    def fit(self, x, y):
        print(f"Iniciando entrenamiento de {self.model_type}...")
        x['text'] = self.prep.preprocessing_data(x['text'])
        print("-> Preprocesado de texto terminado.")

        texts_transformed, _, self.objectPrepro = self.prep.data_representation(x['text'], weight_type='tf-idf')
        print("-> Transformación a embeddings (TF-IDF) terminada.")

        self.model.fit(texts_transformed, y)
        print("-> Entrenamiento completado exitosamente.")

    def predict(self, x_test):
        print("Iniciando predicción...")
        x_test['text'] = self.prep.preprocessing_data(x_test['text'])

        if self.objectPrepro is None:
            raise ValueError("El modelo no ha sido entrenado o no se ha guardado el vectorizador.")
            
        texts_matrix = self.objectPrepro.transform(x_test['text'])
        
        y_hat = self.model.predict(texts_matrix)
        print("-> Predicción terminada.")
        
        return y_hat

    def evaluate(self, y_true, y_hat, labels, evaluate_type: Literal["sk_learn_metrics", "all_metrics"]):
        metrics = Compute_Metrics(y_hat, y_true, labels)

        if evaluate_type.lower() == "all_metrics":
            return metrics.compute_all()

        elif evaluate_type.lower() == "sk_learn_metrics":
            return metrics.full_report()

        else:
            raise ValueError(f"Tipo de métrica no encontrado: {evaluate_type}")

    def save_model(self, path="./Models/Modelos_Basicos/", name="model"):
        os.makedirs(path, exist_ok=True)
        
        package = {
            'model': self.model,
            'vectorizer': self.objectPrepro
        }
        
        joblib.dump(package, path + name + '.joblib')
        print(f"Modelo guardado en: {path + name}")

    def load_model(self, path="./Models/Modelos_Basicos/", name="model.joblib"):
        full_path = path + name
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"No se encuentra el modelo en {full_path}")

        package = joblib.load(full_path)
        
        if isinstance(package, dict) and 'model' in package:
            self.model = package['model']
            self.objectPrepro = package.get('vectorizer', None)
        else:
            self.model = package
            
        print("Modelo cargado correctamente.")
        return self.model