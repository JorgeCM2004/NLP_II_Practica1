
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .F_Compute_Metrics import Compute_Metrics
from .F_Preprocess_Data import Preprocess_Data

from typing import Literal

class Basic_Models(Preprocess_Data, Compute_Metrics):

    def __init__(self, model_type: Literal['Random_Forest', 'Naive_Bayes', 'LogReg', 'Linear_SVM']):
        
        self.model_type = model_type
        self.model = self._select_model()
        self.prep = Preprocess_Data()
        self.objectPrepro = None

        
    def _select_model(self):
        if self.model_type == 'Random_Forest':
            return RandomForestClassifier(random_state=42)
        elif self.model_type == 'Naive_Bayes':
            return GaussianNB()
        elif self.model_type == 'LogReg':
            return LogisticRegression(solver='liblinear', random_state=42)
        elif self.model_type == 'Linear_SVM':
            return SVC(kernel='linear', probability=True, random_state=42)
        else:
            raise ValueError(f"Tipo de modelo desconocido: {self.model_type}")
        
    def fit(self, data):
        x_train, y_train = self.split_data(data)
        x_train['text'] = self.prep.preprocessing_data(x_train['text'])
        texts_transformed,_,self.objectPrepro = self.prep.data_representation(x_train['text'], 'tf-idf')
        x_train['text'] = texts_transformed
        self.model.fit(x_train, y_train)
         

    def transform(self, data):

        x_test, y_test = self.split_data(data) 
        
        x_test['text'] = self.prep.preprocessing_data(x_test['text'])

        x_test['texts'] = self.objectPrepro.transform(x_test['text'])
        
        y_hat = self.model.predict(x_test)
        
        return y_hat, y_test
        
        
    def evaluate(y_true, y_hat, evaluate_type : Literal["sk_learn_metrics", "all_metrics"]):
        Compute_Metrics(y_hat, y_true)
        
        if evaluate_type.lower() == "all_metrics":
            return Compute_Metrics.compute_all()
        
        elif evaluate_type.lower() == "sk_learn_metrics":
            return Compute_Metrics.full_report()
        
        else:
            raise ValueError(f"Not metric type found: {evaluate_type}")
        
        
        
