

import pandas as pd

from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, classification_report)

from sklearn.utils.multiclass import unique_labels


class Compute_Metrics():
    
    def __init__(self, y_pred : list[int], y_true : list[int], labels : list[str] = None):
        self.labels = sorted(labels)
        
        if y_true is None or y_pred is None:
            raise ValueError(" 'y_true' and 'y_pred' can not be none None.")
        
        if len(y_true) != len(y_pred):
            raise ValueError(f"The size of 'y_true' ({len(y_true)}) and 'y_pred' ({len(y_pred)}) are not the same.")
            
        if len(y_true) == 0:
            raise ValueError("The label lists can not be empty.")
        
        self.y_true = y_true
        self.y_pred = y_pred
        
        if labels is None:
            self.labels = sorted(list(unique_labels(y_true, y_pred)))
        else:
            self.labels = sorted(list(labels))
        
    
    def __call__(self, *args, **kwds): 
        return self.compute_all()
    
    
    def compute_all(self):
        metrics = {
            "accuracy": self.accuracy(),
            "macro_f1": self.macro_f1(),
            "f1_per_class": self.macro_f1_per_class(),
            "confusion_matrix": self.confusion_matrix()
        }
        return metrics
    
    def accuracy(self) -> float:
        acc = accuracy_score(self.y_true, self.y_pred)
        print(acc)
        return acc
    
    def macro_f1(self)-> float:
        return f1_score(self.y_true, self.y_pred, labels=self.labels, average='macro', zero_division=0)
    
    def macro_f1_per_class(self):
        scores = f1_score(self.y_true, self.y_pred, labels=self.labels, average=None, zero_division=0)
        return dict(zip(self.labels, scores))
    
    def confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred, labels=self.labels)
        cm_df = pd.DataFrame(cm, index=self.labels, columns=self.labels)
        cm_df.index.name = 'Real'
        cm_df.columns.name = 'Predicción'
        return cm_df
    
    
    def full_report(self, as_dict: bool = True):

        return classification_report(self.y_true, self.y_pred, labels=self.labels, output_dict=as_dict, zero_division=0)
    