






class Compute_Metrics():
    
    def __init__(self, y_pred : list[int], y_true : list[int], labels : list[str] = None):
        self.labels = sorted(labels)
        
        if not y_pred or not y_true or len(y_pred) != len(y_true):
            raise("Not valid labels, sizes do not fit or empty data")
        
    
    def __call__(self, *args, **kwds) -> dict: 
        self.compute_all()
    
    
    def compute_all(self):
        pass
    
    def acuraccy(self):
        pass
    
    def macro_F1(self):
        pass
    
    def macro_F1_per_class(self):
        pass
    
    def confusion_matrix(self):
        pass