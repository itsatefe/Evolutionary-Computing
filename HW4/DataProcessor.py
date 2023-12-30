import pandas as pd

class DataProcessor:
    def __init__(self,data_path):
        self.path = data_path
        self.dataset = None
        
    def load_data(self):
        self.dataset = pd.read_csv(self.path)
        
    def num_features(self):
        return self.dataset.shape[1] - 1
    
    def get_labels(self):
        return dataset['LABEL']
    



