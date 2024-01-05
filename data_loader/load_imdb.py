import pandas as pd

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def read_to_pandas(self):
        df = pd.read_csv(self.data_path)
        return df