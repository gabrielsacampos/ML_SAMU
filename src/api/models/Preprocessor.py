import pandas as pd
from sklearn.model_selection import train_test_split
from api.models.Resampler import Resampler

class Preprocessor:
    def __init__(self, path_dataset):
        self.dataset = pd.read_csv(path_dataset)
        self.X_train = None
        self.y_train = None
        self.X_train_resampled = None
        self.y_train_resampled = None
        self.X_test = None
        self.y_test = None
    
    def process(self, test_size=0.2, seed=42):
        X = self.dataset.drop(columns='outcome')
        y = self.dataset['outcome']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        resampler = Resampler(self.X_train, self.y_train)
        self.X_train_resampled, self.y_train_resampled = resampler.execute_undersampler()
        
        return self.X_train_resampled, self.X_test, self.y_train_resampled,  self.y_test 