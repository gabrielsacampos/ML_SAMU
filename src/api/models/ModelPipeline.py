import pickle
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from api.models import Preprocessor
from api.models.CONSTANTS.FEATURES_CONSTANTS import TIMESTAMP, AGE, GENDER, TYPES, SUBTYPES


class ModelPipeline:
    def __init__(self, pipeline_path):
        self.TIMESTAMP = TIMESTAMP
        self.AGE = AGE
        self.GENDER = GENDER
        self.TYPES = TYPES
        self.SUBTYPES = SUBTYPES
        self.model = None
        self.y_pred = None
        self.X_train = None
        self.y_train = None

        if not pipeline_path.endswith('.pkl'):
            raise ValueError('Invalid pipeline path. Please provide a .pkl file.')
        else:
            self.model = pickle.load(open(pipeline_path, 'rb'))
        
        """"
            Features constants
        """
        self.features = (
            ('timestamp', TIMESTAMP),
            ('age', AGE),
            ('gender', GENDER),
            ('types', TYPES),
            ('subtypes', SUBTYPES)
        )


    def get_features(self):
        return self.features

    def predict(self, X_input: pd.DataFrame):
        """"
            Model serialized has already been trained
        """
        self.y_pred = self.model.predict(X_input)
        return self.y_pred

    def recall_mean(self, X_train, y_train, n_folds=5, random_state=42,):
        """
            New training to check the model's performance
        """
        kfold = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
        scores = cross_val_score(self.model, X_train, y_train, cv=kfold, scoring='recall')
        return scores.mean()

    