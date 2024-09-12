from models.Preprocessor import Preprocessor
from models.Resampler import Resampler
from models.ModelPipeline import ModelPipeline
import pandas as pd

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

model = ModelPipeline('src/api/ML/models/pipeline.pkl')
preprocessor = Preprocessor('src/api/ML/data/samu_cases_ready.csv')
processed_data = preprocessor.process()

X_train_resampled, X_test, y_train_resampled, y_test = processed_data


def test_processed_data():
    """"
    It should return a tuple with 4 dataframes
    """
    dataframes = [X_train_resampled, X_test, y_train_resampled, y_test]
    
    for df in dataframes:
        assert isinstance(df, (pd.DataFrame, pd.Series))

def test_model_pipeline():
    """
    It should return a recall over 0.70
    """
    recall = model.recall_mean(X_train=X_train_resampled, y_train=y_train_resampled)
    assert recall > 0.70

def test_predict():
    """
    It should return a prediction
    """

    FEATURES_VALUES = model.get_features()
    
    type_test = FEATURES_VALUES[3][1][0]
    subtype_test = FEATURES_VALUES[4][1][0]
    timestamp_test = 'early' #
    age_test = 1 #
    gender_test = 1 #Female

    print(type_test, subtype_test, timestamp_test, age_test, gender_test)

    
    X_input = pd.DataFrame({
        'timestamp': [timestamp_test],
        'type': [type_test],
        'subtype': [subtype_test],
        'gender': [gender_test],
        'age': [age_test],
    })
    
    result = model.predict(X_input)
    assert result[0] == 1 or result[0] == 0
    


    

