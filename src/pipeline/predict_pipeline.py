# import sys
# import pandas as pd
# from src.excption import CustomException
# from src.utils import load_object
# import os

# class PredictPipeline:
#     def __init__(self):
#         pass

#     def predict(self,features):
#         try:
#             model_path=os.path.join("artifacts","model.pkl")
#             preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
#             print("Before Loading")
#             model=load_object(file_path=model_path)
#             preprocessor=load_object(file_path=preprocessor_path)
#             print("After Loading")
#             data_scaled=preprocessor.fit_transform(features)
#             preds=model.predict(data_scaled)
#             return preds
        
#         except Exception as e:
#             raise CustomException(e,sys)



# class CustomData:
#     def __init__(  self,
#         gender: str,
#         race_ethnicity: str,
#         parental_level_of_education,
#         lunch: str,
#         test_preparation_course: str,
#         reading_score: int,
#         writing_score: int):

#         self.gender = gender

#         self.race_ethnicity = race_ethnicity

#         self.parental_level_of_education = parental_level_of_education

#         self.lunch = lunch

#         self.test_preparation_course = test_preparation_course

#         self.reading_score = reading_score

#         self.writing_score = writing_score

#     def get_data_as_data_frame(self):
#         try:
#             custom_data_input_dict = {
#                 "gender": [self.gender],
#                 "race_ethnicity": [self.race_ethnicity],
#                 "parental_level_of_education": [self.parental_level_of_education],
#                 "lunch": [self.lunch],
#                 "test_preparation_course": [self.test_preparation_course],
#                 "reading_score": [self.reading_score],
#                 "writing_score": [self.writing_score],
#             }

#             return pd.DataFrame(custom_data_input_dict)

#         except Exception as e:
#             raise CustomException(e, sys)

import sys
import pandas as pd
from src.excption import CustomException
from src.utils import load_object
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            print('error', e)
            raise CustomException(e,sys)


class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


def get_preprocessor():
    numeric_features = ['reading_score', 'writing_score']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    return preprocessor
