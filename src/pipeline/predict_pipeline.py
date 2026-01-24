import os
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")

            print("✅ Loading model & preprocessor")

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            print("✅ Transforming input data")
            data_scaled = preprocessor.transform(features)

            print("✅ Making prediction")
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            print("🔥 ERROR INSIDE PredictPipeline 🔥")
            print(e)
            raise e  # keep TEMPORARY while debugging


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame({
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            })

        except Exception as e:
            print("🔥 ERROR INSIDE CustomData 🔥")
            print(e)
            raise e
