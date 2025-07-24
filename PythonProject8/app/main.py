# Importing necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import xgboost as xgb
import pandas as pd
import numpy as np
import json
import os
import logging

# Logging Setup
log_file_path = os.path.join(os.path.dirname(__file__), 'app.log')
logging.basicConfig(
    filename=log_file_path,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Enums for Island and Sex
class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    Male = "male"
    Female = "female"

# Pydantic Input Model

class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island

# Initialize FastAPI
app = FastAPI()
logger.info("FastAPI app initialized.")

# Load XGBoost Model

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "data", "model.json")
model = xgb.XGBClassifier()

try:
    model.load_model(model_path)
    logger.info("Model loaded successfully from %s", model_path)
except Exception as e:
    logger.exception("Failed to load model: %s", e)

# Predict Endpoint

@app.post("/predict")
def predict_penguin(features: PenguinFeatures):
    logger.info("Received prediction request: %s", features.json())

    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])

        # One-hot encoding
        input_data = pd.get_dummies(input_data)

        # Ensure expected columns
        expected_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
                         'body_mass_g', 'year',
                         'sex_female', 'sex_male',
                         'island_Biscoe', 'island_Dream', 'island_Torgersen']

        for col in expected_cols:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[expected_cols]

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Map output
        species_map = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
        species = species_map.get(prediction, "Unknown")

        logger.info("Prediction successful: %s", species)
        return {"predicted_species": species}

    except Exception as e:
        logger.exception("Prediction failed due to error: %s", e)
        return {"error": "Prediction failed. Check server logs for details."}


@app.get("/")
def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the Penguin Classifier API!"}