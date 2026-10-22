
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "loan_pipeline.joblib"

# Load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Check that loan_pipeline.joblib is inside the artifacts folder."
    )

model = joblib.load(MODEL_PATH)

# Create API
app = FastAPI(
    title="LoanOps AI API",
    description="Loan application prediction using a trained ML pipeline.",
    version="1.0.0",
)


class LoanApplication(BaseModel):
    Gender: Optional[str] = None
    Married: Optional[str] = None
    Dependents: Optional[str] = None
    Education: Optional[str] = None
    Self_Employed: Optional[str] = None
    ApplicantIncome: Optional[float] = Field(default=None, ge=0)
    CoapplicantIncome: Optional[float] = Field(default=None, ge=0)
    LoanAmount: Optional[float] = Field(default=None, ge=0)
    Loan_Amount_Term: Optional[float] = Field(default=None, gt=0)
    Credit_History: Optional[float] = Field(default=None, ge=0, le=1)
    Property_Area: Optional[str] = None


@app.get("/")
def home():
    return {
        "message": "Welcome to LoanOps AI API",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict_loan(application: LoanApplication):
    try:
        # Convert incoming JSON to a DataFrame.
        input_data = pd.DataFrame([application.model_dump()])

        # Use the complete saved pipeline for preprocessing and prediction.
        prediction = int(model.predict(input_data)[0])

        result = {
            1: "Loan Approved",
            0: "Loan Not Approved",
        }.get(prediction, "Unknown")

        response = {
            "prediction": prediction,
            "result": result,
            "note": (
                "This is a model prediction for educational purposes, "
                "not an actual lending decision."
            ),
        }

        # Include probability only if the model supports it.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            if 1 in classes:
                approved_index = classes.index(1)
                response["approval_probability"] = round(
                    float(probabilities[approved_index]), 4
                )

        return response

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        ) from error
