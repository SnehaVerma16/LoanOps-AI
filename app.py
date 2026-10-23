
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import sqlite3

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
import psycopg

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "loan_pipeline.joblib"


# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")

DATABASE_PATH = Path(
    os.getenv("DATABASE_PATH", str(BASE_DIR / "loanops.db"))
)
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Check that loan_pipeline.joblib is inside the artifacts folder."
    )

model = joblib.load(MODEL_PATH)


def get_connection():
    if DATABASE_URL:
        return psycopg.connect(DATABASE_URL)
    return sqlite3.connect(DATABASE_PATH, timeout=10)


def init_db():
    with get_connection() as conn:
        if DATABASE_URL:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    result TEXT NOT NULL,
                    approval_probability DOUBLE PRECISION
                )
            """)
        else:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    result TEXT NOT NULL,
                    approval_probability REAL
                )
            """)


def save_prediction(input_data, prediction, result, probability):
    created_at = datetime.now(timezone.utc).isoformat()
    serialized_input = json.dumps(input_data, default=str)

    with get_connection() as conn:
        if DATABASE_URL:
            cursor = conn.execute("""
                INSERT INTO predictions (
                    created_at, input_data, prediction,
                    result, approval_probability
                ) VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (
                created_at, serialized_input, prediction,
                result, probability
            ))
            record_id = cursor.fetchone()[0]
        else:
            cursor = conn.execute("""
                INSERT INTO predictions (
                    created_at, input_data, prediction,
                    result, approval_probability
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                created_at, serialized_input, prediction,
                result, probability
            ))
            record_id = cursor.lastrowid

    return record_id


def get_predictions(limit=50):
    with get_connection() as conn:
        if DATABASE_URL:
            cursor = conn.execute("""
                SELECT id, created_at, input_data, prediction,
                       result, approval_probability
                FROM predictions
                ORDER BY id DESC
                LIMIT %s
            """, (limit,))
            rows = cursor.fetchall()
            columns = [item.name for item in cursor.description]
            records = [dict(zip(columns, row)) for row in rows]
        else:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT id, created_at, input_data, prediction,
                       result, approval_probability
                FROM predictions
                ORDER BY id DESC
                LIMIT ?
            """, (limit,)).fetchall()
            records = [dict(row) for row in rows]

    for record in records:
        record["input_data"] = json.loads(record["input_data"])

    return records


init_db()


# Create API
app = FastAPI(
    title="LoanOps AI API",
    description="Loan application prediction using a trained ML pipeline.",
    version="1.1.0",
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
        "history_endpoint": "/predictions",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict_loan(application: LoanApplication):
    try:
        input_dict = application.model_dump()
        input_data = pd.DataFrame([input_dict])

        prediction = int(model.predict(input_data)[0])

        result = {
            1: "Loan Approved",
            0: "Loan Not Approved",
        }.get(prediction, "Unknown")

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            if 1 in classes:
                approved_index = classes.index(1)
                probability = round(
                    float(probabilities[approved_index]), 4
                )

        # Save every successful prediction.
        record_id = save_prediction(
            input_data=input_dict,
            prediction=prediction,
            result=result,
            probability=probability,
        )

        response = {
            "prediction": prediction,
            "result": result,
            "note": (
                "This is a model prediction for educational purposes, "
                "not an actual lending decision."
            ),
            "record_id": record_id,
        }

        if probability is not None:
            response["approval_probability"] = probability

        return response

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction or logging failed: {str(error)}",
        ) from error


@app.get("/predictions")
def prediction_history(
    limit: int = Query(default=50, ge=1, le=500)
):
    try:
        records = get_predictions(limit)
        return {
            "count": len(records),
            "predictions": records,
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve prediction history: {str(error)}",
        ) from error
