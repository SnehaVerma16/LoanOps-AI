# LoanOps AI

An end-to-end machine learning project for loan approval prediction, featuring a trained scikit-learn pipeline, a FastAPI prediction service, and automated API tests.

## Features

- Data preprocessing and missing-value handling
- Machine learning model comparison and hyperparameter tuning
- Reusable preprocessing and prediction pipeline
- REST API built with FastAPI
- Input validation with Pydantic
- Automated tests with Pytest
- Model evaluation report

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- FastAPI and Uvicorn
- Joblib
- Pytest

## Model Evaluation

Held-out test set results from the current experiment:

| Metric | Score |
|---|---:|
| Accuracy | 85.37% |
| Precision | 83.17% |
| Recall | 98.82% |
| F1-score | 90.32% |
| ROC-AUC | 84.52% |

These results are from a single held-out split and do not guarantee performance on new applicants. The project is educational and is not intended to make real lending decisions.

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start the API:

   ```bash
   python -m uvicorn app:app --reload
   ```

4. Open the interactive API documentation at `http://127.0.0.1:8000/docs`.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Predict loan application outcome |

## Run Tests

```bash
python -m pytest -v
```

The current test suite contains four tests covering the home endpoint, health endpoint, valid prediction, and negative-income validation.

## Project Structure

```text
LoanOps-AI/
├── artifacts/
│   ├── loan_pipeline.joblib
│   └── model_report.json
├── app.py
├── requirements.txt
├── test_app.py
├── README.md
└── .gitignore
```

## Disclaimer

This project is for educational and portfolio purposes only. Predictions should not be treated as actual loan approvals or as a substitute for responsible lending review.
