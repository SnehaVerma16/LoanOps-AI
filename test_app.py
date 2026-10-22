
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to LoanOps AI API"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_valid_loan_prediction():
    payload = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 2000,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["result"] in ["Loan Approved", "Loan Not Approved"]
    assert 0 <= result["approval_probability"] <= 1


def test_invalid_negative_income():
    payload = {
        "ApplicantIncome": -5000
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
