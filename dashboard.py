
import streamlit as st
import requests
import pandas as pd
API_URL = "http://127.0.0.1:8000"
st.set_page_config(
    page_title="LoanOps AI Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 LoanOps AI")
st.caption("Loan Prediction & Monitoring Dashboard")

# API health status
try:
    response = requests.get(f"{API_URL}/health", timeout=30)
    response.raise_for_status()
    health = response.json()

    if health.get("model_loaded"):
        st.success("API Status: Healthy | ML Model Loaded")
    else:
        st.warning("API is responding, but the model is not loaded.")

except requests.RequestException:
    st.error("API unavailable. Please try again later.")

st.divider()
st.subheader("Loan Prediction")

st.write("Enter applicant details to get a model prediction.")

with st.form("loan_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox(
            "Dependents", ["0", "1", "2", "3+"]
        )
        education = st.selectbox(
            "Education", ["Graduate", "Not Graduate"]
        )
        self_employed = st.selectbox(
            "Self Employed", ["No", "Yes"]
        )
        property_area = st.selectbox(
            "Property Area", ["Urban", "Semiurban", "Rural"]
        )

    with col2:
        applicant_income = st.number_input(
            "Applicant Income", min_value=0, value=5000
        )
        coapplicant_income = st.number_input(
            "Coapplicant Income", min_value=0, value=2000
        )
        loan_amount = st.number_input(
            "Loan Amount", min_value=1, value=150
        )
        loan_term = st.number_input(
            "Loan Term (months)", min_value=1, value=360
        )
        credit_history = st.selectbox(
            "Credit History", [1, 0]
        )

    submitted = st.form_submit_button(
        "Predict Loan Status",
        type="primary",
        use_container_width=True,
    )

if submitted:
    payload = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }

    try:
        with st.spinner("Getting prediction..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=90,
            )

        if response.ok:
            result = response.json()

            st.subheader("Prediction Result")

            if result.get("prediction") == 1:
                st.success("Loan Approved (Model Prediction)")
            else:
                st.error("Loan Not Approved (Model Prediction)")

            probability = result.get("approval_probability")

            if probability is not None:
                st.metric(
                    "Approval Probability",
                    f"{float(probability):.2%}",
                )

            with st.expander("View API Response"):
                st.json(result)

            st.caption(
                "Educational prediction only; not an actual "
                "lending decision."
            )

        else:
            st.error(f"API Error: HTTP {response.status_code}")
            st.code(response.text)

    except requests.RequestException as exc:
        st.error(f"Could not reach the prediction API: {exc}")

st.divider()
st.header("Prediction Monitoring")

if st.button("Refresh Metrics"):
    st.rerun()

try:
    history_response = requests.get(
        f"{API_URL}/predictions",
        params={"limit": 500},
        timeout=15,
    )
    history_response.raise_for_status()

    history_data = history_response.json()
    predictions = history_data.get("predictions", [])

    total = len(predictions)
    approved = sum(
        1 for item in predictions
        if item.get("prediction") == 1
    )
    not_approved = sum(
        1 for item in predictions
        if item.get("prediction") == 0
    )
    approval_rate = (approved / total * 100) if total else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Predictions", total)
    col2.metric("Approved", approved)
    col3.metric("Not Approved", not_approved)
    col4.metric("Approval Rate", f"{approval_rate:.1f}%")

    st.subheader("Recent Prediction History")

    if predictions:
        history_rows = []

        for item in predictions:
            history_rows.append({
                "Record ID": item.get("id"),
                "Timestamp (IST)": (
                    pd.to_datetime(item.get("created_at"), utc=True)
                    .tz_convert("Asia/Kolkata")
                    .strftime("%d-%m-%Y %I:%M:%S %p")
                ),
                "Result": item.get("result"),
                "Prediction": item.get("prediction"),
                "Approval Probability": (
                    f"{item['approval_probability']:.2%}"
                    if item.get("approval_probability") is not None
                    else "N/A"
                ),
            })

        st.dataframe(
            history_rows,
            use_container_width=True,
            hide_index=True,
        )
        
        # Convert history into a DataFrame for charts
        if predictions:
            chart_df = pd.DataFrame(predictions)

            st.subheader("Prediction Analytics")

            approved_count = int(
                (chart_df["prediction"] == 1).sum()
            )
            rejected_count = int(
                (chart_df["prediction"] == 0).sum()
            )

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("#### Approval Distribution")

                distribution_df = pd.DataFrame({
                    "Status": ["Approved", "Not Approved"],
                    "Count": [approved_count, rejected_count],
                }).set_index("Status")

                st.bar_chart(distribution_df)

            with chart_col2:
                st.markdown("#### Prediction Probability")

                probability_df = chart_df[
                    ["id", "approval_probability"]
                ].dropna().sort_values("id")

                if not probability_df.empty:
                    probability_df = probability_df.rename(
                        columns={
                            "id": "Record ID",
                            "approval_probability": "Probability",
                        }
                    ).set_index("Record ID")

                    st.line_chart(probability_df)
                else:
                    st.info("No probability data available yet.")

    else:
        st.info("No predictions recorded yet.")

except requests.RequestException as exc:
    st.error(f"Could not load prediction history: {exc}")
