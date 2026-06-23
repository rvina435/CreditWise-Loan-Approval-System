import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="CreditWise Loan Approval",
    page_icon="🏦",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================
model = joblib.load("models/loan_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ==========================
# HEADER
# ==========================
st.title("🏦 CreditWise Loan Approval System")
st.markdown(
    "AI-Powered Loan Approval Prediction using Machine Learning"
)

# ==========================
# SIDEBAR INPUTS
# ==========================
st.sidebar.header("Applicant Details")

applicant_income = st.sidebar.number_input(
    "Applicant Income",
    min_value=0
)

coapplicant_income = st.sidebar.number_input(
    "Coapplicant Income",
    min_value=0
)

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100
)

dependents = st.sidebar.number_input(
    "Dependents",
    min_value=0
)

credit_score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900
)

existing_loans = st.sidebar.number_input(
    "Existing Loans",
    min_value=0
)

dti_ratio = st.sidebar.number_input(
    "DTI Ratio",
    min_value=0.0
)

savings = st.sidebar.number_input(
    "Savings",
    min_value=0
)

collateral_value = st.sidebar.number_input(
    "Collateral Value",
    min_value=0
)

loan_amount = st.sidebar.number_input(
    "Loan Amount",
    min_value=0
)

loan_term = st.sidebar.number_input(
    "Loan Term (Months)",
    min_value=1
)

# ==========================
# TABS
# ==========================
tab1, tab2 = st.tabs(
    ["🔮 Prediction", "📊 Business Insights"]
)

# ==========================
# PREDICTION TAB
# ==========================
with tab1:

    st.subheader("Loan Approval Prediction")

    if st.button("Predict Loan Approval"):

        total_income = (
            applicant_income
            + coapplicant_income
        )

        loan_income_ratio = (
            loan_amount /
            (total_income + 1)
        )

        savings_loan_ratio = (
            savings /
            (loan_amount + 1)
        )

        collateral_coverage = (
            collateral_value /
            (loan_amount + 1)
        )

        input_data = pd.DataFrame([[
            applicant_income,
            coapplicant_income,
            age,
            dependents,
            credit_score,
            existing_loans,
            dti_ratio,
            savings,
            collateral_value,
            loan_amount,
            loan_term,
            total_income,
            loan_income_ratio,
            savings_loan_ratio,
            collateral_coverage,

            1,  # Employment_Status_Salaried
            0,  # Employment_Status_Self-employed
            0,  # Employment_Status_Unemployed

            1,  # Marital_Status_Single

            0,  # Loan_Purpose_Car
            0,  # Loan_Purpose_Education
            1,  # Loan_Purpose_Home
            0,  # Loan_Purpose_Personal

            0,  # Property_Area_Semiurban
            1,  # Property_Area_Urban

            0,  # Education_Level_Not Graduate

            1,  # Gender_Male

            0,  # Employer_Category_Government
            0,  # Employer_Category_MNC
            1,  # Employer_Category_Private
            0   # Employer_Category_Unemployed
        ]])

        input_scaled = scaler.transform(
            input_data
        )

        prediction = model.predict(
            input_scaled
        )

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        # ==========================
        # METRICS
        # ==========================
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Approval %",
            f"{probability*100:.2f}%"
        )

        col2.metric(
            "Credit Score",
            credit_score
        )

        col3.metric(
            "Loan Amount",
            loan_amount
        )

        st.divider()

        if prediction[0] == 1:
            st.success(
                "✅ Loan Approved"
            )
        else:
            st.error(
                "❌ Loan Rejected"
            )

        # ==========================
        # RISK LEVEL
        # ==========================
        if probability > 0.80:
            st.success(
                "🟢 Low Risk Applicant"
            )

        elif probability > 0.50:
            st.warning(
                "🟡 Medium Risk Applicant"
            )

        else:
            st.error(
                "🔴 High Risk Applicant"
            )

        # ==========================
        # APPLICANT SUMMARY
        # ==========================
        with st.expander(
            "Applicant Summary"
        ):

            st.write(
                f"Applicant Income: {applicant_income}"
            )

            st.write(
                f"Credit Score: {credit_score}"
            )

            st.write(
                f"Loan Amount: {loan_amount}"
            )

            st.write(
                f"DTI Ratio: {dti_ratio}"
            )

# ==========================
# BUSINESS INSIGHTS TAB
# ==========================
with tab2:

    st.subheader(
        "Top Business Insights"
    )

    st.info(
        """
        Top Features Affecting Loan Approval:

        1. Credit Score
        2. DTI Ratio
        3. Applicant Income
        4. Loan Income Ratio
        5. Loan Amount
        """
    )

    feature_data = pd.DataFrame({
        "Feature":[
            "Credit Score",
            "DTI Ratio",
            "Applicant Income",
            "Loan Income Ratio",
            "Loan Amount"
        ],
        "Importance":[
            0.289,
            0.270,
            0.056,
            0.049,
            0.042
        ]
    })

    st.bar_chart(
        feature_data.set_index(
            "Feature"
        )
    )