import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction App")
st.markdown("Enter applicant details below to check loan approval status.")

# Sidebar navigation
option = st.sidebar.radio("Choose Input Mode", ["Single Prediction", "Batch Prediction"])

# -------------------------
# Helper function
# -------------------------
def preprocess_input(df):
    df = df.copy()
    
    # Encoding categorical variables (as done in training)
    cat_map = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
    }

    for col, mapping in cat_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Fill any missing values with median (just in case)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Apply scaling
    numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df

# -------------------------
# Single Prediction UI
# -------------------------
if option == "Single Prediction":
    with st.form("prediction_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        applicant_income = st.number_input("Applicant Income", min_value=0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
        loan_amount = st.number_input("Loan Amount (in 1000s)", min_value=0)
        loan_term = st.number_input("Loan Term (in days)", min_value=0)
        credit_history = st.selectbox("Credit History", [1.0, 0.0])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        })

        processed = preprocess_input(input_df)
        prediction = model.predict(processed)[0]

        result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"
        st.success(f"Prediction: {result}")

# -------------------------
# Batch Prediction UI
# -------------------------
elif option == "Batch Prediction":
    st.markdown("### 📂 Upload a CSV file for Batch Prediction")

    # ✅ Step 1: Instructions for user
    st.info("""
    📌 **Please ensure your CSV contains only the following columns:**

    `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`,  
    `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`,  
    `Loan_Amount_Term`, `Credit_History`, `Property_Area`

    🚫 Do **NOT** include columns like `Loan_ID` or `Loan_Status`.

    ✅ You can also [download a sample CSV template](#) below.
    """)

    # ✅ Step 2: Downloadable Sample Template
    required_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                         'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                         'Loan_Amount_Term', 'Credit_History', 'Property_Area']

    sample_df = pd.DataFrame(columns=required_features)
    sample_csv = sample_df.to_csv(index=False).encode("utf-8")

    st.download_button("📥 Download Sample CSV Template",
                       data=sample_csv,
                       file_name="sample_loan_input.csv",
                       mime="text/csv")

    # ✅ Step 3: Upload CSV
    csv_file = st.file_uploader("Upload CSV", type=["csv"])

    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            st.write("📄 Uploaded Data Preview:", df.head())

            # ✅ Step 4: Keep only required columns
            if 'Loan_ID' in df.columns:
                df = df.drop(columns=['Loan_ID'])

            # ✅ Step 5: Validate all required columns are present
            missing_cols = [col for col in required_features if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing columns in uploaded file: {missing_cols}")
            else:
                # ✅ Step 6: Preprocess & Predict
                processed_df = preprocess_input(df)
                preds = model.predict(processed_df)
                df['Loan_Status_Predicted'] = np.where(preds == 1, "Approved", "Not Approved")

                st.success("✅ Batch predictions complete!")
                st.dataframe(df)

                # ✅ Step 7: Download predictions
                csv_download = df.to_csv(index=False).encode("utf-8")
                st.download_button("📤 Download Predictions as CSV",
                                   data=csv_download,
                                   file_name="loan_predictions.csv",
                                   mime="text/csv")
        except Exception as e:
            st.error(f"⚠️ Error in prediction: {str(e)}")
