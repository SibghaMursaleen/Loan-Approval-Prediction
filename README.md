# Loan Approval Prediction

This project predicts whether a loan application will be **Approved** or **Not Approved** using machine learning models trained on applicant and loan details. It includes a professional Streamlit UI for both single and batch predictions.

## 🚀 Features
- **Single Prediction**: Enter applicant details through a user-friendly form.
- **Batch Prediction**: Upload a CSV file for multiple loan applications at once.
- **Download Results**: Export predictions as a CSV file.
- **Preprocessing**: Automatic handling of missing values, categorical encoding, and scaling.

## 📂 Project Structure
- `loan_model.pkl` → Saved ML model.
- `scaler.pkl` → Saved StandardScaler for input preprocessing.
- `label_encoders.pkl` → Saved encoders for categorical variables.
- `app.py` → Streamlit UI for predictions.
- `requirements.txt` → Dependencies for the project.

## 🛠️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/loan-approval-prediction.git
   cd loan-approval-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage
Run the Streamlit app:
```bash
streamlit run app.py
```

## 📊 Dataset
The model is trained on a loan approval dataset containing applicant demographics, financial details, and credit history.

## 📌 Tech Stack
- **Python**
- **Scikit-learn**
- **Pandas & NumPy**
- **Streamlit**

## ✨ Author
Developed by **Sibgha Mursaleen**  
Feel free to contribute or provide feedback!
