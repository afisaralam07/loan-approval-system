from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'loan_prediction_secret_key_2024'

# Load model and encoders
try:
    model = pickle.load(open("loan_model.pkl", "rb"))
    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    label_encoders = None

def safe_encode(column, value):
    """Safely encode categorical values"""
    if label_encoders is None or column not in label_encoders:
        return 0
    le = label_encoders[column]
    try:
        return le.transform([str(value)])[0]
    except ValueError:
        return le.transform([le.classes_[0]])[0]

@app.route("/")
def home():
    """Home page with input form"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Process form data and redirect to result page"""
    try:
        # Get all form data
        form_data = {
            'ApplicantName': request.form.get("ApplicantName", "Applicant"),
            'Gender': request.form.get("Gender", "Male"),
            'Married': request.form.get("Married", "No"),
            'Education': request.form.get("Education", "Graduate"),
            'Self_Employed': request.form.get("Self_Employed", "No"),
            'Property_Area': request.form.get("Property_Area", "Urban"),
            'ApplicantIncome': float(request.form.get("ApplicantIncome", 0)),
            'CoapplicantIncome': float(request.form.get("CoapplicantIncome", 0)),
            'LoanAmount_input': float(request.form.get("LoanAmount", 0)),
            'Loan_Amount_Term': float(request.form.get("Loan_Amount_Term", 360)),
            'Credit_History': float(request.form.get("Credit_History", 1)),
            'Dependents': float(request.form.get("Dependents", 0))
        }

        # Store in session-like dictionary (we'll pass via template)
        # Calculate features
        LoanAmount = form_data['LoanAmount_input'] / 1000
        Total_Income = form_data['ApplicantIncome'] + form_data['CoapplicantIncome']
        Income_Per_Dependent = Total_Income / (form_data['Dependents'] + 1)
        Loan_to_Income_Ratio = (form_data['LoanAmount_input']) / Total_Income if Total_Income > 0 else 0
        
        # Calculate EMI
        r = 0.10 / 12
        n = form_data['Loan_Amount_Term']
        P = form_data['LoanAmount_input']
        if r > 0 and n > 0:
            EMI = (P * r * (1 + r)**n) / ((1 + r)**n - 1)
        else:
            EMI = 0
        
        Balance_Income = Total_Income - (EMI * 12)
        Credit_Category = 'Good' if form_data['Credit_History'] >= 1 else 'Poor'

        # Encode categorical variables
        Gender_enc = safe_encode('Gender', form_data['Gender'])
        Married_enc = safe_encode('Married', form_data['Married'])
        Education_enc = safe_encode('Education', form_data['Education'])
        Self_Employed_enc = safe_encode('Self_Employed', form_data['Self_Employed'])
        Property_Area_enc = safe_encode('Property_Area', form_data['Property_Area'])
        Credit_Category_enc = safe_encode('Credit_Category', Credit_Category)

        # Create feature array
        features = np.array([[
            Gender_enc, Married_enc, form_data['Dependents'], Education_enc,
            Self_Employed_enc, form_data['ApplicantIncome'], form_data['CoapplicantIncome'],
            LoanAmount, form_data['Loan_Amount_Term'], form_data['Credit_History'],
            Property_Area_enc, Total_Income, Income_Per_Dependent,
            Loan_to_Income_Ratio, EMI, Balance_Income, Credit_Category_enc
        ]])

        # Make prediction
        if model is not None:
            prediction = model.predict(features)
            probability = model.predict_proba(features)
            approval_prob = round(probability[0][1] * 100, 2)
        else:
            # Fallback if model not loaded
            approval_prob = 50.0

        # Determine result
        if approval_prob >= 75:
            result_status = "approved"
            result_text = "Loan Approved"
            risk_level = "Low Risk"
            color ="#10b981"
            recommendation = "Excellent profile! Your loan application is likely to be approved with favorable interest rates."
        elif approval_prob >= 60:
            result_status = "likely"
            result_text = "Loan Likely Approved ✓"
            risk_level = "Medium-Low Risk"
            color = "#3b82f6"
            recommendation = "Good profile. You have strong chances of approval. Consider applying with a co-applicant for better terms."
        elif approval_prob >= 45:
            result_status = "review"
            result_text = "Under Review"
            risk_level = "Medium Risk"
            color = "#f59e0b"
            recommendation = "Borderline case. Bank may require additional documentation or collateral. Consider reducing loan amount."
        else:
            result_status = "rejected"
            result_text = "Loan Rejected"
            risk_level = "High Risk"
            color = "#ef4444"
            recommendation = "High risk profile. Work on improving credit score, increasing income stability, or reducing existing debts before reapplying."

        # Calculate DTI
        dti_ratio = (EMI * 12 / Total_Income * 100) if Total_Income > 0 else 0

        # Prepare all data for result page
        result_data = {
            'applicant_name': form_data['ApplicantName'],
            'result_text': result_text,
            'result_status': result_status,
            'approval_prob': approval_prob,
            'rejection_prob': round(100 - approval_prob, 2),
            'risk_level': risk_level,
            'color': color,
            'recommendation': recommendation,
            'emi': f"₹{EMI:,.0f}",
            'total_income': f"₹{Total_Income:,.0f}",
            'dti_ratio': round(dti_ratio, 1),
            'balance_income': f"₹{Balance_Income:,.0f}",
            'loan_amount': f"₹{form_data['LoanAmount_input']:,.0f}",
            'loan_term': int(form_data['Loan_Amount_Term']),
            'monthly_income': f"₹{form_data['ApplicantIncome']:,.0f}",
            'coapplicant_income': f"₹{form_data['CoapplicantIncome']:,.0f}",
            'dependents': int(form_data['Dependents']),
            'credit_history': "Good" if form_data['Credit_History'] == 1 else "Poor",
            'property_area': form_data['Property_Area'],
            'education': form_data['Education'],
            'employment': "Self Employed" if form_data['Self_Employed'] == "Yes" else "Salaried",
            'marital_status': "Married" if form_data['Married'] == "Yes" else "Unmarried",
            'gender': form_data['Gender']
        }

        return render_template("result.html", **result_data)

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        flash("Error processing your application. Please check all inputs and try again.", "error")
        return redirect(url_for('home'))

@app.route("/about")
def about():
    """About page"""
    return render_template("about.html")

@app.errorhandler(404)
def not_found(e):
    return render_template("index.html"), 404

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)