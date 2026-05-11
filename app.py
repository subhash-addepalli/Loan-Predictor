# ============================================================
#  LOAN PREDICTOR - FLASK WEB SERVER
#  This file starts your web server and connects your
#  trained ML model to the loan form.
#
#  Run this file with:  python app.py
#  Then open:           http://localhost:5000
# ============================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

# ── Create the Flask app ──────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allows the HTML form to talk to this server

# ── Load the trained model ────────────────────────────────────
print("Loading trained model...")
model = joblib.load("loan_model.pkl")
print("Model loaded successfully!")

# ── Route 1: Serve the HTML form ──────────────────────────────
# When someone opens http://localhost:5000 in their browser,
# Flask will serve the loan-form.html file automatically.
@app.route("/")
def home():
    return send_from_directory(".", "loan-form.html")

# ── Route 2: Handle prediction requests ──────────────────────
# When the form is submitted, it sends data to /predict
# This function reads the data, runs the model, and returns
# "Approved" or "Rejected" back to the browser.
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the form data sent from the browser (as JSON)
        data = request.get_json()
        print(f"\nReceived application: {data}")

        # ── Convert form values to numbers (same as model.py) ──

        gender = 1 if data.get("gender") == "Male" else 0

        married = 1 if data.get("married") == "Married" else 0

        dependents_raw = data.get("dependents", "0")
        dependents = 3 if dependents_raw == "3+" else int(dependents_raw)

        education = 1 if data.get("education") == "Graduate" else 0

        self_employed = 1 if data.get("self_employed") == "Yes" else 0

        applicant_income   = float(data.get("applicant_income", 0))
        coapplicant_income = float(data.get("coapplicant_income", 0))

        loan_amount = float(data.get("loan_amount", 0)) / 1000
        # Note: dataset stores LoanAmount in thousands, so we divide by 1000

        loan_term = float(data.get("loan_amount_term", 360))

        credit_history = float(data.get("credit_history", 1))

        area_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
        property_area = area_map.get(data.get("property_area", "Urban"), 2)

        # ── Build the input array in the exact column order ────
        # This order MUST match the order used when training in model.py
        features = np.array([[
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            property_area
        ]])

        # ── Run the prediction ─────────────────────────────────
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        # prediction = 1 means Approved, 0 means Rejected
        result    = "Approved" if prediction == 1 else "Rejected"
        confidence = round(max(probability) * 100, 1)

        print(f"Prediction: {result} (confidence: {confidence}%)")

        # ── Send the result back to the browser ────────────────
        return jsonify({
            "prediction": result,
            "confidence": confidence,
            "approved":   bool(prediction == 1)
        })

    except Exception as e:
        # If something goes wrong, send the error message back
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400


# ── Start the server ──────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  LoanSense server is running!")
    print("  Open your browser and go to:")
    print("  http://localhost:5000")
    print("="*50 + "\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
