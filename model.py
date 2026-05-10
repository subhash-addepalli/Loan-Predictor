# ============================================================
#  LOAN PREDICTOR - ML MODEL
#  Run this file ONCE to train and save your model.
#  After running, a file called "loan_model.pkl" will appear.
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ── STEP 1: Load the dataset ─────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("train.csv")
print(f"Dataset loaded! Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# ── STEP 2: Drop the Loan_ID column (not useful for prediction)
df.drop("Loan_ID", axis=1, inplace=True)

# ── STEP 3: Fill missing values ───────────────────────────────
# Some rows have empty cells — we fill them with the most common value
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)

# For loan amount, fill with the median (average ignoring outliers)
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)

print("Missing values filled!")

# ── STEP 4: Convert text columns to numbers ───────────────────
# ML models only understand numbers, not words like "Male" or "Yes"
df["Gender"]        = df["Gender"].map({"Male": 1, "Female": 0})
df["Married"]       = df["Married"].map({"Yes": 1, "No": 0})
df["Education"]     = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
df["Loan_Status"]   = df["Loan_Status"].map({"Y": 1, "N": 0})

# Dependents: "3+" becomes 3
df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

# Property_Area has 3 values — convert to numbers
df["Property_Area"] = df["Property_Area"].map({"Urban": 2, "Semiurban": 1, "Rural": 0})

print("Text columns converted to numbers!")

# ── STEP 5: Split into input (X) and output (y) ───────────────
# X = all the input columns (what we know about the applicant)
# y = the answer column (approved or not)
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

print(f"\nInput columns: {list(X.columns)}")
print(f"Target column: Loan_Status (1=Approved, 0=Rejected)")

# ── STEP 6: Split into training data and test data ────────────
# We train on 80% of the data, and test on 20% to check accuracy
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42      # fixes randomness so results are reproducible
)
print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")

# ── STEP 7: Train the model ───────────────────────────────────
# RandomForestClassifier = a powerful, beginner-friendly ML model
# Think of it as 100 decision trees voting together
print("\nTraining the model... (this takes a few seconds)")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained!")

# ── STEP 8: Test accuracy ─────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.1f}%")
print("(This means the model correctly predicts this % of new applications)")

# ── STEP 9: Show which columns matter most ────────────────────
print("\nTop features (most important for prediction):")
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

for feature, importance in feature_importance.items():
    bar = "█" * int(importance * 40)
    print(f"  {feature:<22} {bar} {importance:.3f}")

# ── STEP 10: Save the model to a file ─────────────────────────
# This saves your trained model so Flask can use it later
joblib.dump(model, "loan_model.pkl")
print("\n✅ Model saved as 'loan_model.pkl'")
print("✅ You can now run app.py to start the web server!")