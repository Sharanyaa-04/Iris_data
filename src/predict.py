import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Sample patient (modify columns to match your trained model)
sample = pd.DataFrame({
    'age': [54],
    'sex': [1],
    'cp': [0],
    'trestbps': [130],
    'chol': [250],
    'fbs': [0],
    'restecg': [1],
    'thalach': [150],
    'exang': [0],
    'oldpeak': [1.2],
    'slope': [1],
    'ca': [0],
    'thal': [2]
})

# Scale sample features
sample_scaled = scaler.transform(sample)

# Predict
prediction = model.predict(sample_scaled)[0]
risk = "At Risk" if prediction == 1 else "Low Risk"
print(f"Prediction: {prediction} ({risk})")
