import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Sample input (4 features for iris)
# Example: SepalLength, SepalWidth, PetalLength, PetalWidth
sample = pd.DataFrame({
    0: [5.1],  # Sepal length
    1: [3.5],  # Sepal width
    2: [1.4],  # Petal length
    3: [0.2]   # Petal width
})

# Scale sample
sample_scaled = scaler.transform(sample)

# Predict
prediction = model.predict(sample_scaled)[0]
print(f"✅ Predicted Iris class: {prediction}")
