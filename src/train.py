import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv('data/heart.csv')
X = df.drop('target', axis=1)
y = df['target']

# Scale features for KNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# Save model and scaler
joblib.dump(knn, 'model/model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("Heart disease KNN model trained and saved.")
