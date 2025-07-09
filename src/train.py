import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load iris data (no headers by default)
df = pd.read_csv('data/iris.data', header=None)

# Split into features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Scale features for KNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# Save model and scaler
joblib.dump(knn, 'model/model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("Iris KNN model trained and saved.")
