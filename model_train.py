import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")  # Replace with your dataset path

# Features and labels
X = data.drop('label', axis=1)
y = data['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model to a .pkl file
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'random_forest_model.pkl'")
