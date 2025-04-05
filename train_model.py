import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
DATA_FILE = "gesture_data.csv"
df = pd.read_csv(DATA_FILE)

# Split features (X) and labels (y)
X = df.iloc[:, :-1].values  # All columns except the last (landmark positions)
y = df.iloc[:, -1].values   # Last column (gesture label)

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
joblib.dump(clf, "gesture_model.pkl")
print("[INFO] Model saved as 'gesture_model.pkl'.")
