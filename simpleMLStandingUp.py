import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# --- Load Data ---
df = pd.read_csv("lift_off_features_labeled.csv")

# Drop rows with missing values
df = df.dropna()

# Separate features and label
X = df.drop(columns=["Label", "Video", "Frame"])
y = df["Label"]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Random Forest Model ---
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# --- Save Model ---
joblib.dump(model, "standing_up_model_rf.pkl")
print("✅ Model saved as 'standing_up_model_rf.pkl'")

