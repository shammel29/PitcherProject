# train_final.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# 1) load features
df = pd.read_csv("features_confirmed.csv")
X = df.drop(["Video","Frame","Keyframe_Type"], axis=1)
y = df["Keyframe_Type"]

# 2) split & train
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 3) evaluate
print(classification_report(y_val, model.predict(X_val)))

# 4) save
joblib.dump(model, "keyframe_classifier_final.joblib")
