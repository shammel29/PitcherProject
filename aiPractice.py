import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dummy biomechanical data
df = pd.DataFrame({
    'elbow_angle': [95, 100, 110, 130, 85, 120],
    'torque': [25.4, 28.7, 30.1, 32.8, 22.9, 29.5],
    'label': [1, 1, 0, 0, 1, 0]
})

# Train/test split
X = df[['elbow_angle', 'torque']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction and accuracy
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
