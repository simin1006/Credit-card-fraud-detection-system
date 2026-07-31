import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================
# LOAD DATA
# ==========================

print("Loading dataset...")

df = pd.read_csv("cleaned_creditcard.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# ==========================
# FEATURES & TARGET
# ==========================

X = df.drop("Class", axis=1)
y = df["Class"]

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Shape :", X_train.shape)
print("Test Shape :", X_test.shape)

# ==========================
# MODEL
# ==========================

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model Training Completed!")

# ==========================
# PREDICTION
# ==========================

y_pred = model.predict(X_test)

# ==========================
# EVALUATION
# ==========================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================
# SAVE MODEL
# ==========================

joblib.dump(model, "fraud_model.pkl")

print("\nModel Saved Successfully!")
print("File Name : fraud_model.pkl")