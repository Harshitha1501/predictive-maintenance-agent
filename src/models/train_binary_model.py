"""
Binary Failure Classification Model Trainer.
Trains a high-performance, class-balanced Random Forest classifier to predict
Machine Failure (0: No Failure, 1: Failure) and compute precise failure probabilities.
Produces 'models/binary_model.pkl' for Jatin's binary classification pipeline.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.preprocessor import CANONICAL_FEATURES


def train_binary_model(
    data_path: str = "data/ai4i2020.csv",
    model_save_path: str = "models/binary_model.pkl",
    random_state: int = 42,
) -> dict:
    """Trains and serializes the binary classification failure prediction model."""
    print(f"Loading training data from {data_path}...")
    df = pd.read_csv(data_path)

    # Encode categorical 'Type'
    le_type = LabelEncoder()
    df["Type"] = le_type.fit_transform(df["Type"])

    X = df[CANONICAL_FEATURES]
    y = df["Machine failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    print(f"Training dataset: {X_train.shape[0]} samples, Testing dataset: {X_test.shape[0]} samples")

    # Balanced Random Forest to address the 3.4% failure class imbalance
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=4,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["No Failure", "Machine Failure"])

    print("=== Binary Classification Report ===")
    print(report)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model_bundle = {
        "model": clf,
        "feature_names": CANONICAL_FEATURES,
        "target_names": ["No Failure", "Machine Failure"],
        "feature_encoders": {"Type": le_type},
        "roc_auc": roc_auc,
        "classification_report": report,
    }

    joblib.dump(model_bundle, model_save_path)
    print(f"Binary classification model saved successfully to: {model_save_path}")
    return model_bundle


if __name__ == "__main__":
    train_binary_model()
