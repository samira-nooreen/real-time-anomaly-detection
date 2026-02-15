"""
ML Training Pipeline for Real-Time Anomaly Detection
- Generates synthetic credit card fraud dataset with realistic features
- Feature engineering (transaction frequency, rolling averages, time encoding)
- Handles class imbalance with SMOTE
- Trains Gradient Boosting + Isolation Forest ensemble
- Reports Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Saves model artifacts to consumer/
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
)
from imblearn.over_sampling import SMOTE
import joblib
import os
import json

SEED = 42
N_TRANSACTIONS = 50000
FRAUD_RATIO = 0.02
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "consumer")


def generate_dataset(n=N_TRANSACTIONS, fraud_ratio=FRAUD_RATIO):
    """Generate synthetic transaction dataset with realistic patterns."""
    np.random.seed(SEED)

    n_fraud = int(n * fraud_ratio)
    n_normal = n - n_fraud

    # Normal transactions
    normal = pd.DataFrame({
        "amount": np.random.exponential(scale=80, size=n_normal).clip(1, 2000),
        "user_id": np.random.randint(1, 1000, size=n_normal),
        "merchant_id": np.random.randint(1, 300, size=n_normal),
        "hour_of_day": np.random.choice(range(8, 23), size=n_normal),
        "day_of_week": np.random.randint(0, 7, size=n_normal),
        "is_fraud": 0,
    })

    # Fraudulent transactions — higher amounts, odd hours, concentrated users
    fraud = pd.DataFrame({
        "amount": np.random.exponential(scale=500, size=n_fraud).clip(100, 5000),
        "user_id": np.random.randint(1, 50, size=n_fraud),
        "merchant_id": np.random.randint(1, 30, size=n_fraud),
        "hour_of_day": np.random.choice([0, 1, 2, 3, 4, 23], size=n_fraud),
        "day_of_week": np.random.randint(0, 7, size=n_fraud),
        "is_fraud": 1,
    })

    df = pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=SEED)
    df["timestamp"] = pd.date_range("2025-01-01", periods=len(df), freq="s")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def engineer_features(df):
    """Create features that reduce false positives."""

    # User-level aggregates
    user_stats = df.groupby("user_id")["amount"].agg(["mean", "std", "count"]).reset_index()
    user_stats.columns = ["user_id", "user_avg_amount", "user_std_amount", "user_tx_count"]
    user_stats["user_std_amount"] = user_stats["user_std_amount"].fillna(0)
    df = df.merge(user_stats, on="user_id", how="left")

    # Merchant-level aggregates
    merchant_stats = df.groupby("merchant_id")["amount"].agg(["mean", "count"]).reset_index()
    merchant_stats.columns = ["merchant_id", "merchant_avg_amount", "merchant_tx_count"]
    df = df.merge(merchant_stats, on="merchant_id", how="left")

    # Amount deviation from user average
    df["amount_deviation"] = (df["amount"] - df["user_avg_amount"]).abs()

    # Time-of-day encoding (cyclical)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    # Is nighttime transaction
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if h < 6 or h >= 23 else 0)

    # Log amount
    df["log_amount"] = np.log1p(df["amount"])

    return df


FEATURE_COLS = [
    "amount",
    "log_amount",
    "hour_sin",
    "hour_cos",
    "is_night",
    "user_avg_amount",
    "user_std_amount",
    "user_tx_count",
    "merchant_avg_amount",
    "merchant_tx_count",
    "amount_deviation",
]


def train():
    print("=" * 60)
    print("REAL-TIME ANOMALY DETECTION — MODEL TRAINING")
    print("=" * 60)

    # 1. Generate data
    print("\n[1/6] Generating synthetic dataset...")
    df = generate_dataset()
    print(f"  Total transactions: {len(df)}")
    print(f"  Fraud ratio: {df['is_fraud'].mean():.2%}")

    # 2. Feature engineering
    print("\n[2/6] Engineering features...")
    df = engineer_features(df)

    X = df[FEATURE_COLS].values
    y = df["is_fraud"].values

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 4. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Handle imbalance with SMOTE
    print("\n[3/6] Applying SMOTE for class balance...")
    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"  After SMOTE — Normal: {sum(y_train_res == 0)}, Fraud: {sum(y_train_res == 1)}")

    # 6. Train supervised model
    print("\n[4/6] Training GradientBoostingClassifier...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=SEED,
    )
    gb_model.fit(X_train_res, y_train_res)

    # 7. Train unsupervised model (Isolation Forest for anomaly scoring)
    print("\n[5/6] Training IsolationForest...")
    iso_model = IsolationForest(
        n_estimators=200,
        contamination=FRAUD_RATIO,
        random_state=SEED,
    )
    iso_model.fit(X_train_scaled)

    # 8. Evaluate
    print("\n[6/6] Evaluating model...")
    y_pred = gb_model.predict(X_test_scaled)
    y_proba = gb_model.predict_proba(X_test_scaled)[:, 1]

    iso_pred = iso_model.predict(X_test_scaled)
    iso_labels = (iso_pred == -1).astype(int)

    # Ensemble: flag as fraud if either model flags it, but trust GB more
    y_ensemble = ((y_proba > 0.4) | (iso_labels == 1)).astype(int)

    print("\n--- GradientBoosting Results ---")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    print("--- Ensemble Results ---")
    print(classification_report(y_test, y_ensemble, target_names=["Normal", "Fraud"]))

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # False positive rate
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    print(f"False Positive Rate: {fpr:.4f}")

    # 9. Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(gb_model, os.path.join(MODEL_DIR, "gb_model.pkl"))
    joblib.dump(iso_model, os.path.join(MODEL_DIR, "iso_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    metadata = {
        "feature_columns": FEATURE_COLS,
        "accuracy": float((y_pred == y_test).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "false_positive_rate": float(fpr),
        "fraud_ratio": FRAUD_RATIO,
        "n_training_samples": len(X_train_res),
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModels saved to {MODEL_DIR}/")
    print(f"  gb_model.pkl, iso_model.pkl, scaler.pkl, model_metadata.json")
    print(f"\nKey Metrics:")
    print(f"  Accuracy:  {metadata['accuracy']:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  FPR:       {fpr:.4f}")

    return metadata


if __name__ == "__main__":
    train()
