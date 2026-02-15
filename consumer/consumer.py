"""
Kafka Consumer — Real-time anomaly detection on transaction stream.
Loads trained ML models and scores each transaction in <200ms.
Publishes results to monitoring API via shared state.
"""

import json
import time
import sys
import os
import numpy as np

from kafka import KafkaConsumer
import joblib

KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
TOPIC = "transactions"
MODEL_DIR = os.path.dirname(__file__)

# Shared state for monitoring API
stats = {
    "total_processed": 0,
    "total_fraud": 0,
    "total_normal": 0,
    "avg_latency_ms": 0.0,
    "recent_frauds": [],
    "latencies": [],
    "start_time": time.time(),
}


def load_models():
    """Load all model artifacts."""
    print("Loading model artifacts...")
    gb_model = joblib.load(os.path.join(MODEL_DIR, "gb_model.pkl"))
    iso_model = joblib.load(os.path.join(MODEL_DIR, "iso_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
        metadata = json.load(f)

    print(f"Models loaded. Features: {metadata['feature_columns']}")
    print(f"Training metrics — Accuracy: {metadata['accuracy']:.4f}, ROC-AUC: {metadata['roc_auc']:.4f}")
    return gb_model, iso_model, scaler, metadata


def build_features(tx, user_cache, merchant_cache):
    """
    Build feature vector from raw transaction.
    Uses running caches for user/merchant aggregates.
    """
    uid = tx["user_id"]
    mid = tx["merchant_id"]
    amount = tx["amount"]

    # Update user cache
    if uid not in user_cache:
        user_cache[uid] = {"amounts": [], "count": 0}
    user_cache[uid]["amounts"].append(amount)
    user_cache[uid]["count"] += 1
    u = user_cache[uid]
    user_avg = np.mean(u["amounts"][-100:])  # rolling window
    user_std = np.std(u["amounts"][-100:]) if len(u["amounts"]) > 1 else 0.0

    # Update merchant cache
    if mid not in merchant_cache:
        merchant_cache[mid] = {"amounts": [], "count": 0}
    merchant_cache[mid]["amounts"].append(amount)
    merchant_cache[mid]["count"] += 1
    m = merchant_cache[mid]
    merchant_avg = np.mean(m["amounts"][-100:])

    hour = tx["hour_of_day"]
    features = [
        amount,                                        # amount
        np.log1p(amount),                              # log_amount
        np.sin(2 * np.pi * hour / 24),                 # hour_sin
        np.cos(2 * np.pi * hour / 24),                 # hour_cos
        1 if hour < 6 or hour >= 23 else 0,            # is_night
        user_avg,                                       # user_avg_amount
        user_std,                                       # user_std_amount
        u["count"],                                     # user_tx_count
        merchant_avg,                                   # merchant_avg_amount
        m["count"],                                     # merchant_tx_count
        abs(amount - user_avg),                         # amount_deviation
    ]
    return np.array([features])


def create_consumer(retries=30, delay=2):
    for attempt in range(retries):
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=KAFKA_BROKER,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                group_id="anomaly-detector",
            )
            print(f"Connected to Kafka at {KAFKA_BROKER}, consuming '{TOPIC}'")
            return consumer
        except Exception as e:
            print(f"Kafka not ready (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(delay)
    print("Failed to connect to Kafka")
    sys.exit(1)


def get_stats():
    """Return current stats (used by monitoring API)."""
    return stats


def main():
    gb_model, iso_model, scaler, metadata = load_models()
    consumer = create_consumer()

    user_cache = {}
    merchant_cache = {}

    print("Consuming transactions and scoring in real-time...\n")

    for message in consumer:
        start = time.time()
        tx = message.value

        # Build features
        features = build_features(tx, user_cache, merchant_cache)
        features_scaled = scaler.transform(features)

        # Predict
        gb_proba = gb_model.predict_proba(features_scaled)[0][1]
        iso_pred = iso_model.predict(features_scaled)[0]

        is_fraud = (gb_proba > 0.4) or (iso_pred == -1)

        latency_ms = (time.time() - start) * 1000

        # Update stats
        stats["total_processed"] += 1
        stats["latencies"].append(latency_ms)
        if len(stats["latencies"]) > 1000:
            stats["latencies"] = stats["latencies"][-1000:]
        stats["avg_latency_ms"] = np.mean(stats["latencies"])

        if is_fraud:
            stats["total_fraud"] += 1
            fraud_record = {
                **tx,
                "fraud_probability": round(float(gb_proba), 4),
                "iso_anomaly": int(iso_pred == -1),
                "latency_ms": round(latency_ms, 2),
                "detected_at": time.time(),
            }
            stats["recent_frauds"].append(fraud_record)
            if len(stats["recent_frauds"]) > 50:
                stats["recent_frauds"] = stats["recent_frauds"][-50:]

            print(
                f"FRAUD | amount=${tx['amount']:.2f} user={tx['user_id']} "
                f"prob={gb_proba:.3f} latency={latency_ms:.1f}ms"
            )
        else:
            stats["total_normal"] += 1

        if stats["total_processed"] % 500 == 0:
            print(
                f"[{stats['total_processed']}] "
                f"fraud={stats['total_fraud']} "
                f"avg_latency={stats['avg_latency_ms']:.1f}ms"
            )


if __name__ == "__main__":
    main()
