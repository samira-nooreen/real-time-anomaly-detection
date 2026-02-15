"""
Consumer runner — wraps consumer.py and periodically writes stats to a JSON file
so the monitoring API can read them (inter-process communication via filesystem).
"""

import json
import time
import sys
import os
import threading
import numpy as np

from kafka import KafkaConsumer
import joblib

KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "kafka:9092")
TOPIC = "transactions"
MODEL_DIR = os.path.dirname(__file__)
STATS_FILE = os.environ.get("STATS_FILE", "/tmp/anomaly_stats.json")

stats = {
    "total_processed": 0,
    "total_fraud": 0,
    "total_normal": 0,
    "avg_latency_ms": 0.0,
    "recent_frauds": [],
    "start_time": time.time(),
}
_latencies = []


def write_stats_periodically():
    """Background thread that writes stats to file every second."""
    while True:
        try:
            with open(STATS_FILE, "w") as f:
                json.dump(stats, f)
        except Exception:
            pass
        time.sleep(1)


def load_models():
    print("Loading model artifacts...")
    gb_model = joblib.load(os.path.join(MODEL_DIR, "gb_model.pkl"))
    iso_model = joblib.load(os.path.join(MODEL_DIR, "iso_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
        metadata = json.load(f)
    print(f"Models loaded. ROC-AUC: {metadata['roc_auc']:.4f}")
    return gb_model, iso_model, scaler, metadata


def build_features(tx, user_cache, merchant_cache):
    uid = tx["user_id"]
    mid = tx["merchant_id"]
    amount = tx["amount"]

    if uid not in user_cache:
        user_cache[uid] = {"amounts": [], "count": 0}
    user_cache[uid]["amounts"].append(amount)
    user_cache[uid]["count"] += 1
    u = user_cache[uid]
    user_avg = float(np.mean(u["amounts"][-100:]))
    user_std = float(np.std(u["amounts"][-100:])) if len(u["amounts"]) > 1 else 0.0

    if mid not in merchant_cache:
        merchant_cache[mid] = {"amounts": [], "count": 0}
    merchant_cache[mid]["amounts"].append(amount)
    merchant_cache[mid]["count"] += 1
    m = merchant_cache[mid]
    merchant_avg = float(np.mean(m["amounts"][-100:]))

    hour = tx["hour_of_day"]
    return np.array([[
        amount,
        np.log1p(amount),
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        1 if hour < 6 or hour >= 23 else 0,
        user_avg,
        user_std,
        u["count"],
        merchant_avg,
        m["count"],
        abs(amount - user_avg),
    ]])


def create_consumer(retries=60, delay=3):
    for attempt in range(retries):
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=KAFKA_BROKER,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                group_id="anomaly-detector",
            )
            print(f"Connected to Kafka at {KAFKA_BROKER}")
            return consumer
        except Exception as e:
            print(f"Waiting for Kafka ({attempt + 1}/{retries}): {e}")
            time.sleep(delay)
    print("Failed to connect to Kafka")
    sys.exit(1)


def main():
    global _latencies

    # Start stats writer thread
    t = threading.Thread(target=write_stats_periodically, daemon=True)
    t.start()

    gb_model, iso_model, scaler, metadata = load_models()
    consumer = create_consumer()

    user_cache = {}
    merchant_cache = {}

    print("Real-time anomaly detection running...\n")

    for message in consumer:
        start = time.time()
        tx = message.value

        features = build_features(tx, user_cache, merchant_cache)
        features_scaled = scaler.transform(features)

        gb_proba = gb_model.predict_proba(features_scaled)[0][1]
        iso_pred = iso_model.predict(features_scaled)[0]
        is_fraud = (gb_proba > 0.4) or (iso_pred == -1)

        latency_ms = (time.time() - start) * 1000

        stats["total_processed"] += 1
        _latencies.append(latency_ms)
        if len(_latencies) > 1000:
            _latencies = _latencies[-1000:]
        stats["avg_latency_ms"] = float(np.mean(_latencies))

        if is_fraud:
            stats["total_fraud"] += 1
            fraud_record = {
                "amount": tx["amount"],
                "user_id": tx["user_id"],
                "merchant_id": tx["merchant_id"],
                "hour_of_day": tx["hour_of_day"],
                "fraud_probability": round(float(gb_proba), 4),
                "iso_anomaly": int(iso_pred == -1),
                "latency_ms": round(latency_ms, 2),
                "detected_at": time.time(),
            }
            stats["recent_frauds"].append(fraud_record)
            if len(stats["recent_frauds"]) > 50:
                stats["recent_frauds"] = stats["recent_frauds"][-50:]
            print(f"FRAUD | ${tx['amount']:.2f} user={tx['user_id']} prob={gb_proba:.3f} latency={latency_ms:.1f}ms")
        else:
            stats["total_normal"] += 1

        if stats["total_processed"] % 500 == 0:
            print(f"[{stats['total_processed']}] fraud={stats['total_fraud']} avg_lat={stats['avg_latency_ms']:.1f}ms")


if __name__ == "__main__":
    main()
