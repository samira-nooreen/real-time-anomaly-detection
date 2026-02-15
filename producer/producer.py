"""
Kafka Producer — Simulates high-throughput transaction stream.
Generates realistic transactions with patterns that the ML model can detect.
"""

import json
import time
import random
import sys
import os

from kafka import KafkaProducer


KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
TOPIC = "transactions"
TPS = int(os.environ.get("TPS", "100"))  # transactions per second


def create_producer(retries=30, delay=2):
    """Create Kafka producer with retry logic."""
    for attempt in range(retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda x: json.dumps(x).encode("utf-8"),
                acks="all",
                retries=3,
            )
            print(f"Connected to Kafka at {KAFKA_BROKER}")
            return producer
        except Exception as e:
            print(f"Kafka not ready (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(delay)
    print("Failed to connect to Kafka")
    sys.exit(1)


def generate_transaction():
    """Generate a single transaction with realistic fraud patterns."""
    is_suspicious = random.random() < 0.02  # ~2% anomaly rate

    if is_suspicious:
        return {
            "amount": round(random.uniform(500, 5000), 2),
            "user_id": random.randint(1, 50),
            "merchant_id": random.randint(1, 30),
            "hour_of_day": random.choice([0, 1, 2, 3, 4, 23]),
            "day_of_week": random.randint(0, 6),
            "timestamp": time.time(),
        }
    else:
        return {
            "amount": round(random.expovariate(1 / 80), 2),
            "user_id": random.randint(1, 1000),
            "merchant_id": random.randint(1, 300),
            "hour_of_day": random.randint(8, 22),
            "day_of_week": random.randint(0, 6),
            "timestamp": time.time(),
        }


def main():
    producer = create_producer()
    count = 0
    batch_start = time.time()
    sleep_interval = 1.0 / TPS

    print(f"Producing ~{TPS} transactions/sec to topic '{TOPIC}'")

    try:
        while True:
            tx = generate_transaction()
            producer.send(TOPIC, tx)
            count += 1

            if count % 1000 == 0:
                elapsed = time.time() - batch_start
                rate = 1000 / elapsed if elapsed > 0 else 0
                print(f"Sent {count} transactions (rate: {rate:.0f} tx/s)")
                batch_start = time.time()

            time.sleep(sleep_interval)
    except KeyboardInterrupt:
        print(f"\nStopped after {count} transactions")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
