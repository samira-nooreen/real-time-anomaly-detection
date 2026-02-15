"""
FastAPI Monitoring Dashboard — Real-time metrics for the anomaly detection system.
Includes a built-in transaction simulator that feeds the ML models directly,
so the dashboard works standalone without Kafka.
"""

import json
import time
import os
import random
import threading

import numpy as np
import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Anomaly Detection Monitor")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "consumer")

# In-memory stats updated by the simulator thread
stats = {
    "total_processed": 0,
    "total_fraud": 0,
    "total_normal": 0,
    "avg_latency_ms": 0.0,
    "recent_frauds": [],
    "latencies": [],
    "start_time": time.time(),
}

# Model artifacts (loaded once at startup)
_gb_model = None
_iso_model = None
_scaler = None
_metadata = None
_user_cache = {}
_merchant_cache = {}
_sim_running = False


def load_models():
    global _gb_model, _iso_model, _scaler, _metadata
    _gb_model = joblib.load(os.path.join(MODEL_DIR, "gb_model.pkl"))
    _iso_model = joblib.load(os.path.join(MODEL_DIR, "iso_model.pkl"))
    _scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
        _metadata = json.load(f)
    print(f"Models loaded. Accuracy={_metadata['accuracy']:.4f}, ROC-AUC={_metadata['roc_auc']:.4f}")


def generate_transaction():
    is_suspicious = random.random() < 0.03
    if is_suspicious:
        return {
            "amount": round(random.uniform(800, 5000), 2),
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


def build_features(tx):
    uid = tx["user_id"]
    mid = tx["merchant_id"]
    amount = tx["amount"]

    if uid not in _user_cache:
        _user_cache[uid] = {"amounts": [], "count": 0}
    _user_cache[uid]["amounts"].append(amount)
    _user_cache[uid]["count"] += 1
    u = _user_cache[uid]
    user_avg = float(np.mean(u["amounts"][-100:]))
    user_std = float(np.std(u["amounts"][-100:])) if len(u["amounts"]) > 1 else 0.0

    if mid not in _merchant_cache:
        _merchant_cache[mid] = {"amounts": [], "count": 0}
    _merchant_cache[mid]["amounts"].append(amount)
    _merchant_cache[mid]["count"] += 1
    m = _merchant_cache[mid]
    merchant_avg = float(np.mean(m["amounts"][-100:]))

    hour = tx["hour_of_day"]
    features = [
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
    ]
    return np.array([features])


def score_transaction(tx):
    features = build_features(tx)
    features_scaled = _scaler.transform(features)
    gb_proba = float(_gb_model.predict_proba(features_scaled)[0][1])
    iso_pred = int(_iso_model.predict(features_scaled)[0])
    is_fraud = (gb_proba > 0.4) or (iso_pred == -1)
    return gb_proba, iso_pred, is_fraud


def simulator_loop():
    """Background thread: generates transactions, scores them, updates stats."""
    global _sim_running
    _sim_running = True
    tps = 50  # transactions per second
    sleep_interval = 1.0 / tps

    print(f"Simulator started: ~{tps} tx/sec")

    while _sim_running:
        tx = generate_transaction()
        start = time.time()
        gb_proba, iso_pred, is_fraud = score_transaction(tx)
        latency_ms = (time.time() - start) * 1000

        stats["total_processed"] += 1
        stats["latencies"].append(latency_ms)
        if len(stats["latencies"]) > 1000:
            stats["latencies"] = stats["latencies"][-1000:]
        stats["avg_latency_ms"] = float(np.mean(stats["latencies"]))

        if is_fraud:
            stats["total_fraud"] += 1
            fraud_record = {
                **tx,
                "fraud_probability": round(gb_proba, 4),
                "iso_anomaly": int(iso_pred == -1),
                "latency_ms": round(latency_ms, 2),
                "detected_at": time.time(),
            }
            stats["recent_frauds"].append(fraud_record)
            if len(stats["recent_frauds"]) > 50:
                stats["recent_frauds"] = stats["recent_frauds"][-50:]
        else:
            stats["total_normal"] += 1

        time.sleep(sleep_interval)


@app.on_event("startup")
async def startup():
    load_models()
    t = threading.Thread(target=simulator_loop, daemon=True)
    t.start()


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anomaly Detection Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 24px 32px;
            border-bottom: 1px solid #2a2a4a;
        }
        .header h1 { font-size: 24px; font-weight: 600; color: #fff; }
        .header .subtitle { color: #888; font-size: 14px; margin-top: 4px; }
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 12px;
        }
        .status-live {
            background: rgba(0, 255, 136, 0.15);
            color: #00ff88;
            border: 1px solid rgba(0, 255, 136, 0.3);
        }
        .status-offline {
            background: rgba(255, 68, 68, 0.15);
            color: #ff4444;
            border: 1px solid rgba(255, 68, 68, 0.3);
        }
        .container { padding: 24px 32px; max-width: 1400px; margin: 0 auto; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .metric-card {
            background: #12121a;
            border: 1px solid #2a2a3a;
            border-radius: 12px;
            padding: 20px;
            transition: border-color 0.2s;
        }
        .metric-card:hover { border-color: #4a4a6a; }
        .metric-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #666;
            margin-bottom: 8px;
        }
        .metric-value { font-size: 32px; font-weight: 700; color: #fff; }
        .metric-value.fraud { color: #ff4444; }
        .metric-value.success { color: #00ff88; }
        .metric-value.warning { color: #ffaa00; }
        .metric-value.info { color: #4488ff; }
        .section-title { font-size: 16px; font-weight: 600; margin-bottom: 16px; color: #fff; }
        .alerts-table {
            width: 100%;
            border-collapse: collapse;
            background: #12121a;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #2a2a3a;
        }
        .alerts-table th {
            background: #1a1a2e;
            padding: 12px 16px;
            text-align: left;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #666;
            border-bottom: 1px solid #2a2a3a;
        }
        .alerts-table td {
            padding: 10px 16px;
            font-size: 14px;
            border-bottom: 1px solid #1a1a2a;
            font-family: 'SF Mono', monospace;
        }
        .alerts-table tr:hover { background: #1a1a2a; }
        .fraud-badge {
            background: rgba(255, 68, 68, 0.15);
            color: #ff4444;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .model-info {
            background: #12121a;
            border: 1px solid #2a2a3a;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }
        .model-info .row {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            font-size: 14px;
        }
        .model-info .label { color: #666; }
        .model-info .value { color: #fff; font-weight: 500; }
        .no-data { text-align: center; padding: 40px; color: #444; font-size: 14px; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .live-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #00ff88;
            border-radius: 50%;
            margin-right: 6px;
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>
            Real-Time Anomaly Detection
            <span class="status-badge status-live" id="statusBadge">
                <span class="live-dot"></span>LIVE
            </span>
        </h1>
        <div class="subtitle">ML-Powered Fraud Detection System | Kafka + Scikit-learn</div>
    </div>

    <div class="container">
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Processed</div>
                <div class="metric-value info" id="totalProcessed">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Fraud Detected</div>
                <div class="metric-value fraud" id="totalFraud">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Normal</div>
                <div class="metric-value success" id="totalNormal">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Fraud Rate</div>
                <div class="metric-value warning" id="fraudRate">0%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Latency</div>
                <div class="metric-value" id="avgLatency">0ms</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Throughput</div>
                <div class="metric-value" id="throughput">0 tx/s</div>
            </div>
        </div>

        <div class="model-info" id="modelInfo" style="display:none">
            <div class="section-title">Model Performance</div>
            <div class="row"><span class="label">Accuracy</span><span class="value" id="mAccuracy">-</span></div>
            <div class="row"><span class="label">Precision</span><span class="value" id="mPrecision">-</span></div>
            <div class="row"><span class="label">Recall</span><span class="value" id="mRecall">-</span></div>
            <div class="row"><span class="label">F1 Score</span><span class="value" id="mF1">-</span></div>
            <div class="row"><span class="label">ROC-AUC</span><span class="value" id="mAUC">-</span></div>
            <div class="row"><span class="label">False Positive Rate</span><span class="value" id="mFPR">-</span></div>
        </div>

        <div class="section-title">Recent Fraud Alerts</div>
        <table class="alerts-table">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Amount</th>
                    <th>User ID</th>
                    <th>Merchant</th>
                    <th>Probability</th>
                    <th>Latency</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody id="alertsBody">
                <tr><td colspan="7" class="no-data">Waiting for data...</td></tr>
            </tbody>
        </table>
    </div>

    <script>
        async function fetchStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();

                document.getElementById('totalProcessed').textContent = data.total_processed.toLocaleString();
                document.getElementById('totalFraud').textContent = data.total_fraud.toLocaleString();
                document.getElementById('totalNormal').textContent = data.total_normal.toLocaleString();

                const fraudRate = data.total_processed > 0
                    ? ((data.total_fraud / data.total_processed) * 100).toFixed(2)
                    : 0;
                document.getElementById('fraudRate').textContent = fraudRate + '%';
                document.getElementById('avgLatency').textContent = data.avg_latency_ms.toFixed(1) + 'ms';

                const uptime = (Date.now() / 1000) - data.start_time;
                const tps = uptime > 0 ? (data.total_processed / uptime).toFixed(0) : 0;
                document.getElementById('throughput').textContent = tps + ' tx/s';

                if (data.model_metadata) {
                    const m = data.model_metadata;
                    document.getElementById('modelInfo').style.display = 'block';
                    document.getElementById('mAccuracy').textContent = (m.accuracy * 100).toFixed(2) + '%';
                    document.getElementById('mPrecision').textContent = (m.precision * 100).toFixed(2) + '%';
                    document.getElementById('mRecall').textContent = (m.recall * 100).toFixed(2) + '%';
                    document.getElementById('mF1').textContent = (m.f1 * 100).toFixed(2) + '%';
                    document.getElementById('mAUC').textContent = m.roc_auc.toFixed(4);
                    document.getElementById('mFPR').textContent = (m.false_positive_rate * 100).toFixed(2) + '%';
                }

                const tbody = document.getElementById('alertsBody');
                if (data.recent_frauds && data.recent_frauds.length > 0) {
                    tbody.innerHTML = data.recent_frauds.slice().reverse().map(f => `
                        <tr>
                            <td>${new Date(f.detected_at * 1000).toLocaleTimeString()}</td>
                            <td>$${f.amount.toFixed(2)}</td>
                            <td>${f.user_id}</td>
                            <td>${f.merchant_id}</td>
                            <td>${(f.fraud_probability * 100).toFixed(1)}%</td>
                            <td>${f.latency_ms.toFixed(1)}ms</td>
                            <td><span class="fraud-badge">FRAUD</span></td>
                        </tr>
                    `).join('');
                }

                const badge = document.getElementById('statusBadge');
                if (data.total_processed > 0) {
                    badge.className = 'status-badge status-live';
                    badge.innerHTML = '<span class="live-dot"></span>LIVE';
                }
            } catch (e) {
                const badge = document.getElementById('statusBadge');
                badge.className = 'status-badge status-offline';
                badge.textContent = 'OFFLINE';
            }
        }

        fetchStats();
        setInterval(fetchStats, 1500);
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/api/stats")
async def get_stats():
    result = {
        "total_processed": stats["total_processed"],
        "total_fraud": stats["total_fraud"],
        "total_normal": stats["total_normal"],
        "avg_latency_ms": stats["avg_latency_ms"],
        "recent_frauds": stats["recent_frauds"][-20:],
        "start_time": stats["start_time"],
        "model_metadata": _metadata,
    }
    return result


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": time.time(), "simulator_active": _sim_running}


def main():
    port = int(os.environ.get("PORT", "3000"))
    print(f"Starting monitoring dashboard on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
