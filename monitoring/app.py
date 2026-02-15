"""
FastAPI Monitoring Dashboard — Real-time metrics for the anomaly detection system.
Provides REST API + HTML dashboard showing throughput, fraud rate, latency, recent alerts.
"""

import json
import time
import os
import threading

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Anomaly Detection Monitor")

# Stats shared with consumer (import or read from file)
# In Docker, consumer writes stats to a shared volume file
STATS_FILE = os.environ.get("STATS_FILE", "/tmp/anomaly_stats.json")

# Fallback in-memory stats for standalone mode
_fallback_stats = {
    "total_processed": 0,
    "total_fraud": 0,
    "total_normal": 0,
    "avg_latency_ms": 0.0,
    "recent_frauds": [],
    "start_time": time.time(),
}


def read_stats():
    try:
        with open(STATS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return _fallback_stats


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
        .header h1 {
            font-size: 24px;
            font-weight: 600;
            color: #fff;
        }
        .header .subtitle {
            color: #888;
            font-size: 14px;
            margin-top: 4px;
        }
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
        .metric-value {
            font-size: 32px;
            font-weight: 700;
            color: #fff;
        }
        .metric-value.fraud { color: #ff4444; }
        .metric-value.success { color: #00ff88; }
        .metric-value.warning { color: #ffaa00; }
        .metric-value.info { color: #4488ff; }
        .section-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            color: #fff;
        }
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
        .no-data {
            text-align: center;
            padding: 40px;
            color: #444;
            font-size: 14px;
        }
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

                // Model info
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

                // Alerts
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
        setInterval(fetchStats, 2000);
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/api/stats")
async def get_stats():
    stats = read_stats()
    # Also load model metadata if available
    meta_path = os.path.join(os.path.dirname(__file__), "..", "consumer", "model_metadata.json")
    try:
        with open(meta_path) as f:
            stats["model_metadata"] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return stats


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


def main():
    port = int(os.environ.get("PORT", "8000"))
    print(f"Starting monitoring dashboard on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
