# Real-Time Anomaly Detection System

![Python](https://img.shields.io/badge/python-3.10-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Build](https://img.shields.io/badge/build-passing-brightgreen)

[🔗 Live Demo ⬅️  ](https://real-time-anomaly-detection.onrender.com)

---

## Overview

The **Real-Time Anomaly Detection System** is designed to **detect unusual patterns in streaming data** using machine learning models.  
It integrates **Kafka for real-time data streaming** and a **monitoring dashboard** for visualization.  
This system is ideal for **IoT, finance, cybersecurity, or any application where early anomaly detection is critical**.

---

## Features

- **Real-Time Data Streaming**: Continuous data processing using Apache Kafka.  
- **Anomaly Detection Models**: Detects outliers and unusual patterns using ML algorithms.  
- **Interactive Monitoring Dashboard**: Displays anomalies, trends, and alerts in real-time.  
- **Flexible Deployment**: Can run as a standalone service or integrate into existing systems.  
- **Hybrid Architecture**: Supports local or cloud deployment.  

---

## Tech Stack

- **Backend**: Python, Flask / FastAPI  
- **Streaming**: Apache Kafka  
- **Machine Learning**: Scikit-learn, PyTorch / TensorFlow  
- **Frontend / Dashboard**: React.js / Dash / Plotly  
- **Database**: PostgreSQL / MongoDB  

---

## Installation

### 1. Clone the repository
git clone https://github.com/samira-nooreen/real-time-anomaly-detection.git
cd real-time-anomaly-detection

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run Kafka (for local streaming)
### Make sure Kafka binaries are available in the 'bin/' folder
bin/zookeeper-server-start.sh config/zookeeper.properties &
sleep 5
bin/kafka-server-start.sh config/server.properties &

### 4. Run the backend
python app.py &

### 5. Run the frontend dashboard
cd dashboard
npm install
npm start &

### 6. Access the application
echo "Open your browser and go to http://localhost:3000"


