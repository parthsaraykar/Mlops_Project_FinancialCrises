#!/bin/bash
echo "Starting Airflow..."
export AIRFLOW_HOME=~/airflow

# Start webserver in background
airflow webserver --port 8080 > logs/webserver.log 2>&1 &
WEBSERVER_PID=$!
echo "Webserver started (PID: $WEBSERVER_PID)"

# Start scheduler in background
airflow scheduler > logs/scheduler.log 2>&1 &
SCHEDULER_PID=$!
echo "Scheduler started (PID: $SCHEDULER_PID)"

echo ""
echo "✓ Airflow is running!"
echo "  Webserver: http://localhost:8080"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "To stop: ./stop_airflow.sh"
