#!/bin/bash
echo "📊 Starting MLflow UI..."
echo "🌐 Access at: http://localhost:5000"
echo "Press Ctrl+C to stop"
docker run --rm \
  -p 5000:5000 \
  -v $(pwd)/mlruns:/app/mlruns \
  financial-stress-model3:latest \
  mlflow ui --host 0.0.0.0 --port 5000
