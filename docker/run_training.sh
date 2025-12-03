#!/bin/bash
echo "🚀 Running training in Docker..."
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/logs:/app/logs \
  financial-stress-model3:latest
echo "✅ Training complete!"
