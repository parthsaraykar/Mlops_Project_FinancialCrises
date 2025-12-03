#!/bin/bash
echo "🐳 Building Docker image..."
docker build -t financial-stress-model3:latest -f docker/Dockerfile.train .
echo "✅ Build complete!"
docker images | grep financial-stress
