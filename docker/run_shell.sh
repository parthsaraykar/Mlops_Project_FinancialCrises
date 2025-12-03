#!/bin/bash
echo "🔧 Opening Docker shell..."
docker run -it --rm \
  -v $(pwd):/app \
  financial-stress-model3:latest \
  /bin/bash
