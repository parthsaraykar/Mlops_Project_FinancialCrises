#!/bin/bash
# ===========================================
# FIX ENVIRONMENT FOR MLOps FINANCIAL CRISES PROJECT
# ===========================================

echo "🔧 Fixing NumPy/PyTorch compatibility issue..."

# Step 1: Remove any old environment
if [ -d "venv_snorkel" ]; then
    echo "🗑️  Removing old virtual environment..."
    rm -rf venv_snorkel
fi

# Step 2: Create a clean environment
echo "📦 Creating new virtual environment..."
python3 -m venv venv_snorkel
source venv_snorkel/bin/activate

# Step 3: Upgrade pip and tools
pip install --upgrade pip setuptools wheel

# Step 4: Install compatible NumPy first
echo "📦 Installing NumPy 1.24.4..."
pip install numpy==1.24.4

# Step 5: Install PyTorch versions compatible with NumPy 1.x
echo "📦 Installing PyTorch 2.0.1 (CPU build)..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# Step 6: Install Snorkel and remaining dependencies
echo "📦 Installing Snorkel and dependencies..."
pip install snorkel==0.9.9 pandas==2.0.3 scikit-learn==1.3.0 pyyaml==6.0.1

# Step 7: Verify installation
echo ""
echo "✅ Verifying installations..."
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import snorkel; print(f'Snorkel: {snorkel.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"

# Step 8: Test integration
echo ""
echo "🧪 Testing NumPy-PyTorch integration..."
python - << 'PYCODE'
import numpy as np
import torch
arr = np.array([1, 2, 3])
tensor = torch.from_numpy(arr)
print("✅ NumPy-PyTorch integration works!")
print("   NumPy array:", arr)
print("   PyTorch tensor:", tensor)
PYCODE

echo ""
echo "✅ Environment setup complete!"
echo "🚀 Next steps:"
echo "   1. source venv_snorkel/bin/activate"
echo "   2. python validate_setup.py"
echo "   3. python src/models/snorkel_labeling.py"
