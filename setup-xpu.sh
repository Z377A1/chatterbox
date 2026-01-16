# 1. Fix System Drivers (Handle the broken libigc/libze conflicts)
sudo apt update
sudo apt install -y --allow-downgrades \
    intel-level-zero-gpu level-zero libze1 libze-dev \
    intel-opencl-icd intel-gmmlib

# 2. Add your user to the hardware access groups
sudo usermod -aG render $USER
sudo usermod -aG video $USER

# 3. Fix Python Environment dependencies
uv pip install setuptools

# 4. Set Environment Variables for Intel GPU
# This ensures PyTorch uses the newer libraries bundled in your .venv
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH
export PYTHONWARNINGS="ignore"

echo "Setup complete. Please run 'newgrp render' before starting your python script."