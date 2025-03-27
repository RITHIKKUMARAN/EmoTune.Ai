#!/bin/bash
# Update pip
pip install --upgrade pip
# Clear pip cache
pip cache purge
# Install opencv-python-headless first
pip install opencv-python-headless==4.9.0.80
# Explicitly uninstall opencv-python if present
pip uninstall -y opencv-python opencv-contrib-python || true
# Test OpenCV import
python -c "import cv2; print('OpenCV version:', cv2.__version__)" || { echo "OpenCV import failed"; exit 1; }
# Install mediapipe without dependencies
pip install mediapipe==0.10.21 --no-deps
# Install torch CPU version explicitly
pip install torch==2.1.0 --extra-index-url https://download.pytorch.org/whl/cpu
# Force reinstall transformers
pip install --force-reinstall transformers==4.49.0
# Install remaining requirements
pip install -r requirements.txt
# Log installed packages for debugging
pip list > installed_packages.txt
