#!/bin/bash
# Update pip
pip install --upgrade pip
# Clear pip cache
pip cache purge
# Install opencv-python-headless and opencv-contrib-python-headless first
pip install opencv-python-headless==4.8.0.74 opencv-contrib-python-headless==4.8.0.74
# Explicitly uninstall opencv-python and opencv-contrib-python if present
pip uninstall -y opencv-python opencv-contrib-python || true
# Test OpenCV import
python -c "import cv2; print('OpenCV version:', cv2.__version__)" || { echo "OpenCV import failed"; exit 1; }
# Force reinstall transformers
pip install --force-reinstall transformers==4.49.0
# Install remaining requirements
pip install -r requirements.txt
# Log installed packages for debugging
pip list > installed_packages.txt
