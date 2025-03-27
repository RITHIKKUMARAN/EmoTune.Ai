#!/bin/bash
# Update pip
pip install --upgrade pip
# Clear pip cache
pip cache purge
# Install opencv-python-headless and opencv-contrib-python-headless first
pip install opencv-python-headless==4.10.0.84 opencv-contrib-python-headless==4.10.0.84
# Explicitly uninstall opencv-python and opencv-contrib-python if present
pip uninstall -y opencv-python opencv-contrib-python || true
# Force reinstall transformers
pip install --force-reinstall transformers==4.49.0
# Install remaining requirements
pip install -r requirements.txt
