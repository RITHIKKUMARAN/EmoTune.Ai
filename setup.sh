#!/bin/bash
pip install --upgrade pip
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0
pip cache purge
# Force reinstall transformers
pip install --force-reinstall transformers==4.49.0
