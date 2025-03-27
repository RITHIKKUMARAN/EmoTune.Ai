#!/bin/bash
pip install --upgrade pip
pip cache purge
pip install --force-reinstall transformers==4.49.0
pip install -r requirements.txt
