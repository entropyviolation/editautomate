#!/bin/bash
# Launch EditAutomate (colon in parent path prevents venv — use system/python pip)
cd "$(dirname "$0")"
python3 main.py
