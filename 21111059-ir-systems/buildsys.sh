#!/bin/bash
jupyter-nbconvert build_system.ipynb --to python
chmod +x build_system.py
python3 build_system.py
