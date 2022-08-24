#!/bin/bash
jupyter-nbconvert search.ipynb --to python
chmod +x search.py
python3 search.py $1
