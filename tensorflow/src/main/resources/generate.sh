#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
PYTHONPATH=../../../../python python ../../../../python/scripts/h2o_deepwater_generate_models.py
