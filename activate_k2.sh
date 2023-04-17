#!/bin/bash

K2_ROOT=$PWD

export PYTHONPATH=$K2_ROOT/k2/python:$K2_ROOT/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/workspace/zhanghui/k2/k2/venv/lib/python3.7/site-packages/paddle/libs/:/workspace/zhanghui/k2/k2/venv/lib/python3.7/site-packages/paddle/fluid/

