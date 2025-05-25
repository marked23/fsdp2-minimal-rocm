#!/usr/bin/env bash

OMP_NUM_THREADS=4 torchrun --nproc_per_node 2 train.py
