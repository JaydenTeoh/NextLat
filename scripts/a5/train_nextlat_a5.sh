#!/bin/bash

# we recommend using single device because we notice inconsistent performance when using multiple GPUs
fabric run --precision bf16-mixed train.py --config config/a5/nextlat_a5.yaml