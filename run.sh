#!/bin/bash

module load cuda/8.0

cd src/
python -u customgrade.py --DATASET_TYPE=yearbook --type=valid
