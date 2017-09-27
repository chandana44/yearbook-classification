#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters; Please provide <type: test|valid> <model_architecture: alexnet|vgg16|resnet> <load_saved_model: True|False> <checkpoint_file_name> <use_pretraining: True|False> <fine_tuning_method: end-to-end|phase-by-phase> as the parameter;"
    exit 1
fi

type=$1
architecture=$2
load_saved_model=$3
checkpoint_file_name=$4
use_pretraining=$5
fine_tuning_method=$6

module load cuda/8.0

cd src/
python -u customgrade.py --DATASET_TYPE=yearbook --type=$type --model_architecture=$architecture --load_saved_model=$load_saved_model --checkpoint_file_name=$checkpoint_file_name --use_pretraining=$use_pretraining --fine_tuning_method=$fine_tuning_method
